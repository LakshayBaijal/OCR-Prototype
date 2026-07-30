"""Tier 2 (alternative) - Gemma-4 via openrouter.ai, hosted for free by OpenRouter. Same job
as ocr/local_vlm.py's Gemma-3 (better text than RapidOCR, no geometry), but the model your
senior specifically asked for, which does NOT run under the local `mlx_vlm.server` at all
(see commands.txt - the local Gemma-4 quantized builds hit a real mlx-vlm loading bug).

This is the canonical home for OpenRouter's HTTP client - stage2/openrouter_direct.py (the
Stage 2 structured-extraction mode) imports run_structured()/DEFAULT_MODEL/MODEL_CHOICES from
here rather than duplicating them, the same way stage2/vlm_direct.py imports from THIS
module's local sibling ocr/local_vlm.py. Kept in its own file, never imported BY
ocr/local_vlm.py or vice versa - they are two independent backends (local GPU vs. remote API,
unlimited-but-lower-quality vs. quota-limited-but-Gemma-4), not one blended thing.

*** QUOTA WARNING - READ BEFORE TURNING THIS ON FOR STAGE 1 ***
OpenRouter's free tier is 50 requests/day (1000/day if $10+ credit has ever been added, no
obligation to spend it). Stage 1 calls the VLM tier PER CROP, not per product - a single
product with 5 photos can easily produce 10-15 crops, so "union" mode (every crop, every
time) can burn the ENTIRE daily quota on 3-5 products. Prefer leaving `union_free_tiers` OFF
when using this tier (escalation-only: only called when RapidOCR's own read is already
weak) unless you specifically want to spend the quota testing a small number of products.

Needs OPENROUTER_API_KEY - see stage2/openrouter_direct.py's module docstring for how to get
one (free, no credit card, no phone verification) and where to put it (.env).
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image

from .types import OCRLine, OCRResult

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "google/gemma-4-26b-a4b-it:free"
MODEL_CHOICES = {
    "gemma-4-26b": "google/gemma-4-26b-a4b-it:free",
    "nemotron-vl-12b": "nvidia/nemotron-nano-12b-v2-vl:free",
    "nemotron-omni-30b": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
}
_RETRIES = 2  # keep low - the daily quota is precious, unlike the local-VLM path

PROMPT = (
    "Read all text printed on this product package. "
    "Output only the text you see, one item per line. No commentary."
)


def available() -> bool:
    """True only if OPENROUTER_API_KEY is actually set - matches ocr/local_vlm.py's
    available() naming so router.py can treat both backends the same way."""
    return bool(os.environ.get("OPENROUTER_API_KEY", "").strip())


def _client():
    from openai import OpenAI

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Sign up free at https://openrouter.ai (no card, "
            "no phone verification), create a key at https://openrouter.ai/keys, and put it "
            "in OCR Prototype/.env (OPENROUTER_API_KEY=...) or export it in your shell."
        )
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def _data_uri_from_array(rgb: np.ndarray) -> str:
    """For Stage 1's in-memory crops (Stage 0 already decoded/cropped these to numpy arrays -
    there is no file path to read here, unlike Stage 2's whole-product calls)."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _data_uri_from_path(path: str) -> str:
    """For Stage 2's whole-product calls, which read the ORIGINAL files by path."""
    data = Path(path).read_bytes()
    mime = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _strip_fence(text: str) -> str:
    fenced = text.strip()
    if fenced.startswith("```"):
        fenced = fenced.split("\n", 1)[1] if "\n" in fenced else fenced
        if fenced.endswith("```"):
            fenced = fenced[:-3]
        fenced = fenced.strip()
        if fenced.startswith("json"):
            fenced = fenced[4:].strip()
    return fenced


def _repair_truncated_json(text: str) -> dict | None:
    text = text.strip()
    if not text.startswith("{"):
        return None
    for cut in range(len(text) - 1, 0, -1):
        candidate = text[:cut].rstrip().rstrip(",") + "}"
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and len(parsed) >= 3:
            return parsed
    return None


def run(rgb: np.ndarray, model: str = DEFAULT_MODEL) -> OCRResult:
    """Stage 1's plain OCR-text call - same (rgb) -> OCRResult contract as
    ocr.local_vlm.run(), so router.py can use this tier as a drop-in alternative backend.
    Raises (never silently returns empty) if the key is missing or every retry fails, mirror
    of local_vlm.run()'s own "the caller (router.py) already checked available() first"
    contract.
    """
    if not available():
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set - see ocr/openrouter_vlm.py's module docstring."
        )

    client = _client()
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": _data_uri_from_array(rgb)}},
            {"type": "text", "text": PROMPT},
        ],
    }]

    t0 = time.perf_counter()
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.3, max_tokens=512)
    elapsed = time.perf_counter() - t0

    text = (resp.choices[0].message.content or "").strip()
    h, w = rgb.shape[:2]
    lines = [
        OCRLine(text=t, bbox=[(0, 0), (w, 0), (w, h), (0, h)], confidence=None)
        for t in text.splitlines() if t.strip()
    ]
    return OCRResult(lines=lines, engine="openrouter_vlm", elapsed=elapsed, cost_usd=0.0)


def run_structured(
    image_paths: list[str], system: str, user: str, model: str = DEFAULT_MODEL,
) -> dict | None:
    """Stage 2's structured-JSON call - same (image_paths, system, user) -> dict|None
    contract as ocr.local_vlm.run_structured() / stage2.gemini_direct.run_structured().
    stage2/openrouter_direct.py imports this rather than re-implementing it.

    Retries only _RETRIES times (default 2) - the free tier's 50-requests/day quota is
    precious, unlike the local-VLM path where retries cost nothing. A 429 (rate limit /
    daily quota spent) is NOT retried - that's not transient - and any other API/auth/network
    error also returns None rather than raising, so a caller always falls back cleanly.
    """
    if not available():
        return None

    try:
        client = _client()
    except RuntimeError:
        return None

    content = [{"type": "image_url", "image_url": {"url": _data_uri_from_path(p)}} for p in image_paths]
    content.append({"type": "text", "text": user})
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]

    for _ in range(_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=0.0, max_tokens=3000,
            )
        except Exception as e:  # noqa: BLE001
            if "429" in str(e) or "rate" in str(e).lower():
                return None  # daily quota spent - not transient, don't retry
            time.sleep(1.5)
            continue

        text = (resp.choices[0].message.content or "").strip()
        if not text:
            time.sleep(1.0)
            continue

        fenced = _strip_fence(text)
        try:
            parsed = json.loads(fenced)
        except json.JSONDecodeError:
            parsed = _repair_truncated_json(fenced)
        if isinstance(parsed, dict):
            return parsed

    return None
