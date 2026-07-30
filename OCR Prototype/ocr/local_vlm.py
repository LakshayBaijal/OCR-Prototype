"""Tier 2 - local VLM, served by MLX. FREE (runs on this machine's GPU), no API, no tokens.

Talks to an `mlx_vlm.server` over HTTP rather than importing the model in-process. That is
deliberate, and it is the difference between a toy and something usable:

  * The weights load ONCE, into the server, and stay resident. Importing mlx_vlm directly
    meant every script paid a ~130s cold load, and every process carried 3-4GB of weights
    in its own memory - which is how the editor got starved and crashed.
  * The server keeps a vision cache and quantised KV cache across requests, so repeat
    calls are far cheaper than a fresh in-process generate().
  * Our benchmark, the Streamlit app and the router all share one loaded model instead of
    each spawning their own copy.

Start it (once, in its own terminal):

    venv/bin/python -m mlx_vlm.server \
        --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit \
        --port 8080 --kv-bits 8 --max-tokens 512

If the server is not running, `available()` is False and the router simply skips this tier -
it never blocks the free RapidOCR path.
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from .types import OCRLine, OCRResult

HOST = "http://127.0.0.1:8080"

# The model to REQUEST. mlx_vlm.server supports several cached models and lazy-swaps
# between them per-request (unload the current one, load the requested one) - so this is
# not "whatever the server happened to start with", it is a genuine choice, and changing it
# is enough to switch models with zero other code changes.
#
# The entire Qwen-VL family is BROKEN under mlx_vlm.server: both Qwen2.5-VL-3B and
# Qwen3-VL-4B crash mid-generation with "RuntimeError: There is no Stream(gpu, N) in current
# thread", from the identical get_rope_index() code path in qwen2_5_vl/language.py and
# qwen3_vl/language.py - a shared bug in mlx-vlm's threading model, not one model version.
# Confirmed on both; do not retry Qwen-VL here until mlx-vlm fixes it upstream.
# Gemma-3 is the only VLM family confirmed to serve correctly on this setup.
MODEL = "mlx-community/gemma-3-4b-it-4bit"

PROMPT = (
    "Read all text printed on this product package. "
    "Output only the text you see, one item per line. No commentary."
)


def served_model() -> str | None:
    """The model this adapter will REQUEST (not "whatever the server advertises" - /v1/models
    lists every model ever cached to disk, since mlx_vlm.server supports lazy-swapping
    between several; it is not a live "currently loaded" indicator). Returns None only if
    the server itself is unreachable.
    """
    try:
        with urllib.request.urlopen(f"{HOST}/v1/models", timeout=2) as r:
            json.loads(r.read())  # just prove the server responds; ignore its contents
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    return MODEL


def available() -> bool:
    """True only if the MLX server is actually up. No server, no tier - never an exception."""
    return served_model() is not None


def _data_uri(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def run(rgb: np.ndarray) -> OCRResult:
    if not available():
        raise RuntimeError(
            "MLX server is not running. Start it with:\n"
            "  venv/bin/python -m mlx_vlm.server "
            f"--model {MODEL} --port 8080 --kv-bits 8 --max-tokens 512"
        )

    payload = {
        "model": served_model() or MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _data_uri(rgb)}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        # Greedy decoding makes this 4-bit model degenerate - on one pack it emitted
        # '75555555...' until it hit the token cap, on another just '1'. A little temperature
        # plus a repetition penalty breaks those loops; the same pack then reads back
        # correctly as "Limca Lime 'n' Lemon".
        "temperature": 0.3,
        "repetition_penalty": 1.05,
        "max_tokens": 512,
    }

    req = urllib.request.Request(
        f"{HOST}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    elapsed = time.perf_counter() - t0

    text = body["choices"][0]["message"]["content"].strip()
    h, w = rgb.shape[:2]

    # A VLM returns prose, not geometry: we know WHAT it read, not WHERE. One box for the
    # whole crop is the honest answer - and it is why this tier cannot replace RapidOCR,
    # which is our only source of per-line boxes and confidences.
    lines = [
        OCRLine(text=t, bbox=[(0, 0), (w, 0), (w, h), (0, h)], confidence=None)
        for t in text.splitlines() if t.strip()
    ]
    return OCRResult(lines=lines, engine="local_vlm", elapsed=elapsed, cost_usd=0.0)


def _strip_fence(text: str) -> str:
    """The model sometimes wraps JSON in a ```json fence despite being asked not to - strip
    it before parsing rather than failing on something trivially recoverable."""
    fenced = text.strip()
    if fenced.startswith("```"):
        fenced = fenced.split("\n", 1)[1] if "\n" in fenced else fenced
        if fenced.endswith("```"):
            fenced = fenced[: -3]
        fenced = fenced.strip()
        if fenced.startswith("json"):
            fenced = fenced[4:].strip()
    return fenced


def _repair_truncated_json(text: str) -> dict | None:
    """Best-effort recovery for a response cut off mid-object (hit max_tokens, or the model
    just stopped early) - trims back to the last complete `"key": value` pair and closes the
    object, rather than discarding an otherwise-good partial record. Returns None if there's
    nothing salvageable (e.g. the text isn't even a JSON object to begin with)."""
    text = text.strip()
    if not text.startswith("{"):
        return None
    # Walk backwards from the end to the last comma that sits after a complete value (a
    # closing quote, brace, bracket, or digit/null/true/false), then close the object there.
    for cut in range(len(text) - 1, 0, -1):
        candidate = text[:cut].rstrip().rstrip(",") + "}"
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and len(parsed) >= 3:  # a handful of fields, not a fluke
            return parsed
    return None


def _chat_json_with_retry(content: list[dict], system: str, max_retries: int) -> dict | None:
    """Shared retry/parse loop behind both run_structured() (images in) and
    run_map_from_text() (text in) - they differ only in what `content` holds, not in how the
    response is retried or parsed.

    Retries on a genuinely empty, non-JSON, or transport-level-failed response (mirrors
    `ondc-intelligence.app.agents.gemini.GeminiClient.generate`'s retry discipline), with the
    temperature nudged UP a little on each retry - a malformed response at temperature 0.2 is
    often the model stuck in a degenerate decoding path, and retrying the IDENTICAL request
    tends to reproduce the same failure; a small temperature bump gives each retry a genuinely
    different chance rather than repeating the same near-miss. If parsing still fails outright,
    `_repair_truncated_json` makes one attempt to salvage a response cut off mid-object before
    giving up on that attempt. Returns None (never raises) only if every attempt fails.
    """
    if not available():
        return None

    for attempt in range(max_retries + 1):
        payload = {
            "model": served_model() or MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": min(0.2 + 0.15 * attempt, 0.7),
            "repetition_penalty": 1.05,
            # A full ~36-field record (Ingredients/Nutritional_Details can run to a paragraph
            # each) needs far more headroom than the 512-token OCR-text prompt above.
            "max_tokens": 3000,
        }
        req = urllib.request.Request(
            f"{HOST}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.loads(resp.read())
            text = body["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError):
            continue  # transport/response-shape failure - worth a retry, same as an empty read

        if not text:
            continue

        fenced = _strip_fence(text)
        try:
            parsed = json.loads(fenced)
        except json.JSONDecodeError:
            parsed = _repair_truncated_json(fenced)
            if parsed is None:
                continue  # nothing salvageable - retry at a higher temperature
        if isinstance(parsed, dict):
            return parsed

    return None


def run_structured(
    rgb_images: list[np.ndarray], system: str, user: str, max_retries: int = 3,
) -> dict | None:
    """Direct image(s) -> structured JSON, the same job Gemini does in `ondc-intelligence` -
    NOT the line-by-line OCR-text contract `run()` above uses. See stage2/vlm_direct.py for
    the caller: it builds `system`/`user` from the target schema and merges this result with
    the heuristic/LayoutLM pipeline as a fallback for whatever comes back null.

    Every image in `rgb_images` is attached to ONE request (front/back/nutrition-panel shots
    of the same product all go in together, the way Gemini receives multiple image_refs),
    since fields live across a product's different faces. See run_map_from_text() below for
    the two-call alternative (OCR each image separately first, then map from the combined
    text alone - no images in the mapping call).
    """
    content = [{"type": "image_url", "image_url": {"url": _data_uri(rgb)}} for rgb in rgb_images]
    content.append({"type": "text", "text": user})
    return _chat_json_with_retry(content, system, max_retries)


def run_map_from_text(text_blob: str, system: str, user: str, max_retries: int = 3) -> dict | None:
    """Structured JSON from TEXT ALONE - no images in this call at all. The two-call
    alternative to run_structured(): OCR each of a product's images separately with run()
    first (per stage2.vlm_direct.extract_product_two_call), concatenate the results into
    `text_blob`, then ask the model to map THAT into the schema, as a "reading comprehension"
    task rather than a "look at pixels and structure them" task. Same system/user prompt as
    run_structured() (the field-confusion rules are about what text means, not about looking
    at pixels, so they transfer as-is) - only the message content differs.

    Known tradeoff, not a bug: a text-only call cannot see genuinely visual (non-text) cues -
    most notably the Diet_Type veg/non-veg dot symbol - so that field will be null more often
    here than in run_structured(). Same retry/repair discipline as run_structured() (see
    _chat_json_with_retry) and the same "never raises, None on total failure" contract.
    """
    content = [{"type": "text", "text": f"{user}\n\nOCR TEXT FROM THE PRODUCT'S IMAGES:\n{text_blob}"}]
    return _chat_json_with_retry(content, system, max_retries)
