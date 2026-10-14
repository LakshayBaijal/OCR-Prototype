"""Direct image -> structured JSON via the REAL Gemini API - not free, but the proven,
production-grade baseline `ondc-intelligence` already uses for this exact job
(app/agents/img2schema_agent.py + app/agents/gemini.py). Reused here to get an honest
quality reference the free/local paths (stage2/vlm_direct.py) can actually be measured
against, before investing further in making the free path good enough on its own.

Needs GEMINI_API_KEY. Get a free-tier key at https://aistudio.google.com/apikey and put it
in a `.env` file in THIS directory (OCR Prototype/.env, one line: GEMINI_API_KEY=...) -
never paste the key into a chat/terminal that gets logged, and never commit that file
(it's in .gitignore already). Nothing here bills or even imports google.genai until a call
is actually attempted - importing this module with no key set is always safe.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from .pipeline import collect_lines_and_barcode  # noqa: F401  (re-exported for parity/tests)
from .schema import ProductExtraction
from .vlm_direct import _SYSTEM_PROMPT, _apply_vlm_dict, _build_user_prompt, _merge_fallback

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

# Same default ondc-intelligence's own .env.example configures for this exact job - a real,
# already-proven setting from that codebase, not a guess.
DEFAULT_MODEL = "gemini-3.1-flash-lite"
EMPTY_RESPONSE_RETRIES = 2


def api_key_available() -> bool:
    """True only if GEMINI_API_KEY is actually set - callers use this to skip straight to
    the heuristic/LayoutLM fallback rather than attempt (and fail) a real API call."""
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())


def _client():
    from google import genai

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get a free-tier key at "
            "https://aistudio.google.com/apikey and put it in OCR Prototype/.env "
            "(GEMINI_API_KEY=...) or export it in your shell."
        )
    return genai.Client(api_key=key)


def run_structured(
    image_paths: list[str], system: str, user: str, model: str = DEFAULT_MODEL,
) -> dict | None:
    """Call the real Gemini API with the image(s) + schema-aware prompt. Same
    (image_paths, system, user) -> dict|None contract as ocr.local_vlm.run_structured(), so
    stage2/vlm_direct.py's prompt-building can be reused as-is and callers can treat this as
    an interchangeable fourth extraction mode.

    Retries only a genuinely EMPTY response (mirrors ondc-intelligence's
    GeminiClient.generate() exactly - that's the one failure mode observed to succeed on
    retry with the same prompt). A malformed-but-present response, missing credentials, or
    a real API/network error are NOT retried - they mean something is actually wrong, not
    transient - and this returns None rather than raising, so a caller always falls back to
    heuristic/LayoutLM cleanly.
    """
    if not api_key_available():
        return None

    from google.genai import types

    try:
        client = _client()
    except RuntimeError:
        return None

    parts = []
    for p in image_paths:
        data = Path(p).read_bytes()
        mime = "image/jpeg" if p.lower().endswith((".jpg", ".jpeg")) else "image/png"
        parts.append(types.Part.from_bytes(data=data, mime_type=mime))
    parts.append(user)

    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        temperature=0.0,
    )

    for attempt in range(EMPTY_RESPONSE_RETRIES + 1):
        try:
            resp = client.models.generate_content(model=model, contents=parts, config=config)
        except Exception:
            return None  # a real API/auth/network error - not the "retry on empty" case

        text = (resp.text or "").strip()
        if not text:
            time.sleep(1.0)
            continue

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None  # malformed JSON from Gemini is a real prompt problem, not transient
        return parsed if isinstance(parsed, dict) else None

    return None


def extract_product_gemini(
    image_paths: list[str], use_fallback: bool = True, model: str = DEFAULT_MODEL,
) -> tuple[ProductExtraction, dict | None]:
    """The real-Gemini analogue of stage2.vlm_direct.extract_product_direct() - same
    PRIMARY-source-plus-fallback merge (see vlm_direct._merge_fallback), same
    (extraction, raw_dict) return contract, so app.py and export_ground_truth.py treat this
    as a fourth interchangeable extraction mode alongside Direct VLM / OCR-then-Map /
    Heuristic+LayoutLM.
    """
    vlm_dict = run_structured(image_paths, _SYSTEM_PROMPT, _build_user_prompt(), model=model)

    extraction = ProductExtraction()
    if vlm_dict:
        # Higher flat confidence than the local-VLM paths (0.75): Gemini is the proven,
        # production-grade baseline here, not the thing being validated.
        _apply_vlm_dict(extraction, vlm_dict, confidence=0.9)
    if use_fallback:
        _merge_fallback(extraction, image_paths)

    return extraction, vlm_dict
