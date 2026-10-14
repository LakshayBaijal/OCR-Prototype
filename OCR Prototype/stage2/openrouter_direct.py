"""Direct image -> structured JSON via OpenRouter (openrouter.ai) - a free, ALREADY-HOSTED
alternative for models that either don't fit the local 16GB Mac (Gemma-4 26B-A4B) or hit a
real local mlx-vlm bug (Gemma-4's PLE-quantization / audio-tower issues documented in
commands.txt). No self-hosting, no MLX, no notebook - OpenRouter runs it, we just call it.

The actual HTTP client (auth, retries, JSON repair) lives in ocr/openrouter_vlm.py - the
canonical home shared with Stage 1's OCR-only tier, the same way stage2/vlm_direct.py reuses
ocr/local_vlm.py rather than re-implementing the local MLX client. This file only adds the
schema-aware prompt + fallback-merge layer on top, mirroring stage2/gemini_direct.py exactly.

Needs OPENROUTER_API_KEY. Get a free key:
  1. Sign up at https://openrouter.ai (email or Google account - no credit card, no phone
     verification, unlike some other providers).
  2. https://openrouter.ai/keys -> Create Key.
  3. Put it in a `.env` file in THIS directory (OCR Prototype/.env, one line:
     OPENROUTER_API_KEY=...) - never paste the key into a chat/terminal that gets logged,
     and never commit that file (already in .gitignore, same as GEMINI_API_KEY).

Free-tier models confirmed vision-capable and genuinely $0-cost as of 2026-07 (verified
directly against openrouter.ai/collections/free-models, not just marketing copy) - see
ocr/openrouter_vlm.py's MODEL_CHOICES. Rate limit: 50 requests/day free, 1000/day if $10+
credit has EVER been added to the account (no obligation to spend it) - fine for interactive
testing or a multi-day ground_truth_100 pass, not for high-volume production traffic.
"""

from __future__ import annotations

from ocr.openrouter_vlm import DEFAULT_MODEL, MODEL_CHOICES, available as api_key_available, run_structured

from .pipeline import collect_lines_and_barcode  # noqa: F401  (re-exported for parity/tests)
from .schema import ProductExtraction
from .vlm_direct import _SYSTEM_PROMPT, _apply_vlm_dict, _build_user_prompt, _merge_fallback

__all__ = ["DEFAULT_MODEL", "MODEL_CHOICES", "api_key_available", "extract_product_openrouter"]


def extract_product_openrouter(
    image_paths: list[str], use_fallback: bool = True, model: str = DEFAULT_MODEL,
) -> tuple[ProductExtraction, dict | None]:
    """The OpenRouter analogue of stage2.gemini_direct.extract_product_gemini() - same
    PRIMARY-source-plus-fallback merge (see vlm_direct._merge_fallback), same
    (extraction, raw_dict) return contract, so app.py and export_ground_truth.py treat this
    as a fifth interchangeable extraction mode.
    """
    vlm_dict = run_structured(image_paths, _SYSTEM_PROMPT, _build_user_prompt(), model=model)

    extraction = ProductExtraction()
    if vlm_dict:
        # Between local-VLM's 0.75 and Gemini's 0.9: a real, capable hosted model, but not
        # yet proven on this exact extraction task the way Gemini has been this session.
        _apply_vlm_dict(extraction, vlm_dict, confidence=0.85)
    if use_fallback:
        _merge_fallback(extraction, image_paths)

    return extraction, vlm_dict
