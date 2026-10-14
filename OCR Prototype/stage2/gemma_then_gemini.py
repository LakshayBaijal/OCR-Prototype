"""Gemma first, Gemini fallback-only - the same escalation `ondc-intelligence` built and
measured this session (app/agents/image2product_agent.py's `_call_model`), ported here
since this is where that OCR/VLM logic actually originates.

Reuses everything that already exists rather than reimplementing it: `ocr.local_vlm`'s
retry/repair loop (`_chat_json_with_retry`, unmodified except for the new optional
`response_schema` param), `stage2.gemini_direct`'s Gemini caller (completely unmodified -
it IS the fallback tier), and `stage2.vlm_direct`'s shared prompt/merge helpers. The only
genuinely new thing here is the escalation order and the tier bookkeeping.

Order, on every call:
  1. Gemma, one HTTP attempt only (`max_retries=0` - isolates a clean "first try" signal,
     since `_chat_json_with_retry` has no way to report which of its OWN internal attempts
     succeeded).
  2. Only if (1) returned nothing usable: Gemma again, this time with its own full
     retry/repair loop (`max_retries` at its normal default) - still free, still local,
     no new retry logic, just a second top-level call.
  3. Only if (2) also returned nothing usable: Gemini - the existing, unmodified caller.

Tier is recorded as one of "gemma-first-try" / "gemma-repaired" / "gemini-fallback" -
same three internal values `ondc-intelligence`'s own `model_tier` uses, for consistency
between the two projects' eval reports.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ocr import local_vlm
from preprocessing import load_rgb

from .gemini_direct import DEFAULT_MODEL as GEMINI_DEFAULT_MODEL
from .gemini_direct import run_structured as gemini_run_structured
from .schema import ProductExtraction
from .vlm_direct import _apply_vlm_dict, _build_user_prompt, _merge_fallback, _SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parent.parent
_ONDC_ROOT = ROOT.parent / "ondc-intelligence"
_SCHEMA_PATH = _ONDC_ROOT / "Input" / "Schemas" / "DigiCatalog" / "DigiCatalog_groceries_schema_format.json"


def _build_response_schema() -> dict[str, Any] | None:
    """Reuse ondc-intelligence's own proven JSON-Schema -> OpenAI-response-format
    converter rather than hand-rolling a second one here - same cross-repo-import
    pattern already used this session in ground_truth_analysis/analyze_fields.py, just
    across repos instead of within one. Returns None (degrading to no schema constraint,
    same as every existing caller in this repo) if ondc-intelligence isn't checked out
    at the expected sibling path, rather than raising - this module should still work,
    just without the strict-mode benefit, if that sibling repo ever moves.
    """
    if not _SCHEMA_PATH.exists():
        return None
    try:
        sys.path.insert(0, str(_ONDC_ROOT))
        from app.domain.schema_convert import definition_to_openai_schema, json_schema_to_definition
    except ImportError:
        return None

    raw = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    definition = json_schema_to_definition(raw)
    # nullable_optional=True: nothing here should be pressured as required from a photo
    # that may simply not show it - mirrors image2product_agent.py's own rationale.
    # strict=True: mlx_vlm.server's grammar-constrained decoding needs every property
    # required+nullable to actually hold the model to the schema's real field names -
    # verified directly this session (ondc-intelligence's own README), not assumed.
    return definition_to_openai_schema(definition, nullable_optional=True, strict=True)


# Built once at import time - the schema doesn't change per call.
RESPONSE_SCHEMA = _build_response_schema()


def _attempt(status: str = "not_attempted", result: dict | None = None) -> dict:
    return {"status": status, "dict": result}


def extract_product_gemma_then_gemini(
    image_paths: list[str], use_fallback: bool = True,
) -> tuple[ProductExtraction, dict | None, dict, dict]:
    """Gemma first (one try, then one repair attempt), Gemini only if both fail. Returns
    (extraction, winning_dict, tier_info, attempts).

    tier_info is {"tier": "gemma-first-try"|"gemma-repaired"|"gemini-fallback",
    "attempts": N}, N being how many Gemma attempts were actually made before the
    winning call (1 or 2) - which tier ultimately WON.

    `attempts` is the new part: {"gemma_original": {...}, "gemma_repaired": {...},
    "gemini_fallback": {...}}, each {"status": "success"|"failed"|"not_attempted",
    "dict": dict|None} - what EVERY tier that actually ran produced, not just the
    winner. "not_attempted" (distinct from "failed") means an earlier tier already
    succeeded, so this one never ran at all - e.g. if gemma_original succeeds,
    gemma_repaired and gemini_fallback both stay "not_attempted", not "failed".
    ondc-intelligence itself does NOT expose this level of detail anywhere (it only
    tracks which tier won, for cost tallying) - this is new, built specifically so a
    caller (the Streamlit UI) can show what each stage actually produced.

    Same PRIMARY-source-plus-fallback merge as extract_product_direct()/
    extract_product_gemini() (see vlm_direct._merge_fallback) - calls the LOW-level
    run_structured() functions directly (not the high-level extract_product_direct()/
    extract_product_gemini() wrappers), since those each call _merge_fallback
    internally; calling them here instead of the low-level functions would run the
    heuristic/LayoutLM pass multiple times depending on which tier won.
    """
    rgb_images = [load_rgb(p) for p in image_paths]
    system, user = _SYSTEM_PROMPT, _build_user_prompt()

    attempts = {
        "gemma_original": _attempt(),
        "gemma_repaired": _attempt(),
        "gemini_fallback": _attempt(),
    }

    original_dict = local_vlm.run_structured(
        rgb_images, system, user, max_retries=0, response_schema=RESPONSE_SCHEMA
    )
    attempts["gemma_original"] = _attempt("success" if original_dict is not None else "failed", original_dict)

    if original_dict is not None:
        tier_info = {"tier": "gemma-first-try", "attempts": 1}
        winning_dict, source = original_dict, "gemma"
    else:
        repaired_dict = local_vlm.run_structured(rgb_images, system, user, response_schema=RESPONSE_SCHEMA)
        attempts["gemma_repaired"] = _attempt("success" if repaired_dict is not None else "failed", repaired_dict)

        if repaired_dict is not None:
            tier_info = {"tier": "gemma-repaired", "attempts": 2}
            winning_dict, source = repaired_dict, "gemma"
        else:
            tier_info = {"tier": "gemini-fallback", "attempts": 2}
            gemini_dict = gemini_run_structured(image_paths, system, user, model=GEMINI_DEFAULT_MODEL)
            attempts["gemini_fallback"] = _attempt("success" if gemini_dict is not None else "failed", gemini_dict)
            winning_dict, source = gemini_dict, "gemini"

    extraction = ProductExtraction()
    if winning_dict:
        # Same confidence convention extract_product_direct()/extract_product_gemini()
        # already use for these two sources - no new value invented for
        # "repaired" vs "first-try" Gemma, since neither side has evidence to split them.
        _apply_vlm_dict(extraction, winning_dict, confidence=0.75 if source == "gemma" else 0.9)
    if use_fallback:
        _merge_fallback(extraction, image_paths)

    return extraction, winning_dict, tier_info, attempts
