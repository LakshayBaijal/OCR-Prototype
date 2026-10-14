"""Direct image -> structured JSON via the local MLX-served VLM (Gemma-3) - the same job
Gemini does in `ondc-intelligence`, running for free on local hardware instead of a paid
API. See ocr/local_vlm.py:run_structured() for the actual model call, and the plan this
implements for the full rationale (classical OCR + regex + LayoutLMv3 alone cannot compete
with a model that actually looks at the image).

This is a SEPARATE extraction path from stage2/heuristic.py + stage2/layoutlm.py, not a
replacement of them: the VLM's direct answer is the PRIMARY source (it reasons over the
whole image, unlike regex-over-garbled-OCR-text), and the existing heuristic+LayoutLM
pipeline runs as a FALLBACK that only fills fields the VLM left null - see
extract_product_direct()'s merge step. The one deliberate exception is barcode-decoded UPC,
which stays the single most trustworthy signal in the whole system (checksum-verified) and
is allowed to override the VLM's guess, exactly like it already does in the heuristic path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ocr import local_vlm
from preprocessing import load_rgb

from . import heuristic
from .pipeline import collect_lines_and_barcode
from .schema import GT_FIELD_ORDER, ProductExtraction

ROOT = Path(__file__).parent.parent
SCHEMA_PATH = ROOT.parent / "Image to OCR to Json Export" / "DigiCatalog_groceries_schema_format.json"

# GT-shaped keys -> the internal ProductExtraction key stage2.heuristic already uses for the
# SAME field, so a VLM guess and a heuristic guess compete/merge on ONE key via
# ProductExtraction.set() rather than silently living under two different names. Fields not
# listed here already use the identical GT-style key internally (heuristic.py's newer
# extractors were written to match schema.to_ground_truth_dict()'s key names directly).
_GT_TO_INTERNAL_KEY = {
    "Product_Name": "Product Name",
    "Country_Of_Origin": "Country Of Origin",
    "UPC_or_EAN": "Upc / Ean",
    "FSSAI_no": "Fssai No",
}

# Fields never worth asking the VLM to fill: catalog/eval metadata no product PHOTO could
# ever answer (Category is a constant, Sub_Category/Images/Variants are structurally this
# pipeline's job to leave alone - see schema.to_ground_truth_dict's own docstring).
_SKIP_FIELDS = {"Category", "Sub_Category", "Images", "Variants"}

_SYSTEM_PROMPT = (
    "You extract ONE structured product record from the attached product image(s) - there "
    "may be several images (front, back, nutrition panel) of the SAME product; use all of "
    "them together, since different facts sit on different faces of the pack.\n"
    "Fill a field ONLY when the image(s) actually show it; otherwise set it to null. Never "
    "guess, infer from world knowledge, or auto-complete - most of all codes, numbers, dates "
    "and country of origin. Copy numbers, units, codes, and long text fields (Ingredients, "
    "Nutritional_Details, Usage_Details, Storage_Instructions, Product_Description, "
    "Bullet_Point) EXACTLY as printed, not paraphrased.\n"
    "\n"
    "FIELD-CONFUSION RULES - these specific mix-ups are common, watch for them:\n"
    "- Manufacturer is the legal company that MAKES or MARKETS the product, printed in fine "
    "print (usually starting 'Manufactured by:', 'Marketed by:', or similar), e.g. 'Pushpam "
    "Foods & Beverages Pvt. Ltd.' or 'Hindustan Unilever Ltd.'. It is NEVER the Brand name, "
    "and NEVER a celebrity/athlete/endorser's name printed as part of marketing copy (e.g. "
    "if a cricketer's name and signature appear as an endorsement, that is NOT the "
    "Manufacturer - leave Manufacturer null unless a real company name is separately printed).\n"
    "- Net_Quantity + UOM is the TOTAL contents of THIS package as sold (e.g. '330 ml', "
    "'400 g'), usually printed near the barcode or on the front of pack as the pack size. It "
    "is NEVER a per-serving number from the nutrition/serving-size line (e.g. 'Serving Size: "
    "30 g' or 'Per 100 ml' is NOT the Net_Quantity - those describe a portion, not the whole "
    "package). If you cannot find a total-package quantity distinct from serving-size "
    "numbers, leave Net_Quantity, UOM, and Capacity null rather than reusing a serving-size "
    "figure.\n"
    "- Pack_Size and Pack_Quantity describe MULTI-UNIT packs (e.g. 'Pack of 6' -> "
    "Pack_Quantity 6). They are NEVER a nutritional value or calorie count - if you're about "
    "to put something like a kcal number in Pack_Size, that is wrong; leave it null instead. "
    "A '13 servings per pack' or 'servings per container' line on a nutrition panel is NOT "
    "Pack_Quantity either - that describes how many servings the contents divide into, not "
    "how many physical units are in the box. Leave Pack_Quantity null unless the pack "
    "explicitly says something like 'Pack of N' or shows N distinct physical units.\n"
    "- Bullet_Point is ONLY short marketing claims/callouts about the product itself (e.g. "
    "'Zero Sugar', 'No Added Preservatives', 'High Protein'). It is NEVER a licence/FSSAI "
    "number, a company name, or an address fragment. BE THOROUGH: scan every face of the "
    "pack for every distinct short claim, badge, stat callout (e.g. 'High Caffeine (99mg/330ml "
    "Can)'), usage tip (e.g. 'Tastes best when chilled'), and hashtag/slogan (e.g. "
    "'#PLAYTHEDAY') - most products carry somewhere between 3 and 8 of these, not 1-2. Missing "
    "one is worse than listing one extra genuine claim; only skip a phrase if it is a near-"
    "duplicate of one you already listed.\n"
    "- Mineral_Source is ONLY for a product explicitly MARKETED as a source of that mineral "
    "(e.g. a bottled water's front-of-pack 'Rich in Magnesium' claim). It is NEVER inferred "
    "just because a mineral compound (e.g. 'Magnesium Carbonate') appears somewhere in the "
    "Ingredients list - an ingredient is not a marketing claim. Leave it null unless the pack "
    "itself calls the mineral out as a feature.\n"
    "- Caffeine_Content is ONLY the short caffeine amount as printed (e.g. '99mg/330ml Can', "
    "'80 mg'), never the surrounding marketing sentence or badge text around it (that fuller "
    "phrase belongs in Bullet_Point instead, not duplicated here).\n"
    "- Recommended_Age covers BOTH a baby/child suitability age AND an adult minimum-age / "
    "age-gate warning (e.g. energy drinks or supplements often print '18+' or 'not "
    "recommended for children' near the caution text) - check caution/warning text for this, "
    "not only dedicated baby-product age labelling.\n"
    "\n"
    "CASING: for SHORT categorical fields (Brand, Product_Name, Country_Of_Origin, "
    "Flavours_or_Spices, Diet_Type, Caffeine_Content, Mineral_Source, and similar single-"
    "value fields), normalize to natural title/sentence case (e.g. 'India', not 'INDIA'; "
    "'Mixed Fruit Flavour', not 'MIXED FRUIT FLAVOUR') even if the pack prints them in all "
    "caps - packaging typography is not the same as correct casing. For LONG free-text "
    "fields (Ingredients, Nutritional_Details, Usage_Details, Storage_Instructions, "
    "Product_Description, Bullet_Point), keep the printed casing/wording verbatim instead - "
    "do not paraphrase or re-case those.\n"
    "Read brand/product names carefully for embedded digits or stylized characters (e.g. "
    "'one8', 'Mi3') - do not drop or substitute a digit that's part of the name.\n"
    "\n"
    "For a Yes/No field (Preservatives, Ready_to_cook, Ready_to_eat, Herbal_or_Ayurvedic): "
    "only answer 'Yes' or 'No' when the pack's printed text actually states it either way. "
    "If the pack is silent on it, use null - do not default to 'No' as a guess just because "
    "the claim isn't printed; 'not shown' and 'shown as No' are different things.\n"
    "Bullet_Point is a JSON array of short printed claims/callouts (e.g. 'No Added "
    "Preservatives', 'Zero Sugar') - use [] if none are shown, not null.\n"
    "Respond with ONLY the JSON object - no commentary, no markdown code fence, no "
    "explanation before or after it."
)


def _load_schema_description() -> str:
    """The field-name/type/description table from the sibling ONDC schema file - almost the
    exact same field list as GT_FIELD_ORDER (same project, same schema lineage), so it's
    reused rather than re-typed. Falls back to a bare field list if the file has moved."""
    try:
        raw = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        lines = []
        for name, spec in raw.get("properties", {}).items():
            if name in _SKIP_FIELDS:
                continue
            desc = spec.get("description", "")
            enum = spec.get("enum")
            extra = f" One of: {enum}." if enum else ""
            lines.append(f"- {name} ({spec.get('type', 'string')}): {desc}{extra}")
        return "\n".join(lines)
    except (OSError, json.JSONDecodeError):
        return "\n".join(f"- {f}" for f in GT_FIELD_ORDER if f not in _SKIP_FIELDS)


def _build_user_prompt() -> str:
    return (
        "Extract one product record as a JSON object with EXACTLY these fields:\n\n"
        + _load_schema_description()
        + "\n\nRespond with only the JSON object."
    )


# Prompt instructions alone don't reliably stop the model from putting FSSAI/licence
# numbers and address fragments into Bullet_Point (seen repeatedly in testing, on both the
# direct and two-call paths) - this is a hard filter, not another prompt tweak, since a
# small model's compliance with "don't do X" instructions is inherently unreliable.
_NOT_A_BULLET_RE = re.compile(
    r"\b(?:lic\.?\s*no\.?|fssai|licence\s*no|license\s*no)\b|\d{10,}", re.I
)


def _looks_like_bullet(text: str) -> bool:
    """A real marketing bullet is a short claim, not a licence number or address fragment
    (which the model sometimes emits into Bullet_Point despite being told not to)."""
    if _NOT_A_BULLET_RE.search(text):
        return False
    return len(text.split()) <= 12  # a genuine callout is short; a leaked address/clause isn't


def _apply_vlm_dict(extraction: ProductExtraction, vlm_dict: dict, confidence: float = 0.75) -> None:
    """Load the VLM's direct-JSON answer into `extraction` as the PRIMARY source (see module
    docstring) - every non-null field gets the same flat confidence, since a VLM doesn't
    expose a real per-field score the way RapidOCR's per-line confidence does."""
    net_qty = vlm_dict.get("Net_Quantity")
    uom = vlm_dict.get("UOM")
    if net_qty is not None:
        combined = f"{net_qty} {uom}".strip() if uom else str(net_qty)
        extraction.set("Net Quantity", combined, confidence, "vlm_direct")

    for bullet in (vlm_dict.get("Bullet_Point") or []):
        bullet = str(bullet).strip()
        if bullet and _looks_like_bullet(bullet):
            extraction.add_bullet(bullet)

    handled = _SKIP_FIELDS | {"Net_Quantity", "UOM", "Bullet_Point"}
    for key, value in vlm_dict.items():
        if key in handled:
            continue
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        internal_key = _GT_TO_INTERNAL_KEY.get(key, key)
        extraction.set(internal_key, str(value), confidence, "vlm_direct")


def _merge_fallback(extraction: ProductExtraction, image_paths: list[str]) -> None:
    """The heuristic+LayoutLM pipeline as a FALLBACK, in place: only fills fields
    `extraction` left blank, or overrides one only when markedly more confident (reusing
    ProductExtraction.set()'s own rule) - so a checksum-verified barcode UPC (confidence 1.0)
    still wins over a VLM guess (0.75), while a weaker heuristic/LayoutLM signal does not
    silently clobber a real VLM answer. Shared by both extract_product_direct() and
    extract_product_two_call(), which differ only in how `extraction` got its PRIMARY values.
    """
    layout_lines, text_lines, bc, _ = collect_lines_and_barcode(image_paths)
    fallback = heuristic.parse(text_lines, layout_lines, bc)
    for name, f in fallback.fields.items():
        extraction.set(name, f.value, f.confidence, f.source)
    # heuristic.py's own keyword-based bullet detector has the identical licence-number-
    # leakage problem the VLM's Bullet_Point does (see _looks_like_bullet) - filter its
    # output here too, or a "clean" VLM result gets re-contaminated by the fallback merge.
    for b in fallback.bullets:
        if _looks_like_bullet(b):
            extraction.add_bullet(b)

    from . import layoutlm

    if layoutlm.available():
        layoutlm.merge_into(extraction, layout_lines)


def extract_product_direct(
    image_paths: list[str], use_fallback: bool = True,
) -> tuple[ProductExtraction, dict | None]:
    """The VLM's direct answer is the PRIMARY source; see _merge_fallback() for the
    fallback/barcode-override rules.

    Returns (extraction, vlm_dict) - `vlm_dict` is the VLM's raw parsed response, or None if
    the call failed outright (server unreachable, every retry timed out, or every retry's
    response was unparseable). This is deliberately surfaced rather than swallowed: a caller
    that only sees the merged `extraction` cannot tell "the VLM ran and legitimately found
    nothing for these fields" apart from "the VLM call failed and this is fallback-only" -
    those are very different situations and app.py's Stage 2 tab needs to show which one
    happened, not present both as the same silent result.

    Never raises: if the MLX server isn't reachable or every retry in run_structured() fails,
    this falls through to fallback-only (or an empty extraction if `use_fallback=False`) so a
    caller can always call .to_ground_truth_dict() on the result.
    """
    rgb_images = [load_rgb(p) for p in image_paths]
    vlm_dict = local_vlm.run_structured(rgb_images, _SYSTEM_PROMPT, _build_user_prompt())

    extraction = ProductExtraction()
    if vlm_dict:
        _apply_vlm_dict(extraction, vlm_dict)
    if use_fallback:
        _merge_fallback(extraction, image_paths)

    return extraction, vlm_dict


def extract_product_two_call(
    image_paths: list[str], use_fallback: bool = True,
) -> tuple[ProductExtraction, dict | None]:
    """The two-call alternative to extract_product_direct(): OCR each image separately
    first (ocr.local_vlm.run() - the plain line-by-line OCR call, not the structured one),
    concatenate the results into one labelled text blob, then run a SEPARATE mapping-only
    call (ocr.local_vlm.run_map_from_text()) over that text - no images in the mapping call
    at all. Turns one hard joint task (see the pixels AND structure them at once) into two
    easier sequential ones (read; then read-and-reason-over-text), which the earlier eval
    doc found scores close to direct extraction while being cheaper/faster per call.

    Same PRIMARY-source-plus-fallback merge as extract_product_direct() (see
    _merge_fallback()) and the same (extraction, vlm_dict) return contract.

    Known tradeoff, not a bug: the mapping call never sees pixels, so purely visual (non-
    text) cues are invisible to it - most notably Diet_Type's veg/non-veg dot symbol, which
    will be null more often here than via the direct one-call path.
    """
    rgb_images = [load_rgb(p) for p in image_paths]
    ocr_results = [local_vlm.run(rgb) for rgb in rgb_images]
    text_blob = "\n\n".join(
        f"--- Image {i + 1} ---\n" + result.text
        for i, result in enumerate(ocr_results) if result.text.strip()
    )

    vlm_dict = None
    if text_blob.strip():
        vlm_dict = local_vlm.run_map_from_text(text_blob, _SYSTEM_PROMPT, _build_user_prompt())

    extraction = ProductExtraction()
    if vlm_dict:
        _apply_vlm_dict(extraction, vlm_dict)
    if use_fallback:
        _merge_fallback(extraction, image_paths)

    return extraction, vlm_dict
