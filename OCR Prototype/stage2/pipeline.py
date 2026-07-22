"""Run the whole stack for one product and hand Stage 2 what it needs.

Stage 0 (preprocess) -> Stage 1 (OCR, per crop) -> barcode (whole product) -> collect every
OCR line WITH its geometry -> Stage 2 parser -> ProductExtraction.

This is the single place that wires the stages together, reused by `eval_stage2.py`, by any
LayoutLMv3 tier, and by the app. It reads a product as a GROUP of images (front/back/…) the
way the rest of the codebase does, because the fields live across different faces.
"""

from __future__ import annotations

import numpy as np

import barcode as barcode_mod
from ocr import router
from preprocessing import run_pipeline

from . import heuristic
from .schema import PositionedLine, ProductExtraction


def _barcode_source_image(steps: list) -> np.ndarray:
    """The oriented/deskewed/normalised image BEFORE background crop - what the bar decoder
    wants (keeps the surrounding contrast it localises on). Mirrors app.py."""
    names = [s.name for s in steps]
    idx = names.index("Crop to Product") if "Crop to Product" in names else None
    return steps[idx - 1].image if idx else steps[0].image


def collect_lines_and_barcode(
    image_paths: list[str], policy: router.Policy | None = None, cfg: dict | None = None
) -> tuple[list[PositionedLine], list[PositionedLine], object, str]:
    """Run Stage 0/1 + barcode across a product's images. Returns
    (layout_lines, text_lines, barcode_result, union_ocr_text).

    Two line pools, not one, because RapidOCR and the local VLM are good at different
    things and neither can stand in for the other (see ocr.router.read_dual):
      * `layout_lines` - ALWAYS RapidOCR's read. The only source with real per-line boxes,
        so it's what position-based extraction (heuristic brand/name-by-size, LayoutLMv3's
        word positions) must use.
      * `text_lines` - the local VLM's read when `policy.use_local_vlm` is on and the MLX
        server is reachable (REPLACING RapidOCR's text, not merged with it - the VLM
        transcribes far more accurately on this dataset), else the same as `layout_lines`.
        Every regex/keyword/block field extractor reads from this pool.
    `union_ocr_text` (used for barcode digit cross-checking and any plain-text display) is
    built from `text_lines`, so it reflects whichever source is actually more accurate.
    """
    policy = policy or router.Policy()
    layout_lines: list[PositionedLine] = []
    text_lines: list[PositionedLine] = []
    group_steps: list[list] = []

    for path in image_paths:
        steps, crops = run_pipeline(str(path), cfg or {})
        group_steps.append(steps)
        for crop in crops:
            h, w = crop.shape[:2]
            geometry_result, text_result = router.read_dual(crop, policy)
            for ln in geometry_result.lines:
                layout_lines.append(PositionedLine(
                    text=ln.text, bbox=ln.bbox, confidence=ln.confidence, img_w=w, img_h=h
                ))
            for ln in text_result.lines:
                text_lines.append(PositionedLine(
                    text=ln.text, bbox=ln.bbox, confidence=ln.confidence, img_w=w, img_h=h
                ))

    union_text = "\n".join(ln.text for ln in text_lines if ln.text)

    # Barcode: decode each image (pre-crop), keep the strongest hit - one signal per product.
    best_bc = None
    for steps in group_steps:
        bc_img = _barcode_source_image(steps)
        r = barcode_mod.read_and_validate(bc_img, union_text)
        if r.code and (best_bc is None or r.source == "bars"):
            best_bc = r
        if r.source == "bars":
            break

    return layout_lines, text_lines, best_bc, union_text


def extract_product(
    image_paths: list[str],
    policy: router.Policy | None = None,
    cfg: dict | None = None,
    use_layoutlm: bool = True,
) -> ProductExtraction:
    """End-to-end Stage 2 for one product: returns the filled ProductExtraction.

    `use_layoutlm` is a MUTUALLY EXCLUSIVE choice of extractor, not an additive merge - this
    changed as part of an explicit "make LayoutLMv3 the sole extractor" experiment (see
    stage2/layoutlm.py's module docstring for the full rationale):
      * `use_layoutlm=True` (default): skip the heuristic parser AND the barcode decode
        entirely. A blank ProductExtraction is filled solely by `layoutlm.merge_into()`. If no
        fine-tuned checkpoint exists yet (`layoutlm.available()` is False), this returns an
        EMPTY extraction rather than falling back to the heuristic - that's the accepted
        tradeoff of no longer treating the heuristic as a safety net.
      * `use_layoutlm=False`: the pre-experiment path - heuristic parser + barcode-decoded
        Upc/Ean, no LayoutLMv3 at all. Kept as a comparison arm, not deleted:
        `eval_stage2.py`'s `--layoutlm` flag toggles between these two alternatives now,
        rather than "heuristic vs. heuristic-plus-model" as it did before.
    Import of layoutlm is lazy so torch/transformers are only touched when this tier runs.
    """
    layout_lines, text_lines, bc, _ = collect_lines_and_barcode(image_paths, policy, cfg)

    if use_layoutlm:
        from . import layoutlm

        extraction = ProductExtraction()
        # LayoutLMv3 reasons over word POSITIONS - it must see layout_lines (real boxes),
        # never text_lines (the VLM's fake whole-crop box would make its spatial signal
        # meaningless).
        layoutlm.merge_into(extraction, layout_lines)
        return extraction

    return heuristic.parse(text_lines, layout_lines, bc)
