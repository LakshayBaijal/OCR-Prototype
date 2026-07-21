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
) -> tuple[list[PositionedLine], object, str]:
    """Run Stage 0/1 + barcode across a product's images. Returns
    (positioned_lines, barcode_result, union_ocr_text)."""
    policy = policy or router.Policy()
    lines: list[PositionedLine] = []
    group_steps: list[list] = []
    all_crops: list[np.ndarray] = []

    for path in image_paths:
        steps, crops = run_pipeline(str(path), cfg or {})
        group_steps.append(steps)
        for crop in crops:
            all_crops.append(crop)
            h, w = crop.shape[:2]
            result = router.read(crop, policy)
            for ln in result.lines:
                lines.append(PositionedLine(
                    text=ln.text, bbox=ln.bbox, confidence=ln.confidence, img_w=w, img_h=h
                ))

    union_text = "\n".join(ln.text for ln in lines if ln.text)

    # Barcode: decode each image (pre-crop), keep the strongest hit - one signal per product.
    best_bc = None
    for steps in group_steps:
        bc_img = _barcode_source_image(steps)
        r = barcode_mod.read_and_validate(bc_img, union_text)
        if r.code and (best_bc is None or r.source == "bars"):
            best_bc = r
        if r.source == "bars":
            break

    return lines, best_bc, union_text


def extract_product(
    image_paths: list[str],
    policy: router.Policy | None = None,
    cfg: dict | None = None,
    use_layoutlm: bool = False,
) -> ProductExtraction:
    """End-to-end Stage 2 for one product: returns the filled ProductExtraction.

    The heuristic parser always runs. When `use_layoutlm` is on AND a fine-tuned checkpoint
    is available, the LayoutLMv3 tier runs too and MERGES in (it only fills fields the
    heuristic left empty or is more confident about - never silently overwrites a trusted
    barcode/regex hit). Import is lazy so torch/transformers are only touched when asked for.
    """
    lines, bc, _ = collect_lines_and_barcode(image_paths, policy, cfg)
    extraction = heuristic.parse(lines, bc)

    if use_layoutlm:
        from . import layoutlm

        if layoutlm.available():
            layoutlm.merge_into(extraction, lines)
    return extraction
