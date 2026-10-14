"""Batch-run PaddleOCR-VL-1.5 over every PNG in one or more folders, writing each
result as a same-name .txt next to the image (imgN.png -> imgN.txt).

Run (from the OCR Prototype folder):

    venv/bin/python3.14 batch_ocr.py \\
        "/Users/lakshay.baijal/Desktop/Projects/Image:Text Enhancement/ondc-intelligence-fea-enhancement-evals/image_enhancement/image_enhancement_processed_silhouette" \\
        "/Users/lakshay.baijal/Desktop/Projects/Image:Text Enhancement/ondc-intelligence-fea-enhancement-evals/image_enhancement/image_enhancement_output_nb_silhouette"

Re-runnable: an image whose .txt already exists is skipped, so a killed/rerun
does not redo already-processed images. Delete a .txt to force redoing that one.

THE ALPHA GOTCHA (same one stage1_ocr_app.py handles): these silhouette PNGs
have a transparent background, but the pixels under alpha=0 still hold the
original shelf colours - flattening onto white before OCR is required, not
optional, or PaddleOCR-VL reads leftover shelf clutter as if it were on the
product. See `_flatten` below (identical to stage1_ocr_app.py's).
"""

from __future__ import annotations

import os

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image

import stage1_ocr

NOT_A_PRODUCT = {"embeddings", "embeddings_index"}


def _flatten(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
        if not has_alpha:
            return np.asarray(im.convert("RGB"))
        rgba = im.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return np.asarray(Image.alpha_composite(white, rgba).convert("RGB"))


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _sort_key(p: Path):
    """imgN ordering: img2 before img10, not after it."""
    digits = "".join(c for c in p.stem if c.isdigit())
    return (int(digits) if digits else 0, p.stem)


def process_dir(folder: Path) -> None:
    images = sorted(
        (p for p in folder.iterdir()
         if p.suffix.lower() in IMAGE_SUFFIXES and p.stem not in NOT_A_PRODUCT),
        key=_sort_key,
    )
    print(f"\n{folder}\n  {len(images)} image(s)")
    for i, path in enumerate(images, 1):
        out_path = path.with_suffix(".txt")
        if out_path.exists():
            print(f"  [{i}/{len(images)}] {path.name} -> skipped (already done)")
            continue
        try:
            text = stage1_ocr.read_tiled_array(_flatten(path))
            out_path.write_text(text, encoding="utf-8")
            print(f"  [{i}/{len(images)}] {path.name} -> {out_path.name} ({len(text)} chars)")
        except Exception as exc:
            print(f"  [{i}/{len(images)}] {path.name} -> ERROR {type(exc).__name__}: {exc}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: batch_ocr.py <folder> [folder ...]")
    for arg in sys.argv[1:]:
        folder = Path(arg)
        if not folder.is_dir():
            print(f"skip (not a directory): {folder}")
            continue
        process_dir(folder)


if __name__ == "__main__":
    main()
