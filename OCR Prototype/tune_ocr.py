"""Why is field recall only 40%? Test whether it is the ENGINE or how we FEED it.

FSSAI is the honest probe: a 14-digit number, printed in tiny type on the back panel, and
we know the exact right answer from the ground truth. If OCR can't get it, nothing
downstream can.

Upscaling the image changed nothing (33% at 1x, 2x and 3x), which means RapidOCR is
resizing internally and throwing our extra pixels away. So the two real levers are:
  - its detection thresholds (limit_side_len / box_thresh / text_score)
  - tiling: cut the pack into overlapping tiles so each is read at native scale

Usage:  venv/bin/python tune_ocr.py
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rapidocr import RapidOCR

import products
from ocr.rapid import _quiet_rapidocr
from preprocessing import load_rgb

ROOT = Path(__file__).parent
DATASET = ROOT / "Dataset"


def digits(t: str) -> str:
    return re.sub(r"\D", "", t)


def read(engine: RapidOCR, rgb: np.ndarray) -> str:
    out = engine(rgb)
    return "\n".join(str(t) for t in out.txts) if out.txts else ""


def read_tiled(engine: RapidOCR, rgb: np.ndarray, grid: int = 2, overlap: float = 0.15) -> str:
    """OCR overlapping tiles. Each tile is a smaller image, so the engine's internal resize
    shrinks it far less - tiny print survives. Overlap stops text on a seam being cut in half."""
    h, w = rgb.shape[:2]
    th, tw = h // grid, w // grid
    oh, ow = int(th * overlap), int(tw * overlap)

    chunks = [read(engine, rgb)]  # keep the whole-image pass too; tiles are additive
    for i in range(grid):
        for j in range(grid):
            y0, y1 = max(0, i * th - oh), min(h, (i + 1) * th + oh)
            x0, x1 = max(0, j * tw - ow), min(w, (j + 1) * tw + ow)
            chunks.append(read(engine, rgb[y0:y1, x0:x1]))
    return "\n".join(chunks)


def main() -> None:
    df = pd.read_csv(ROOT / "Ground Truth.csv", low_memory=False)
    cols = [c for c in df.columns if "Filename" in c]
    all_names = sorted(p.name for p in DATASET.glob("*.jpg"))

    cases: list[tuple[str, list[Path]]] = []
    for _, row in df.sample(n=200, random_state=5).iterrows():
        f = str(row.get("Fssai No", "")).strip()
        if not f.isdigit() or len(f) < 10:
            continue
        names = products.group_for(
            [str(row[c]).strip() for c in cols
             if isinstance(row[c], str) and str(row[c]).strip()],
            all_names,
        )
        imgs = [DATASET / n for n in names if (DATASET / n).exists()]
        if imgs:
            cases.append((f, imgs))
        if len(cases) >= 20:
            break

    base = RapidOCR()
    # Detect more, and keep lower-confidence reads: tiny print is exactly what a 0.5 score
    # threshold throws away.
    sensitive = RapidOCR(params={
        "Det.limit_side_len": 1600,
        "Det.limit_type": "max",
        "Det.box_thresh": 0.3,
        "Det.unclip_ratio": 2.0,
        "Global.text_score": 0.3,
    })
    _quiet_rapidocr()  # both constructions reset RapidOCR's logger to INFO; silence it here

    setups = {
        "baseline (default)": lambda rgb: read(base, rgb),
        "sensitive thresholds": lambda rgb: read(sensitive, rgb),
        "tiled 2x2 (default)": lambda rgb: read_tiled(base, rgb, 2),
        "tiled 3x3 (default)": lambda rgb: read_tiled(base, rgb, 3),
        "tiled 3x3 + sensitive": lambda rgb: read_tiled(sensitive, rgb, 3),
    }

    hits = {k: 0 for k in setups}
    for fssai, imgs in cases:
        for name, fn in setups.items():
            for p in imgs:
                if fssai in digits(fn(load_rgb(str(p)))):
                    hits[name] += 1
                    break

    n = len(cases)
    print(f"\nFSSAI recovery - {n} products where we KNOW the number is printed\n")
    print(f"  {'setup':24} {'recovered':>10}")
    print(f"  {'-'*36}")
    for name in setups:
        print(f"  {name:24} {hits[name]:4}/{n}  ({hits[name]/n*100:3.0f}%)")
    print()


if __name__ == "__main__":
    import evaluation
    with evaluation.capture("tune_ocr"):
        main()
