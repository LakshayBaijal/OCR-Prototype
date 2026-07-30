"""Dump every product's OCR'd text (rows 0..99, matching ground_truth_100) to plain .txt
files, one per product - so ground truth can be filled/verified from TEXT instead of reading
every image directly. Text costs a small fraction of an image's tokens, so this is the
token-cheap alternative to viewing all ~90 products' photos one by one.

Uses the SAME free-tier OCR this project already relies on (RapidOCR always; the local
Gemma-3 VLM too if the MLX server is reachable - see commands.txt to start it), unioned via
ocr/router.py exactly like the rest of the pipeline. Falls back to RapidOCR-only gracefully
if the MLX server isn't up - no error, just lower text quality for that run.

Each output file is labelled per source image (front/back/etc.), like:

    --- Image_1_8901030817182.0.jpg ---
    <OCR'd text>

    --- Image_2_8901030817182.0.jpg ---
    <OCR'd text>

so whoever reads it (human or Claude) knows which face of the pack each line came from -
important, since e.g. FSSAI/manufacturer info is usually only on ONE face.

Usage:
    venv/bin/python export_ocr_text.py --start 0 --n 100
    venv/bin/python export_ocr_text.py --start 11 --n 5          # just rows 11-15
    venv/bin/python export_ocr_text.py --start 0 --n 100 --no-vlm  # RapidOCR only, faster
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

import products
from ocr import router
from ocr.rapid import _quiet_rapidocr
from preprocessing import load_rgb

ROOT = Path(__file__).parent
DATASET = ROOT / "Dataset"
DEFAULT_OUT = ROOT.parent / "Image to OCR to Json Export" / "ocr_text"


def main(start: int, n: int, out_dir: Path, use_vlm: bool) -> None:
    _quiet_rapidocr()
    df = pd.read_csv(ROOT / "Ground Truth.csv", low_memory=False)
    cols = [c for c in df.columns if "Filename" in c]
    all_names = sorted(p.name for p in DATASET.glob("*.jpg"))

    policy = router.Policy(use_local_vlm=use_vlm, union_free_tiers=use_vlm)
    out_dir.mkdir(parents=True, exist_ok=True)

    if use_vlm:
        from ocr import local_vlm
        if not local_vlm.available():
            print("NOTE: --no-vlm not set, but the MLX server isn't reachable at "
                  "127.0.0.1:8080 - falling back to RapidOCR-only for every row (see "
                  "commands.txt to start the server for better text quality).")

    end = min(start + n, len(df))
    written, skipped = 0, 0
    for i in range(start, end):
        row = df.iloc[i]
        img_names = products.group_for(
            [str(row[c]).strip() for c in cols if isinstance(row[c], str) and str(row[c]).strip()],
            all_names,
        )
        paths = [DATASET / nm for nm in img_names if (DATASET / nm).exists()]
        if not paths:
            skipped += 1
            continue

        blocks = []
        for p in paths:
            rgb = load_rgb(str(p))
            result = router.read(rgb, policy)
            blocks.append(f"--- {p.name} ---\n{result.text or '(no text found)'}")

        out_path = out_dir / f"row{i}.txt"
        out_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
        written += 1
        print(f"row{i}: {len(paths)} image(s) -> {out_path.name}")

    print(f"\nwrote {written} file(s) to {out_dir} (skipped {skipped} with no resolvable images)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0, help="starting Ground Truth.csv row index")
    ap.add_argument("--n", type=int, default=100, help="number of products to process")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory for rowN.txt files")
    ap.add_argument("--no-vlm", action="store_true",
                     help="RapidOCR only, skip the local VLM tier entirely (faster, no MLX server needed)")
    a = ap.parse_args()
    main(a.start, a.n, a.out, not a.no_vlm)
