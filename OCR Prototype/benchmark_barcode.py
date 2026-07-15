"""How often does the barcode pipeline actually work, end to end?

Four separate hit rates, because each stage fails for a different reason and lumping them
together would hide which one to fix:

  1. bar decode      - barcode visible AND cv2 could read the bars
  2. checksum valid  - of what we found (bars or OCR digits), how much is a well-formed code
  3. OFF lookup      - of the valid codes, how many exist in Open Food Facts
  4. cross-check     - of the found lookups, does canonical brand/name agree with OCR

Usage:  venv/bin/python benchmark_barcode.py --n 40
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from barcode import read_and_validate
from ocr import rapid
from preprocessing import load_rgb

ROOT = Path(__file__).parent
DATASET = ROOT / "Dataset"


def main(n: int) -> None:
    df = pd.read_csv(ROOT / "Ground Truth.csv", low_memory=False)
    cols = [c for c in df.columns if "Filename" in c]
    sample = df.sample(n=min(n, len(df)), random_state=11)

    decoded_bars = valid = looked_up = matches_gt = 0
    brand_checks = []
    products = 0

    for _, row in sample.iterrows():
        gt_upc = str(row.get("Upc / Ean", "")).strip()
        try:
            gt_upc = str(int(float(gt_upc)))
        except ValueError:
            continue

        paths = [
            DATASET / str(row[c]).strip()
            for c in cols
            if isinstance(row[c], str) and (DATASET / str(row[c]).strip()).exists()
        ]
        if not paths:
            continue
        products += 1

        best = None
        for p in paths:
            rgb = load_rgb(str(p))
            text = rapid.run(rgb).text
            r = read_and_validate(rgb, text, ocr_brand=row.get("Brand"))
            if r.code and (best is None or r.source == "bars"):
                best = r
            if r.source == "bars":
                break  # a bar decode is as good as it gets; stop early

        if best is None or best.code is None:
            continue

        if best.source == "bars":
            decoded_bars += 1
        valid += best.valid_checksum
        looked_up += best.looked_up
        if best.valid_checksum and best.code == gt_upc:
            matches_gt += 1
        if best.brand_match is not None:
            brand_checks.append(best.brand_match)

    print(f"\n{'='*66}")
    print(f"BARCODE PIPELINE - {products} products")
    print(f"{'='*66}\n")
    print(f"  1. bar decode succeeded          : {decoded_bars:3}/{products}  ({decoded_bars/products*100:.0f}%)")
    print(f"  2. checksum-valid code found     : {valid:3}/{products}  ({valid/products*100:.0f}%)")
    print(f"     of those, matches GT exactly  : {matches_gt:3}/{max(valid,1)}  ({matches_gt/max(valid,1)*100:.0f}%)")
    print(f"  3. found in Open Food Facts      : {looked_up:3}/{max(valid,1)}  ({looked_up/max(valid,1)*100:.0f}% of valid codes)")
    if brand_checks:
        avg = sum(brand_checks) / len(brand_checks)
        low = sum(1 for b in brand_checks if b < 0.5)
        print(f"  4. brand cross-checks run        : {len(brand_checks)}  (avg similarity {avg:.2f}, "
              f"{low} flagged <0.5 similarity)")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    main(ap.parse_args().n)
