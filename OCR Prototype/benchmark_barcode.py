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
import difflib
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

import products
from barcode import read_and_validate
from ocr import rapid
from preprocessing import load_rgb

ROOT = Path(__file__).parent
DATASET = ROOT / "Dataset"


def main(n: int) -> None:
    df = pd.read_csv(ROOT / "Ground Truth.csv", low_memory=False)
    cols = [c for c in df.columns if "Filename" in c]
    all_names = sorted(p.name for p in DATASET.glob("*.jpg"))
    sample = df.sample(n=min(n, len(df)), random_state=11)

    decoded_bars = valid = looked_up = matches_gt = 0
    unverified_surfaced = 0
    unverified_closeness = []  # fraction of GT digits that survive, per surfaced candidate
    brand_checks = []
    n_products = 0

    for _, row in sample.iterrows():
        gt_upc = str(row.get("Upc / Ean", "")).strip()
        try:
            gt_upc = str(int(float(gt_upc)))
        except ValueError:
            continue

        names = products.group_for(
            [str(row[c]).strip() for c in cols
             if isinstance(row[c], str) and str(row[c]).strip()],
            all_names,
        )
        paths = [DATASET / n for n in names if (DATASET / n).exists()]
        if not paths:
            continue
        n_products += 1

        best = None
        best_unverified = None
        for p in paths:
            rgb = load_rgb(str(p))
            text = rapid.run(rgb).text
            r = read_and_validate(rgb, text, ocr_brand=row.get("Brand"))
            if r.code and (best is None or r.source == "bars"):
                best = r
            if r.unverified_candidate and not best_unverified:
                best_unverified = r.unverified_candidate
            if r.source == "bars":
                break  # a bar decode is as good as it gets; stop early

        if best is None or best.code is None:
            # Nothing verified - but is OCR's raw (checksum-failed) reading actually a USEFUL
            # starting point for a human, or unrelated noise? Not "does it exactly match GT" -
            # unverified_candidate always fails the checksum by definition, and a genuine GT
            # barcode always PASSES it, so an exact match is structurally near-impossible
            # regardless of how close the read is; that would misleadingly always read ~0%.
            # Instead: similarity ratio to GT (shared digits in matching positions) - a
            # single-digit miss like "8906055800108" vs "8904055800108" scores high, unrelated
            # noise like a phone number scores low.
            if best_unverified:
                unverified_surfaced += 1
                closeness = difflib.SequenceMatcher(None, best_unverified, gt_upc).ratio()
                unverified_closeness.append(closeness)
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
    print(f"BARCODE PIPELINE - {n_products} products")
    print(f"{'='*66}\n")
    print(f"  1. bar decode succeeded          : {decoded_bars:3}/{n_products}  ({decoded_bars/n_products*100:.0f}%)")
    print(f"  2. checksum-valid code found     : {valid:3}/{n_products}  ({valid/n_products*100:.0f}%)")
    print(f"     of those, matches GT exactly  : {matches_gt:3}/{max(valid,1)}  ({matches_gt/max(valid,1)*100:.0f}%)")
    print(f"  3. found in Open Food Facts      : {looked_up:3}/{max(valid,1)}  ({looked_up/max(valid,1)*100:.0f}% of valid codes)")
    if unverified_surfaced:
        avg_close = sum(unverified_closeness) / len(unverified_closeness)
        near_miss = sum(1 for c in unverified_closeness if c >= 0.85)
        print(f"  5. unverified candidates surfaced : {unverified_surfaced:3} (for human review, on top of the "
              f"{valid} verified above)")
        print(f"     avg similarity to true GT code   : {avg_close:.0%}  "
              f"({near_miss}/{unverified_surfaced} are near-misses, >=85% similar - i.e. plausibly "
              f"1-2 digits off and worth a human's 5-second glance, not unrelated noise)")
    if brand_checks:
        avg = sum(brand_checks) / len(brand_checks)
        low = sum(1 for b in brand_checks if b < 0.5)
        print(f"  4. brand cross-checks run        : {len(brand_checks)}  (avg similarity {avg:.2f}, "
              f"{low} flagged <0.5 similarity)")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    import evaluation
    with evaluation.capture("benchmark_barcode"):
        main(ap.parse_args().n)
