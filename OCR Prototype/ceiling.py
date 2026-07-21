"""What is actually RECOVERABLE from these images? (the ceiling)

Field recall sits at 45% and we were about to go chasing the other 55% with better engines.
This checks whether that 55% exists at all.

The finding that prompted this: "Pass Pass Pulse Candy" has 3 images - and image 3 is a
byte-identical duplicate of image 1, image 2 is a zoomed crop of it. All three show the front
of a candy wrapper. There is no barcode, no MRP, no FSSAI in any of them. Yet the ground truth
lists all three (its MRP is even Rs 200 - a wholesale box price for a single sweet). The
ground truth comes from a CATALOG DATABASE, not from the photographs.

So we measure two things, deterministically and for free:

  1. DISTINCT VIEWS - how many of a product's "3-5 images" are actually different pictures?
     (hash the pixels; duplicates and re-crops inflate the apparent coverage)

  2. BACK PANEL PRESENT - does ANY image show the informative panel? MRP, FSSAI, barcode and
     manufacturer all live there. We detect it by its unmistakable vocabulary ("ingredients",
     "nutritional", "fssai", "best before", "customer care", ...). No panel photographed
     => those fields are UNOBTAINABLE by any OCR engine, free or paid.

Then recall is re-reported against what is reachable, instead of against 100%.

Usage:  venv/bin/python ceiling.py --n 60
"""

from __future__ import annotations

import argparse
import hashlib
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

import products
from evaluate_ocr import found
from ocr import cache, router
from preprocessing import load_rgb

ROOT = Path(__file__).parent
DATASET = ROOT / "Dataset"

# Vocabulary that only ever appears on the informative/back panel of a pack.
PANEL_MARKERS = (
    "ingredient", "nutrition", "nutritional", "fssai", "lic no", "best before",
    "customer care", "manufactured", "mfd", "mfg", "packed", "net wt", "net weight",
    "storage", "allergen", "consumer", "shelf life", "batch", "expiry",
)

# Fields that physically live on that panel. If the panel was never shot, these are gone.
PANEL_FIELDS = ["Mrp", "Fssai No", "Upc / Ean", "Manufacturer"]
FRONT_FIELDS = ["Brand", "Net Quantity"]


def has_panel(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in PANEL_MARKERS)


def main(n: int) -> None:
    df = pd.read_csv(ROOT / "Ground Truth.csv", low_memory=False)
    cols = [c for c in df.columns if "Filename" in c]
    all_names = sorted(p.name for p in DATASET.glob("*.jpg"))
    sample = df.sample(n=min(n, len(df)), random_state=7)

    listed = distinct = 0
    with_panel = 0
    n_products = 0

    # per field: recoverable (panel present) vs actually recovered
    stat = {f: {"gt": 0, "reachable": 0, "got": 0} for f in PANEL_FIELDS + FRONT_FIELDS}

    for _, row in sample.iterrows():
        names = products.group_for(
            [str(row[c]).strip() for c in cols
             if isinstance(row[c], str) and str(row[c]).strip()],
            all_names,
        )
        paths = [DATASET / n for n in names if (DATASET / n).exists()]
        if not paths:
            continue
        n_products += 1

        # 1. how many of these images are actually DIFFERENT pictures?
        seen: set[str] = set()
        texts: list[str] = []
        for p in paths:
            rgb = load_rgb(str(p))
            seen.add(cache.checksum(rgb))
            texts.append(router.read(rgb).text)

        listed += len(paths)
        distinct += len(seen)

        all_text = "\n".join(texts)
        panel = has_panel(all_text)
        with_panel += panel

        for f in PANEL_FIELDS + FRONT_FIELDS:
            got = found(f, row.get(f), all_text)
            if got is None:
                continue
            stat[f]["gt"] += 1
            # a panel field is only reachable if a panel was photographed
            reachable = panel if f in PANEL_FIELDS else True
            stat[f]["reachable"] += reachable
            stat[f]["got"] += bool(got)

    print(f"\n{'='*74}")
    print(f"RECALL CEILING - what is even POSSIBLE from these photos?  ({n_products} products)")
    print(f"{'='*74}\n")

    print("1. ARE THE 3-5 IMAGES PER PRODUCT ACTUALLY DIFFERENT PICTURES?")
    print(f"   images listed in the CSV : {listed}")
    print(f"   distinct by pixel hash   : {distinct}   ({distinct/listed*100:.0f}%)")
    print(f"   -> {listed - distinct} are exact duplicates of another image of the same product")

    print(f"\n2. WAS THE INFORMATIVE (BACK) PANEL EVER PHOTOGRAPHED?")
    print(f"   products with a back panel visible: {with_panel}/{n_products} "
          f"({with_panel/n_products*100:.0f}%)")
    print(f"   -> for the other {n_products - with_panel}, MRP / FSSAI / barcode / manufacturer")
    print(f"      are NOT IN ANY IMAGE. No OCR engine can ever recover them.")

    print(f"\n3. RECALL vs THE CEILING (not vs 100%)")
    print(f"   {'field':16} {'in GT':>6} {'reachable':>10} {'recovered':>10} "
          f"{'raw recall':>11} {'vs ceiling':>11}")
    print(f"   {'-'*70}")
    for f in PANEL_FIELDS + FRONT_FIELDS:
        s = stat[f]
        if not s["gt"]:
            continue
        raw = s["got"] / s["gt"]
        ceil = s["got"] / s["reachable"] if s["reachable"] else 0.0
        print(f"   {f:16} {s['gt']:6} {s['reachable']:10} {s['got']:10} "
              f"{raw:10.0%} {ceil:10.0%}")
    print()
    print("   'vs ceiling' = recall counting ONLY products where the value was photographed.")
    print("   That is the number the OCR engine is actually responsible for.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    import evaluation
    with evaluation.capture("ceiling"):
        main(ap.parse_args().n)
