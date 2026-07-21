"""Stage 2 benchmark - how well does the Layout/Field Parser recover structured fields?

Runs the whole stack (Stage 0 -> 1 -> barcode -> Stage 2 parser) on a sample of products and
scores the extracted ProductExtraction against the Ground Truth.csv row with
`json_metrics.score` (fuzzy, order-independent, numeric-aware). Reports:

  * per-field F1  - the headline, one number per field
  * precision / recall / F1 overall  - across all fields
  * coverage      - how often each field was populated at all (vs abstained)

Scored per PRODUCT (fields live across a product's front/back faces), same grouping as every
other benchmark here. Uses the FAST heuristic Stage-0 segmenter by default (BGREMOVE=0) so a
100-product run is minutes, not hours; set BGREMOVE=1 for the exact app pipeline.

Usage:
  venv/bin/python eval_stage2.py --n 100
  venv/bin/python eval_stage2.py --n 100 --vlm         # union the local VLM into Stage 1 OCR
  venv/bin/python eval_stage2.py --n 100 --layoutlm    # also run the LayoutLMv3 tier (needs
                                                       #   a trained checkpoint - see train_layoutlm.py)
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("BGREMOVE", "0")  # fast heuristic Stage-0; BGREMOVE=1 for the model
warnings.filterwarnings("ignore")

import pandas as pd

import products
from json_metrics import score
from ocr import router
from ocr.rapid import _quiet_rapidocr
from stage2 import TARGET_FIELDS, extract_product

ROOT = Path(__file__).parent
DATASET = ROOT / "Dataset"


def _gt_row(row: pd.Series) -> dict:
    """Ground-truth dict for the fields Stage 2 targets, blanks dropped to None so an empty
    GT cell scores as an abstention rather than a wrong expectation."""
    out = {}
    for f in TARGET_FIELDS:
        v = row.get(f)
        out[f] = None if (v is None or pd.isna(v) or not str(v).strip()) else str(v).strip()
    return out


def main(n: int, use_vlm: bool, use_layoutlm: bool) -> None:
    _quiet_rapidocr()
    df = pd.read_csv(ROOT / "Ground Truth.csv", low_memory=False)
    cols = [c for c in df.columns if "Filename" in c]
    all_names = sorted(p.name for p in DATASET.glob("*.jpg"))
    sample = df.sample(n=min(n, len(df)), random_state=7)

    policy = router.Policy(use_local_vlm=use_vlm, union_free_tiers=use_vlm, allow_paid=False)

    per_field = {f: {"f1": 0.0, "n": 0, "populated": 0} for f in TARGET_FIELDS}
    agg = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    n_prod = 0

    for _, row in sample.iterrows():
        img_names = products.group_for(
            [str(row[c]).strip() for c in cols if isinstance(row[c], str) and str(row[c]).strip()],
            all_names,
        )
        paths = [str(DATASET / nm) for nm in img_names if (DATASET / nm).exists()]
        if not paths:
            continue
        n_prod += 1

        gt = _gt_row(row)
        pred = extract_product(paths, policy=policy, use_layoutlm=use_layoutlm).to_scoring_dict()

        s = score(gt, pred)
        for k in agg:
            agg[k] += s[k]
        for f in TARGET_FIELDS:
            fs = score({f: gt[f]}, {f: pred[f]})["f1"]
            per_field[f]["f1"] += fs
            per_field[f]["n"] += 1
            per_field[f]["populated"] += int(pred[f] is not None)

    if not n_prod:
        print("no products sampled")
        return

    # ---------------------------------------------------------------- report
    tiers = "rapidocr+VLM" if use_vlm else "rapidocr"
    extras = tiers + (" + LayoutLMv3" if use_layoutlm else "")
    print(f"\n{'='*76}")
    print(f"STAGE 2 - FIELD EXTRACTION  ({n_prod} products, Stage-1 engines: {extras})")
    print(f"{'='*76}\n")

    print(f"  {'field':20} {'F1':>7} {'coverage':>10}")
    print(f"  {'-'*40}")
    for f in TARGET_FIELDS:
        d = per_field[f]
        print(f"  {f:20} {d['f1']/d['n']:6.0%} {d['populated']/d['n']:9.0%}")
    print(f"  {'-'*40}")
    print(f"  {'OVERALL (macro)':20} {sum(d['f1']/d['n'] for d in per_field.values())/len(per_field):6.0%}")
    print()
    print(f"  precision {agg['precision']/n_prod:.0%}   recall {agg['recall']/n_prod:.0%}   "
          f"F1 {agg['f1']/n_prod:.0%}   (leaf-level, across all fields)")
    print()
    print("  F1       = fuzzy match of the extracted value vs ground truth (numeric-aware)")
    print("  coverage = how often the field was populated at all (not abstained)")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of products to sample")
    ap.add_argument("--vlm", action="store_true", help="union the local VLM tier into Stage 1 OCR")
    ap.add_argument("--layoutlm", action="store_true", help="also run the LayoutLMv3 tier")
    a = ap.parse_args()
    import evaluation
    with evaluation.capture("eval_stage2"):
        main(a.n, a.vlm, a.layoutlm)
