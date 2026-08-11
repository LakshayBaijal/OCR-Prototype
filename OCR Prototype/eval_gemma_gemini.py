"""Run the new Gemma-first/Gemini-fallback pipeline (stage2.gemma_then_gemini) over real
products and score it against `ondc-intelligence`'s own hand-curated ground truth - the
SAME 100-product/417-image set and SAME field-matching engine used throughout that
project's own eval reports this session, so these numbers are genuinely comparable, not
a second, differently-shaped benchmark.

Reports, per row and in aggregate: status, the tier that served each result
(gemma original / gemma repaired / gemini fallback - internally "gemma-first-try" /
"gemma-repaired" / "gemini-fallback", matching ondc-intelligence's own `model_tier`
values byte-for-byte), and precision/recall/F1 via ondc-intelligence's own
evals/gt_match.py, imported directly rather than re-derived.

Usage:
    venv/bin/python eval_gemma_gemini.py --n 20
    venv/bin/python eval_gemma_gemini.py --start 0 --n 100
    venv/bin/python eval_gemma_gemini.py --n 20 --with-fallback   # also let heuristic+LayoutLM fill gaps

`--with-fallback` is OFF by default deliberately: ondc-intelligence's own pipeline has no
heuristic/LayoutLM tier at all, so leaving it on here by default would let some rows show
non-null fields from a completely different source after all 3 VLM tiers failed - making
the numbers NOT comparable to anything ondc-intelligence has measured. Use --with-fallback
only when you explicitly want that separate, non-comparable variant.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

import products
from ocr.rapid import _quiet_rapidocr
from stage2 import GT_FIELD_ORDER
from stage2.gemma_then_gemini import extract_product_gemma_then_gemini

ROOT = Path(__file__).parent
DATASET = ROOT / "Dataset"
OUT_DIR = ROOT / "Output" / "gemma_gemini_output"
ONDC_ROOT = ROOT.parent / "ondc-intelligence"
ONDC_GT_DIR = ONDC_ROOT / "evals" / "ground_truth" / "image2product_gt"

sys.path.insert(0, str(ONDC_ROOT / "evals"))
from evals_image2product import FIELD_TYPES  # noqa: E402  (36 fields, ondc-intelligence's own set)
from gt_match import classify_field, compute_prf, pred_value  # noqa: E402

TIER_DISPLAY = {
    "gemma-first-try": "gemma original",
    "gemma-repaired": "gemma repaired",
    "gemini-fallback": "gemini fallback",
}


def _ordered(gt_dict: dict) -> dict:
    """Same re-key export_ground_truth.py's own _ordered() does - matches GT_FIELD_ORDER
    so the written JSON is easy to diff by eye against the ground truth."""
    return {k: gt_dict.get(k) for k in GT_FIELD_ORDER}


def main(start: int, n: int, use_fallback: bool) -> None:
    _quiet_rapidocr()

    if not ONDC_GT_DIR.is_dir():
        sys.exit(f"error: expected ondc-intelligence ground truth at {ONDC_GT_DIR}, not found.")

    df = pd.read_csv(ROOT / "Ground Truth.csv", low_memory=False)
    cols = [c for c in df.columns if "Filename" in c]
    all_names = sorted(p.name for p in DATASET.glob("*.jpg"))

    end = min(start + n, len(df), 100)  # no hand-curated GT exists past row99
    if start + n > 100:
        print(f"note: clipping to row99 - ondc-intelligence's hand-curated ground truth only covers rows 0-99.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tier_counts: dict[str, int] = defaultdict(int)
    total_counts: dict[str, int] = defaultdict(int)
    per_field_counts: dict[str, dict[str, int]] = {f: defaultdict(int) for f in FIELD_TYPES}
    latencies: list[float] = []
    rows_ok = 0

    print(f"{'row':>5}  {'imgs':>4}  {'status':>10}  {'tier':>16}  product")
    print("-" * 88)

    import time

    for i in range(start, end):
        row = df.iloc[i]
        img_names = products.group_for(
            [str(row[c]).strip() for c in cols if isinstance(row[c], str) and str(row[c]).strip()],
            all_names,
        )
        paths = [str(DATASET / nm) for nm in img_names if (DATASET / nm).exists()]

        gt_path = ONDC_GT_DIR / f"row{i}.json"
        if not paths or not gt_path.exists():
            print(f"{i:>5}  {'-':>4}  {'skipped':>10}  {'-':>16}  (no images or no ground truth)")
            continue

        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        product_name = str(row.get("Product Name", ""))[:40]

        t0 = time.perf_counter()
        try:
            extraction, _vlm_dict, tier_info, _attempts = extract_product_gemma_then_gemini(
                paths, use_fallback=use_fallback
            )
        except Exception as exc:
            print(f"{i:>5}  {len(paths):>4}  {'ERROR':>10}  {'-':>16}  {product_name}  ({type(exc).__name__}: {exc})")
            continue
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        tier = tier_info["tier"]
        tier_counts[tier] += 1
        rows_ok += 1

        pred = _ordered(extraction.to_ground_truth_dict())
        (OUT_DIR / f"row{i}.json").write_text(json.dumps(pred, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        matched = 0
        for field, ftype in FIELD_TYPES.items():
            c = classify_field(ftype, pred_value(field, pred), gt.get(field))
            per_field_counts[field][c] += 1
            total_counts[c] += 1
            if c in ("correct", "empty_gt"):
                matched += 1

        print(
            f"{i:>5}  {len(paths):>4}  {elapsed:>8.1f}s  {TIER_DISPLAY[tier]:>16}  "
            f"{product_name}  ({matched}/{len(FIELD_TYPES)})"
        )

    print("\n" + "=" * 88)
    print(f"rows: {rows_ok} scored\n")

    total_tiered = sum(tier_counts.values())
    if total_tiered:
        print("model tier:")
        for tier in ("gemma-first-try", "gemma-repaired", "gemini-fallback"):
            n_tier = tier_counts.get(tier, 0)
            pct = 100 * n_tier / total_tiered
            print(f"  {TIER_DISPLAY[tier]:<16} {n_tier:>4}/{total_tiered}  ({pct:.1f}%)")
        fallback_pct = 100 * tier_counts.get("gemini-fallback", 0) / total_tiered
        print(
            f"\n  {fallback_pct:.1f}% of rows actually reached Gemini - that share is what "
            "determines whether Gemma's cost/local-hosting benefit survives at this volume, "
            "not the token price alone."
        )

    if latencies:
        s = sorted(latencies)
        print(
            f"\nlatency (s): min {min(s):.1f}  mean {sum(s) / len(s):.1f}  "
            f"median {s[len(s) // 2]:.1f}  max {max(s):.1f}"
        )

    if rows_ok:
        total_cells = rows_ok * len(FIELD_TYPES)
        n_match = total_counts["correct"] + total_counts["empty_gt"]
        prf = compute_prf(total_counts)
        gt_present = total_counts["correct"] + total_counts["mismatch"] + total_counts["missing"]
        print(f"\noverall match: {n_match}/{total_cells} ({n_match / total_cells:.1%})")
        print(
            f"over the {gt_present} GT-present cells: correct {total_counts['correct']}  "
            f"mismatch {total_counts['mismatch']}  missing {total_counts['missing']}"
        )
        print(f"precision: {prf['precision']:.3f}  recall: {prf['recall']:.3f}  F1: {prf['f1']:.3f}")

        print("\nper-field (field, correct, mismatch, missing, empty_gt):")
        for field in FIELD_TYPES:
            c = per_field_counts[field]
            print(
                f"  {field:<24} {c['correct']:>3}  {c['mismatch']:>3}  {c['missing']:>3}  {c['empty_gt']:>3}"
            )

    print(f"\n{rows_ok} record(s) written to {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0, help="starting row index")
    ap.add_argument("--n", type=int, default=100, help="number of products to process")
    ap.add_argument(
        "--with-fallback", action="store_true",
        help="also let heuristic+LayoutLM fill gaps (separate, NOT comparable to ondc-intelligence's own numbers)",
    )
    a = ap.parse_args()

    from evaluation import capture

    with capture("eval_gemma_gemini"):
        main(a.start, a.n, a.with_fallback)
