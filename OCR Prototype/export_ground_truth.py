"""Run the real Stage 0->1->barcode->Stage 2 pipeline on real products and write its output
in the EXACT JSON shape as `Image to OCR to Json Export/ground_truth_100/rowN.json` - so this
pipeline's own guesses can be dropped next to, and diffed against, that hand/image-derived
ground truth.

LayoutLMv3 runs by default (see stage2/pipeline.py) since it's local inference against a local
checkpoint - no per-call cost to gate it behind. Pass --no-layoutlm to compare heuristic-only.

Usage:
  venv/bin/python export_ground_truth.py --start 0 --n 10
  venv/bin/python export_ground_truth.py --start 0 --n 10 --out /path/to/out_dir
  venv/bin/python export_ground_truth.py --start 0 --n 10 --no-layoutlm
  venv/bin/python export_ground_truth.py --start 0 --n 10 --vlm     # union local VLM into Stage 1
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("BGREMOVE", "0")  # fast heuristic Stage-0; BGREMOVE=1 for the exact app pipeline
warnings.filterwarnings("ignore")

import pandas as pd

import products
from ocr import router
from ocr.rapid import _quiet_rapidocr
from stage2 import GT_FIELD_ORDER, extract_product

ROOT = Path(__file__).parent
DATASET = ROOT / "Dataset"
DEFAULT_OUT = ROOT / "Output" / "pipeline_ground_truth"


def _ordered(gt_dict: dict) -> dict:
    """Re-key into GT_FIELD_ORDER so the written JSON matches ground_truth_100's key order
    exactly (cosmetic, but makes the two easy to diff by eye)."""
    return {k: gt_dict.get(k) for k in GT_FIELD_ORDER}


def main(start: int, n: int, out_dir: Path, use_vlm: bool, use_layoutlm: bool) -> None:
    _quiet_rapidocr()
    df = pd.read_csv(ROOT / "Ground Truth.csv", low_memory=False)
    cols = [c for c in df.columns if "Filename" in c]
    all_names = sorted(p.name for p in DATASET.glob("*.jpg"))

    policy = router.Policy(use_local_vlm=use_vlm, union_free_tiers=use_vlm, allow_paid=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    end = min(start + n, len(df))
    written, skipped = 0, 0
    for i in range(start, end):
        row = df.iloc[i]
        img_names = products.group_for(
            [str(row[c]).strip() for c in cols if isinstance(row[c], str) and str(row[c]).strip()],
            all_names,
        )
        paths = [str(DATASET / nm) for nm in img_names if (DATASET / nm).exists()]
        if not paths:
            skipped += 1
            continue

        extraction = extract_product(paths, policy=policy, use_layoutlm=use_layoutlm)
        record = _ordered(extraction.to_ground_truth_dict())

        out_path = out_dir / f"row{i}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
            f.write("\n")
        written += 1

    print(f"wrote {written} record(s) to {out_dir} (skipped {skipped} with no resolvable images)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0, help="starting Ground Truth.csv row index")
    ap.add_argument("--n", type=int, default=10, help="number of products to process")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory for rowN.json files")
    ap.add_argument("--vlm", action="store_true", help="union the local VLM tier into Stage 1 OCR")
    ap.add_argument("--no-layoutlm", action="store_true", help="skip the LayoutLMv3 tier (heuristic only)")
    a = ap.parse_args()
    main(a.start, a.n, a.out, a.vlm, not a.no_layoutlm)
