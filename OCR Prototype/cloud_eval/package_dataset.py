"""Package the SAME 100-product eval set used everywhere else this project benchmarks a
mode (`ground_truth_100`, scored via `export_ground_truth.py`) into one zip that's ready to
upload as a Kaggle Dataset (or unzip into a mounted Google Drive for Colab).

Deliberately local-only: Kaggle/Colab give you an isolated cloud VM with no access to this
machine's filesystem, so the eval images + Ground Truth.csv + schema have to travel there as
one archive. This is the ONLY piece of cloud_eval/ that runs here rather than in the notebook
- see cloud_eval/run_batch.py (the notebook-side script) and cloud_eval/README.md (the
step-by-step upload/run instructions).

Usage:
    venv/bin/python cloud_eval/package_dataset.py
    venv/bin/python cloud_eval/package_dataset.py --start 0 --n 100 --out cloud_eval/cloud_eval_data.zip
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))  # products.py lives at the project root, this script doesn't

import products  # noqa: E402
DATASET = ROOT / "Dataset"
GT_CSV = ROOT / "Ground Truth.csv"
SCHEMA_JSON = ROOT.parent / "Image to OCR to Json Export" / "DigiCatalog_groceries_schema_format.json"
DEFAULT_OUT = Path(__file__).parent / "cloud_eval_data.zip"


def main(start: int, n: int, out_path: Path) -> None:
    df = pd.read_csv(GT_CSV, low_memory=False)
    cols = [c for c in df.columns if "Filename" in c]
    all_names = sorted(p.name for p in DATASET.glob("*.jpg"))

    end = min(start + n, len(df))
    image_names: set[str] = set()
    resolved_rows = 0
    for i in range(start, end):
        row = df.iloc[i]
        seeds = [str(row[c]).strip() for c in cols if isinstance(row[c], str) and str(row[c]).strip()]
        group = products.group_for(seeds, all_names)
        found = [n for n in group if (DATASET / n).exists()]
        if found:
            image_names.update(found)
            resolved_rows += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(image_names):
            zf.write(DATASET / name, arcname=f"images/{name}")
        zf.write(GT_CSV, arcname="Ground Truth.csv")
        if SCHEMA_JSON.exists():
            zf.write(SCHEMA_JSON, arcname="DigiCatalog_groceries_schema_format.json")
        else:
            print(f"WARNING: schema file not found at {SCHEMA_JSON} - not bundled. "
                  "run_batch.py falls back to a bare field list without it, which is fine "
                  "but gives the model less field-level guidance.")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"wrote {out_path} ({size_mb:.0f} MB): {len(image_names)} images, "
          f"{resolved_rows}/{end - start} rows resolved (rows {start}..{end - 1})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0, help="starting Ground Truth.csv row index")
    ap.add_argument("--n", type=int, default=100, help="number of products to package (matches ground_truth_100 by default)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output zip path")
    a = ap.parse_args()
    main(a.start, a.n, a.out)
