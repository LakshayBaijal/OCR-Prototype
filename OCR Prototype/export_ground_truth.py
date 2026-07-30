"""Run the real Stage 0->1->barcode->Stage 2 pipeline on real products and write its output
in the EXACT JSON shape as `Image to OCR to Json Export/ground_truth_100/rowN.json` - so this
pipeline's own guesses can be dropped next to, and diffed against, that hand/image-derived
ground truth.

Default: heuristic+LayoutLM (see stage2/pipeline.py). --no-layoutlm skips LayoutLM (or, in
any --vlm-* / --gemini mode, skips the heuristic+LayoutLM fallback instead).

Four alternative, mutually exclusive extraction modes (stage2/vlm_direct.py,
stage2/gemini_direct.py, stage2/openrouter_direct.py), all falling back to heuristic+LayoutLM
for whatever's left null:
  --vlm-direct    local MLX VLM (Gemma-3), one call: images in, structured JSON out directly.
                  Needs the MLX server running - see ocr/local_vlm.py's module docstring.
  --vlm-two-call  local MLX VLM, two calls: OCR each image separately first, then a SEPARATE
                  text-only call maps the combined OCR text into the schema.
  --gemini        the REAL Gemini API - not free, but the proven production-grade baseline
                  this project measures the free/local paths against. Needs GEMINI_API_KEY
                  (see stage2/gemini_direct.py's module docstring for how to set it).
  --openrouter    a free, ALREADY-HOSTED model via openrouter.ai (default: Gemma-4 26B-A4B -
                  sidesteps the local mlx-vlm PLE-quantization bug entirely). Needs
                  OPENROUTER_API_KEY (see stage2/openrouter_direct.py's module docstring).
                  --openrouter-model picks which of MODEL_CHOICES to use.

Usage:
  venv/bin/python export_ground_truth.py --start 0 --n 10
  venv/bin/python export_ground_truth.py --start 0 --n 10 --out /path/to/out_dir
  venv/bin/python export_ground_truth.py --start 0 --n 10 --no-layoutlm
  venv/bin/python export_ground_truth.py --start 0 --n 10 --vlm     # union local VLM into Stage 1 OCR
  venv/bin/python export_ground_truth.py --start 0 --n 1 --vlm-direct     # local VLM, direct
  venv/bin/python export_ground_truth.py --start 0 --n 1 --vlm-two-call   # local VLM, OCR-then-map
  venv/bin/python export_ground_truth.py --start 0 --n 1 --gemini         # real Gemini API
  venv/bin/python export_ground_truth.py --start 0 --n 1 --openrouter     # free hosted Gemma-4
  venv/bin/python export_ground_truth.py --start 0 --n 1 --openrouter --openrouter-model nemotron-vl-12b
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


def main(
    start: int, n: int, out_dir: Path, use_vlm: bool, use_layoutlm: bool,
    use_vlm_direct: bool, use_vlm_two_call: bool, use_gemini: bool,
    use_openrouter: bool, openrouter_model: str,
) -> None:
    _quiet_rapidocr()
    df = pd.read_csv(ROOT / "Ground Truth.csv", low_memory=False)
    cols = [c for c in df.columns if "Filename" in c]
    all_names = sorted(p.name for p in DATASET.glob("*.jpg"))

    policy = router.Policy(use_local_vlm=use_vlm, union_free_tiers=use_vlm, allow_paid=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    alt_mode = use_vlm_direct or use_vlm_two_call or use_gemini or use_openrouter
    extract_fn = None
    if use_openrouter:
        from stage2.openrouter_direct import MODEL_CHOICES, extract_product_openrouter

        model_id = MODEL_CHOICES[openrouter_model]

        def extract_fn(paths, use_fallback):
            return extract_product_openrouter(paths, use_fallback=use_fallback, model=model_id)
    elif use_gemini:
        from stage2.gemini_direct import extract_product_gemini as extract_fn
    elif use_vlm_two_call:
        from stage2.vlm_direct import extract_product_two_call as extract_fn
    elif use_vlm_direct:
        from stage2.vlm_direct import extract_product_direct as extract_fn

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

        if alt_mode:
            extraction, vlm_dict = extract_fn(paths, use_fallback=use_layoutlm)
            if vlm_dict is None:
                print(f"  row{i}: call FAILED (no key/server unreachable, or every retry failed) "
                      f"- {'using fallback only' if use_layoutlm else 'result is empty'}")
        else:
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
    ap.add_argument("--no-layoutlm", action="store_true",
                     help="skip the LayoutLMv3 tier (or, with an alternative mode, skip its heuristic+LayoutLM fallback)")
    ap.add_argument("--vlm-direct", action="store_true",
                     help="local MLX VLM, direct image->JSON path (stage2/vlm_direct.py)")
    ap.add_argument("--vlm-two-call", action="store_true",
                     help="local MLX VLM, OCR-then-map two-call path (stage2/vlm_direct.py)")
    ap.add_argument("--gemini", action="store_true",
                     help="the real Gemini API (stage2/gemini_direct.py) - needs GEMINI_API_KEY")
    ap.add_argument("--openrouter", action="store_true",
                     help="a free, already-hosted model via openrouter.ai (stage2/openrouter_direct.py) "
                          "- needs OPENROUTER_API_KEY")
    ap.add_argument("--openrouter-model", default="gemma-4-26b",
                     choices=["gemma-4-26b", "nemotron-vl-12b", "nemotron-omni-30b"],
                     help="which free OpenRouter model to use with --openrouter (default: gemma-4-26b)")
    a = ap.parse_args()
    main(a.start, a.n, a.out, a.vlm, not a.no_layoutlm, a.vlm_direct, a.vlm_two_call, a.gemini,
         a.openrouter, a.openrouter_model)
