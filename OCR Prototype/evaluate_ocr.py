"""Stage 1 benchmark.

There is no ground-truth transcription in the dataset, so Character Error Rate is not
computable. But the CSV holds the values that are physically PRINTED on the pack - so we
ask the question that actually matters downstream:

    Field recall: after OCR, can we still find the values we know are on this product?

Scored per PRODUCT, not per image: a product has 3-5 faces and the FSSAI number lives on
exactly one of them, so scoring the front-of-pack shot against it would be nonsense. We
union the OCR text across a product's images, then look for each field.

Answers four questions:
  1. Does preprocessing (Stage 0) actually improve OCR, or is it decoration?
  2. What fraction of images does the free tier fail on? (this prices the paid tier)
  3. Does the local VLM rescue those failures?
  4. Is there non-Latin script we are silently dropping?

Usage:  venv/bin/python evaluate_ocr.py --n 60
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from pathlib import Path

# Benchmarks run Stage 0 with the FAST heuristic segmenter, NOT the BiRefNet matte
# (~10s/image), so a 200-product run finishes in minutes instead of hours. The matte is a
# crop-QUALITY improvement for the app (stops it damaging glossy packs); it does not
# materially change OCR field recall - measured ~equal. Set BGREMOVE=1 to benchmark the exact
# app pipeline instead. (setdefault: an explicit BGREMOVE=1 in the environment still wins.)
os.environ.setdefault("BGREMOVE", "0")

warnings.filterwarnings("ignore")

import pandas as pd

import products
from json_metrics import _norm
from ocr import router
from ocr.types import OCRResult
from preprocessing import load_rgb, run_pipeline

ROOT = Path(__file__).parent
DATASET = ROOT / "Dataset"

# field -> how to look for it in raw OCR text
NUMERIC = {"Mrp", "Fssai No", "Upc / Ean"}
TEXTUAL = {"Brand", "Manufacturer"}
FIELDS = ["Brand", "Mrp", "Net Quantity", "Fssai No", "Upc / Ean", "Manufacturer"]


def digits(value) -> str:
    """Digit signature of a value. 155.0 -> '155' (not '1550'), so the float formatting the
    CSV happens to use does not corrupt the string we search for."""
    s = str(value).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return re.sub(r"\D", "", s)


def found(field: str, value, text: str) -> bool | None:
    """Is this ground-truth value recoverable from the OCR text? None = not in the GT."""
    if pd.isna(value) or not str(value).strip() or str(value).strip() in {"-", "--"}:
        return None

    if field in NUMERIC:
        d = digits(value)
        return bool(d) and d in re.sub(r"\D", "", text)

    if field == "Net Quantity":
        # "25 units" is the CSV's normalisation; the pack says "25 Tea Bags". Only the
        # NUMBER is genuinely printed, so that is all we can honestly look for.
        d = re.sub(r"\D", "", str(value))
        return bool(d) and d in re.sub(r"\D", "", text)

    v, t = _norm(value), _norm(text)
    if not v:
        return None
    if v in t:
        return True
    toks = v.split()
    return sum(1 for tok in toks if tok in t) / len(toks) >= 0.75


def ocr_text(path: Path, preprocess: bool, policy: router.Policy) -> tuple[str, list[OCRResult]]:
    """OCR one image, optionally through Stage 0 first."""
    if preprocess:
        _, crops = run_pipeline(str(path))
    else:
        crops = [load_rgb(str(path))]

    results = [router.read(c, policy) for c in crops]
    return "\n".join(r.text for r in results), results


def main(n: int, use_vlm: bool) -> None:
    policy = router.Policy(
        use_local_vlm=use_vlm, union_free_tiers=use_vlm, allow_paid=False
    )
    df = pd.read_csv(ROOT / "Ground Truth.csv", low_memory=False)
    img_cols = [c for c in df.columns if "Filename" in c]
    all_names = sorted(p.name for p in DATASET.glob("*.jpg"))

    sample = df.sample(n=min(n, len(df)), random_state=7)

    recall = {f: {"pre": 0, "raw": 0, "n": 0} for f in FIELDS}
    weak = total_imgs = non_latin = 0
    escalated_files: list[str] = []

    for _, row in sample.iterrows():
        names = products.group_for(
            [str(row[c]).strip() for c in img_cols
             if isinstance(row[c], str) and str(row[c]).strip()],
            all_names,
        )
        paths = [DATASET / n for n in names if (DATASET / n).exists()]
        if not paths:
            continue

        pre_all, raw_all = [], []
        for p in paths:
            t_pre, res_pre = ocr_text(p, True, policy)
            t_raw, _ = ocr_text(p, False, policy)
            pre_all.append(t_pre)
            raw_all.append(t_raw)

            for r in res_pre:
                total_imgs += 1
                if any(s.startswith("ESCALATE") for s in r.trail):
                    weak += 1
                    escalated_files.append(p.name)
                if r.has_non_latin():
                    non_latin += 1

        pre_text, raw_text = "\n".join(pre_all), "\n".join(raw_all)

        for f in FIELDS:
            got_pre = found(f, row.get(f), pre_text)
            if got_pre is None:
                continue
            recall[f]["n"] += 1
            recall[f]["pre"] += int(got_pre)
            recall[f]["raw"] += int(bool(found(f, row.get(f), raw_text)))

    # ---------------------------------------------------------------- report
    print(f"\n{'='*70}")
    tiers = "rapidocr UNION local VLM" if use_vlm else "rapidocr only"
    print(f"OCR BENCHMARK - {len(sample)} products, {total_imgs} images, free ({tiers})")
    seg = "BiRefNet matte" if os.environ.get("BGREMOVE") != "0" else "fast heuristic (BGREMOVE=0)"
    print(f"Stage-0 segmenter: {seg}")
    print(f"{'='*70}\n")

    print("1. FIELD RECALL - can we still find what we know is printed on the pack?")
    print(f"   (scored per product: OCR text unioned across its {len(img_cols)} faces)\n")
    print(f"   {'field':16} {'preprocessed':>13} {'raw':>8}   {'delta':>7}   n")
    print(f"   {'-'*56}")
    for f in FIELDS:
        d = recall[f]
        if not d["n"]:
            continue
        p, r = d["pre"] / d["n"], d["raw"] / d["n"]
        print(f"   {f:16} {p:12.0%} {r:7.0%}   {p-r:+6.0%}   {d['n']}")

    tot_n = sum(d["n"] for d in recall.values())
    tot_p = sum(d["pre"] for d in recall.values())
    tot_r = sum(d["raw"] for d in recall.values())
    if tot_n:
        print(f"   {'-'*56}")
        print(f"   {'OVERALL':16} {tot_p/tot_n:12.0%} {tot_r/tot_n:7.0%}   "
              f"{(tot_p-tot_r)/tot_n:+6.0%}   {tot_n}")

    print(f"\n2. ESCALATION RATE (prices the paid tier)")
    if total_imgs:
        rate = weak / total_imgs
        print(f"   {weak}/{total_imgs} images ({rate:.0%}) failed the free tier's quality bar")
        print(f"   -> at that rate, Google Vision for all 20,746 images would cost "
              f"${rate * 20746 * 0.0015:,.2f}")

    print(f"\n3. NON-LATIN SCRIPT (do we need Indic support?)")
    print(f"   {non_latin}/{total_imgs} images contained non-Latin characters")

    if escalated_files:
        print(f"\n4. SAMPLE OF IMAGES THE FREE TIER STRUGGLED ON")
        for f in escalated_files[:8]:
            print(f"   {f}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="number of products to sample")
    ap.add_argument("--vlm", action="store_true",
                    help="enable the free local-VLM tier (slow; ~4GB model download)")
    a = ap.parse_args()
    import evaluation
    with evaluation.capture("evaluate_ocr"):
        main(a.n, a.vlm)
