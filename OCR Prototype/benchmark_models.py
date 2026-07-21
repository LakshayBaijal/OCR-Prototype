"""Benchmark every FREE OCR engine on the PREPROCESSED crops.

All engines here are open-source and run entirely on this machine. Nothing is billed, no
tokens are spent, nothing leaves the laptop.

Scored on what actually matters downstream:

  fields   - ground-truth values recovered (Brand/MRP/NetQty/FSSAI/UPC/Manufacturer).
             THE headline number. More text is worthless if it is not the text we need.
  ceiling  - the same, but counting only products whose back panel was photographed.
             40% of products never had their panel shot, so their MRP/FSSAI/barcode are in
             NO image and no engine can ever read them. Scoring against 100% punishes every
             engine for missing data that does not exist.
  chars    - raw volume of text (a tiebreak, not a goal)
  CJK      - hallucinated Chinese glyphs. RapidOCR's default model is English+Chinese, so it
             emits Chinese when it meets Devanagari. This is JUNK being injected into our text.
  deva     - real Devanagari captured (Hindi is genuinely on these packs)
  sec/img  - wall clock

Add or remove engines in ENGINES below. To benchmark a different VLM, just restart the MLX
server with a different --model; `local_vlm` follows whatever is being served.

Usage:
  venv/bin/python benchmark_models.py --n 25
  venv/bin/python benchmark_models.py --n 25 --raw     # skip Stage 0, to prove it earns its keep
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

# Fast heuristic Stage-0 segmenter by default (not the ~10s/image BiRefNet matte), so runs
# finish in minutes. Set BGREMOVE=1 to use the model. See evaluate_ocr.py for the rationale.
os.environ.setdefault("BGREMOVE", "0")

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# EasyOCR fetches its weights over HTTPS with plain urllib, which on this python.org build
# has no CA bundle wired up ("CERTIFICATE_VERIFY_FAILED"). Point the default TLS context at
# certifi's bundle - verification still happens, it just now knows where the roots are.
import ssl

import certifi

ssl._create_default_https_context = lambda *a, **k: ssl.create_default_context(
    cafile=certifi.where()
)

import products
from ceiling import has_panel
from evaluate_ocr import FIELDS, found
from ocr import local_vlm, rapid
from preprocessing import load_rgb, run_pipeline

ROOT = Path(__file__).parent
DATASET = ROOT / "Dataset"

PANEL_FIELDS = {"Mrp", "Fssai No", "Upc / Ean", "Manufacturer"}


def cjk(t: str) -> int:
    return sum(1 for c in t if 0x4E00 <= ord(c) <= 0x9FFF)


def deva(t: str) -> int:
    return sum(1 for c in t if 0x900 <= ord(c) <= 0x97F)


# ---------------------------------------------------------------- engines


def engine_rapidocr(crops: list[np.ndarray]) -> str:
    return "\n".join(rapid.run(c).text for c in crops)


def engine_local_vlm(crops: list[np.ndarray]) -> str:
    return "\n".join(local_vlm.run(c).text for c in crops)


_easy = None


def engine_easyocr(crops: list[np.ndarray]) -> str:
    """EasyOCR with English + Hindi. The point of including it is the Devanagari model:
    RapidOCR ships English+Chinese, so on Hindi text it emits Chinese glyphs - it is not
    merely missing Hindi, it is inventing junk."""
    global _easy
    if _easy is None:
        import easyocr

        _easy = easyocr.Reader(["en", "hi"], gpu=True, verbose=False)

    out: list[str] = []
    for c in crops:
        out.extend(_easy.readtext(c, detail=0))  # detail=0 -> list[str]
    return "\n".join(out)


def engine_union(crops: list[np.ndarray]) -> str:
    """All three free engines, merged - the chosen default.

    They fail on DIFFERENT things, so the union strictly dominates any single one:
      - rapidocr : digits (barcode, FSSAI, MRP) + bounding boxes, but hallucinates Chinese
                   on Devanagari
      - easyocr  : reads the Hindi rapidocr invents Chinese for, zero CJK junk
      - local_vlm: clean prose and fields the other two miss entirely
    All free, and latency is irrelevant here, so there is no reason not to run all three.
    """
    return (
        engine_rapidocr(crops) + "\n" + engine_easyocr(crops) + "\n" + engine_local_vlm(crops)
    )


ENGINES = {
    "rapidocr": engine_rapidocr,
    "easyocr": engine_easyocr,
    "local_vlm": engine_local_vlm,
    "union": engine_union,
}


# ---------------------------------------------------------------- run


def main(n: int, raw: bool, only: list[str] | None) -> None:
    df = pd.read_csv(ROOT / "Ground Truth.csv", low_memory=False)
    cols = [c for c in df.columns if "Filename" in c]
    all_names = sorted(p.name for p in DATASET.glob("*.jpg"))
    sample = df.sample(n=min(n, len(df)), random_state=7)

    names = only or list(ENGINES)
    stat = {
        e: {"got": 0, "gt": 0, "reach": 0, "chars": 0, "cjk": 0, "deva": 0, "t": 0.0, "err": 0}
        for e in names
    }
    n_prod = 0

    for _, row in sample.iterrows():
        img_names = products.group_for(
            [str(row[c]).strip() for c in cols
             if isinstance(row[c], str) and str(row[c]).strip()],
            all_names,
        )
        paths = [DATASET / n for n in img_names if (DATASET / n).exists()]
        if not paths:
            continue
        n_prod += 1

        # Stage 0 once per product; every engine gets the SAME pixels.
        crops: list[np.ndarray] = []
        for p in paths:
            if raw:
                crops.append(load_rgb(str(p)))
            else:
                crops.extend(run_pipeline(str(p))[1])

        panel = None
        for name in names:
            fn = ENGINES[name]
            t0 = time.perf_counter()
            try:
                text = fn(crops)
            except Exception as exc:
                stat[name]["err"] += 1
                print(f"  ! {name} failed on {paths[0].name}: {type(exc).__name__}: {exc}"[:110])
                continue
            stat[name]["t"] += time.perf_counter() - t0

            if panel is None:
                panel = has_panel(text)

            stat[name]["chars"] += len(text)
            stat[name]["cjk"] += cjk(text)
            stat[name]["deva"] += deva(text)

            for f in FIELDS:
                got = found(f, row.get(f), text)
                if got is None:
                    continue
                stat[name]["gt"] += 1
                stat[name]["reach"] += (panel if f in PANEL_FIELDS else True)
                stat[name]["got"] += bool(got)

    n_img = sum(1 for _ in range(0))  # not tracked per engine; time is per product
    print(f"\n{'='*88}")
    print(f"FREE OCR ENGINES on {'RAW' if raw else 'PREPROCESSED'} images - {n_prod} products")
    print(f"{'='*88}\n")
    print(f"  {'engine':12} {'fields':>8} {'vs ceiling':>11} {'chars':>8} "
          f"{'CJK junk':>9} {'devanagari':>11} {'sec/product':>12} {'errors':>7}")
    print(f"  {'-'*84}")
    for e in names:
        s = stat[e]
        if not s["gt"]:
            print(f"  {e:12} {'(no results)':>8}")
            continue
        f_pct = s["got"] / s["gt"]
        c_pct = s["got"] / s["reach"] if s["reach"] else 0
        print(f"  {e:12} {f_pct:8.0%} {c_pct:11.0%} {s['chars']:8} "
              f"{s['cjk']:9} {s['deva']:11} {s['t']/max(n_prod,1):11.1f}s {s['err']:7}")
    print()
    print("  fields     = ground-truth values recovered (the number that matters)")
    print("  vs ceiling = same, ignoring products whose back panel was never photographed")
    print("  CJK junk   = hallucinated Chinese glyphs (lower is better; 0 is correct)")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--raw", action="store_true", help="skip Stage 0 preprocessing")
    ap.add_argument("--only", nargs="*", help=f"subset of: {list(ENGINES)}")
    a = ap.parse_args()
    import evaluation
    with evaluation.capture("benchmark_models"):
        main(a.n, a.raw, a.only)
