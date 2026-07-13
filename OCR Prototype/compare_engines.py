"""Head-to-head: does the free local VLM actually rescue the images RapidOCR fails on?

This is the decisive question for a free-only pipeline. RapidOCR is fast but weak (40% field
recall, and it hallucinates Chinese glyphs on Devanagari). If the local VLM cannot beat it on
the hard cases, the second free tier is not worth its 3.1GB and we need a different answer.

Runs ONLY on images the router flagged as weak, so we spend GPU time where it matters.

Usage:  venv/bin/python compare_engines.py --n 12
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from evaluate_ocr import FIELDS, found
from ocr import local_vlm, rapid, router
from preprocessing import load_rgb

ROOT = Path(__file__).parent
DATASET = ROOT / "Dataset"


def cjk(text: str) -> int:
    """Chinese glyphs are pure hallucination on Indian packaging - a direct quality signal."""
    return sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF)


def devanagari(text: str) -> int:
    return sum(1 for c in text if 0x900 <= ord(c) <= 0x97F)


def main(n: int) -> None:
    df = pd.read_csv(ROOT / "Ground Truth.csv", low_memory=False)
    img_cols = [c for c in df.columns if "Filename" in c]

    # Find images where the free tier 1 is weak - those are the only ones worth the VLM's time.
    weak: list[tuple[Path, pd.Series]] = []
    for _, row in df.sample(n=400, random_state=3).iterrows():
        for c in img_cols:
            v = row.get(c)
            if not isinstance(v, str):
                continue
            p = DATASET / v.strip()
            if not p.exists():
                continue
            res = router.read(load_rgb(str(p)))  # cached; tier 1 only
            if router.weakness(res, router.Policy()):
                weak.append((p, row))
                break
        if len(weak) >= n:
            break

    print(f"\n{'='*76}")
    print(f"RAPIDOCR  vs  LOCAL VLM (Qwen2.5-VL-3B 4-bit)  -  {len(weak)} images RapidOCR struggled on")
    print(f"{'='*76}\n")

    agg = {"rapid": {"chars": 0, "hits": 0, "cjk": 0, "dev": 0, "n": 0},
           "vlm": {"chars": 0, "hits": 0, "cjk": 0, "dev": 0, "n": 0}}

    for p, row in weak:
        rgb = load_rgb(str(p))
        r = rapid.run(rgb)
        v = local_vlm.run(rgb)

        for tag, res in (("rapid", r), ("vlm", v)):
            hits = sum(1 for f in FIELDS if found(f, row.get(f), res.text) is True)
            agg[tag]["chars"] += res.char_count
            agg[tag]["hits"] += hits
            agg[tag]["cjk"] += cjk(res.text)
            agg[tag]["dev"] += devanagari(res.text)
            agg[tag]["n"] += 1

        print(f"{p.name}")
        print(f"   rapidocr : {r.char_count:4} chars, {sum(1 for f in FIELDS if found(f, row.get(f), r.text) is True)} fields, "
              f"{cjk(r.text)} CJK, {devanagari(r.text)} devanagari  ({r.elapsed:.1f}s)")
        print(f"   local_vlm: {v.char_count:4} chars, {sum(1 for f in FIELDS if found(f, row.get(f), v.text) is True)} fields, "
              f"{cjk(v.text)} CJK, {devanagari(v.text)} devanagari  ({v.elapsed:.1f}s)")

    print(f"\n{'-'*76}")
    print(f"{'engine':12} {'chars':>8} {'fields found':>14} {'CJK (junk)':>12} {'devanagari':>12}")
    print(f"{'-'*76}")
    for tag in ("rapid", "vlm"):
        a = agg[tag]
        print(f"{tag:12} {a['chars']:8} {a['hits']:14} {a['cjk']:12} {a['dev']:12}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    main(ap.parse_args().n)
