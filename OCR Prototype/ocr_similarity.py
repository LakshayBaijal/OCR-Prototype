"""Compare OCR text between two folders of matching-stem .txt files (produced by
batch_ocr.py) using fuzzy, character-level precision/recall.

WHY CHARACTER N-GRAMS, NOT WORDS: OCR output is noisy on both sides - the exact
same real word can come out spelled slightly differently between two runs (see
img16: raw reads "ΕΛΛΙΟΛΑΔΟ", enhanced reads a garbled variant). Word-level exact
match would score that as 100% wrong - a totally different word - even though
most of the letters agree. Character n-grams give partial credit for the parts
that DID come through correctly, so a small spelling drift costs a little, not
everything. This is the same idea as chrF (Popović 2015), the standard
character-n-gram metric for comparing generated text against a reference in
machine translation evaluation - and it is language-agnostic, so it works on
the Greek label text here exactly the same way as the English ones.

METRIC: for n in {1, 2, 3}, treat each text as a multiset of overlapping
character n-grams (after lowercasing + collapsing whitespace). For each n:
    overlap_n    = sum of min(count in raw, count in enhanced) over shared n-grams
    precision_n  = overlap_n / (total n-grams in enhanced)   "how much of what's
                   shown now was actually there before"
    recall_n     = overlap_n / (total n-grams in raw)        "how much of what
                   was really there survived"
Final precision/recall = average of precision_n / recall_n over n=1,2,3.

Undefined cases are reported as None, not 0 - a 0 would silently claim "totally
wrong" when the real answer is "nothing to compare against":
    raw text empty       -> recall is None (nothing existed to preserve)
    enhanced text empty  -> precision is None (nothing was produced to judge)
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

NOT_A_PRODUCT = {"embeddings", "embeddings_index"}
NGRAM_SIZES = (1, 2, 3)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def ngram_counts(text: str, n: int) -> Counter:
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def precision_recall(raw_text: str, enh_text: str) -> tuple[float | None, float | None]:
    raw_n, enh_n = normalize(raw_text), normalize(enh_text)
    precisions, recalls = [], []
    for n in NGRAM_SIZES:
        raw_c, enh_c = ngram_counts(raw_n, n), ngram_counts(enh_n, n)
        raw_total, enh_total = sum(raw_c.values()), sum(enh_c.values())
        overlap = sum(min(c, enh_c[g]) for g, c in raw_c.items())
        if enh_total:
            precisions.append(overlap / enh_total)
        if raw_total:
            recalls.append(overlap / raw_total)
    precision = sum(precisions) / len(precisions) if precisions else None
    recall = sum(recalls) / len(recalls) if recalls else None
    return precision, recall


def matched_stems(raw_dir: Path, enh_dir: Path) -> list[str]:
    raw = {p.stem for p in raw_dir.glob("*.txt")} - NOT_A_PRODUCT
    enh = {p.stem for p in enh_dir.glob("*.txt")} - NOT_A_PRODUCT
    return sorted(raw & enh, key=lambda s: (int("".join(c for c in s if c.isdigit()) or 0), s))


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: ocr_similarity.py <raw_dir> <enhanced_dir>")
    raw_dir, enh_dir = Path(sys.argv[1]), Path(sys.argv[2])
    stems = matched_stems(raw_dir, enh_dir)

    rows = []
    for stem in stems:
        raw_text = (raw_dir / f"{stem}.txt").read_text(encoding="utf-8")
        enh_text = (enh_dir / f"{stem}.txt").read_text(encoding="utf-8")
        precision, recall = precision_recall(raw_text, enh_text)
        rows.append({
            "stem": stem,
            "raw_text": raw_text,
            "enhanced_text": enh_text,
            "precision": precision,
            "recall": recall,
        })

    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
