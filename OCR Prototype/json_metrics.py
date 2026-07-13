"""
json_metrics.py — leaf-level comparison of a predicted product JSON vs ground truth
===================================================================================
The scoring for two of the three benchmarks:

  * eval_json      (image/OCR-text -> ProductExtraction)
  * eval_convert   (third-party record -> ProductExtraction)

Both call `score(gt, pred)`, which walks the two objects down to their scalar
leaves and returns three numbers, with fuzzy string / numeric matching and
order-independent lists:

  precision — of the prediction's populated leaves, how many are backed by the GT
  recall    — of the GT's populated leaves, how many the prediction recovered,
              PLUS credit for a field left empty in BOTH (a matching null/default).
              So an all-empty prediction reads high on recall — that's intended;
              F1 still flags it because its precision is 0.
  f1        — harmonic mean (the headline)

(The third benchmark, OCR, is scored separately by character accuracy —
see bench_common.cer_accuracy.)

Public API:
  score(gt, pred)        -> {"precision", "recall", "f1"}
  field_scores(gt, pred) -> {top_level_field: f1}     # per-field F1 breakdown
"""

from __future__ import annotations

import difflib
import re
from typing import Any

_EMPTY_STRINGS = {"", "null", "none", "n/a", "na", "-", "--"}


def _norm(s: Any) -> str:
    s = re.sub(r"\s+", " ", str(s).lower().strip())
    return s.strip(" \t.,:;-_/|®™©()[]")


def is_empty(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, str):
        return _norm(x) in _EMPTY_STRINGS
    if isinstance(x, (list, dict)):
        return len(x) == 0
    return False


def _num(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    m = re.findall(r"-?\d+\.?\d*", str(x))
    return float(m[0]) if m else None


def str_sim(a: Any, b: Any) -> float:
    """Fuzzy string similarity in [0, 1] = max(token-set F1, character ratio)."""
    na, nb = _norm(a), _norm(b)
    if na == nb:
        return 1.0
    if not na or not nb:
        return 0.0
    ta, tb = set(na.split()), set(nb.split())
    token_f1 = 2 * len(ta & tb) / (len(ta) + len(tb)) if (ta and tb) else 0.0
    ratio = difflib.SequenceMatcher(None, na, nb).ratio()
    return max(token_f1, ratio)


def scalar_sim(a: Any, b: Any) -> float:
    """Similarity of two scalar leaves in [0, 1]."""
    if isinstance(a, bool) or isinstance(b, bool):
        return 1.0 if a == b else 0.0
    na, nb = _num(a), _num(b)
    # numeric proximity only when BOTH look purely numeric (so "9 g" vs "9%" still
    # falls through to string similarity rather than matching loosely on the 9)
    if na is not None and nb is not None \
            and _norm(a).replace(".", "", 1).lstrip("-").isdigit() \
            and _norm(b).replace(".", "", 1).lstrip("-").isdigit():
        denom = max(abs(na), abs(nb), 1.0)
        return max(0.0, 1.0 - abs(na - nb) / denom)
    return str_sim(a, b)


def _leaf_count(a: Any) -> float:
    """Number of non-empty scalar leaves in `a` (its weight)."""
    if is_empty(a):
        return 0.0
    if isinstance(a, dict):
        return sum(_leaf_count(v) for v in a.values())
    if isinstance(a, list):
        if all(not isinstance(x, (dict, list)) for x in a):
            return float(len([x for x in a if not is_empty(x)]))
        return sum(_leaf_count(x) for x in a)
    return 1.0


def _coverage(a: Any, b: Any) -> tuple[float, float]:
    """How much of `a`'s populated content is present in `b`.
    Returns (matched_weight, total_weight); weight == number of leaves."""
    if is_empty(a):
        return 0.0, 0.0

    if isinstance(a, dict):
        s = w = 0.0
        for k, v in a.items():
            if is_empty(v):
                continue
            bv = b.get(k) if isinstance(b, dict) else None
            cs, cw = _coverage(v, bv)
            s += cs
            w += cw
        return s, w

    if isinstance(a, list):
        A = [x for x in a if not is_empty(x)]
        if not A:
            return 0.0, 0.0
        B = [x for x in b if not is_empty(x)] if isinstance(b, list) else []
        if all(not isinstance(x, (dict, list)) for x in A):
            # list of scalars: each item earns its best match among the others
            s = sum(max((scalar_sim(x, y) for y in B), default=0.0) for x in A)
            return s, float(len(A))
        # list of objects: greedy best-match by coverage ratio, weighted by leaves
        s = w = 0.0
        for x in A:
            xw = _leaf_count(x)
            best = max((cs / cw for y in B for cs, cw in [_coverage(x, y)] if cw), default=0.0)
            s += best * xw
            w += xw
        return s, w

    # scalar leaf
    if is_empty(b):
        return 0.0, 1.0
    return scalar_sim(a, b), 1.0


def score(gt: dict, pred: dict) -> dict:
    """{'precision', 'recall', 'f1'} for one (gt, pred) pair (fuzzy leaf match)."""
    rs, rw = _coverage(gt, pred)     # recall content:    GT leaves found in prediction
    ps, pw = _coverage(pred, gt)     # precision content: prediction leaves backed by GT
    # Credit fields empty in BOTH (a matching null/default): add to recall's
    # numerator AND denominator so a correct abstention counts as recovered.
    both_null = sum(1 for k in set(gt) | set(pred)
                    if is_empty(gt.get(k)) and is_empty(pred.get(k)))
    recall = (rs + both_null) / (rw + both_null) if (rw + both_null) else 1.0
    # empty prediction vs populated GT -> precision 0 (not a vacuous 1.0)
    precision = ps / pw if pw else (1.0 if rw == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def field_scores(gt: dict, pred: dict) -> dict:
    """Per-top-level-field F1, over fields populated on at least one side."""
    out = {}
    for k in sorted(set(gt) | set(pred)):
        gv, pv = gt.get(k), pred.get(k)
        if is_empty(gv) and is_empty(pv):
            continue
        out[k] = score({k: gv}, {k: pv})["f1"]
    return out
