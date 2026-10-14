"""What the pixels say, as comparable SETS - the core of the OCR-only verifier.

This module exists because of how the previous design failed. It asked "did the Brand
field change?", which forces something to first decide WHICH line of text is the brand
- and `stage2/heuristic.py` decided that by picking the physically largest line
(`layout:largest-text`, its own confidence 0.4). On a shelf photo that is a coin flip:
it read "COCO" on one image and "Kelloys" on the other and reported a brand change on a
box where the brand never moved. Nearly every false finding in the audit traced back to
that guess, not to OCR misreading characters.

So nothing here identifies fields. It extracts SETS of things OCR is actually reliable
at seeing, and compares the sets:

    tokens        every word-ish string          -> retention / invention
    digit_runs    every run of >=3 digits        -> the strongest single signal
    measures      every number+unit, canonical   -> "500 g", "1 kg" == "1000 g"
    prices        every currency-marked amount   -> MRP-ish values, unlabelled

"Did 200 g become 258 g?" is answerable from `measures` alone, with no idea which field
either belonged to. That is the whole point: the question the verifier actually needs
answered never required field identification in the first place.

Digits carry most of the weight here deliberately. They are what OCR reads best (see
`benchmark_models.py`, which records RapidOCR as the engine that recovers "digits
(barcode, FSSAI, MRP)"), and they are what matters commercially - a price, a weight, a
licence number, a barcode. A digit run present in the enhanced image and absent from
the original is fabrication, and needs no interpretation to call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as _dc_field

# --------------------------------------------------------------------------- #
# OCR confusion - characters the recogniser genuinely swaps on poor crops.
# --------------------------------------------------------------------------- #
#
# A single swap from this table is an OCR artifact, not evidence the pack changed.
# Applied by folding both sides to a canonical member before comparing, so "1OO g" and
# "100 g" compare equal. Kept deliberately small: only pairs that are visually
# near-identical in common pack typefaces. Widening it would start hiding real
# differences, which is the opposite of this tool's job.
_CONFUSION_FOLD = str.maketrans({
    "O": "0", "o": "0", "D": "0", "Q": "0",
    "I": "1", "l": "1", "i": "1", "|": "1",
    "S": "5", "s": "5",
    "B": "8",
    "G": "6",
    "Z": "2", "z": "2",
})

# Canonical unit multipliers, so "1 kg" == "1000 g" and "1 l" == "1000 ml". Same table
# as verify.compare uses - kept here too so this module stands alone as "what the
# pixels say", with no dependency on the comparison layer.
UNIT_TO_BASE = {
    "kg": ("g", 1000.0), "kgs": ("g", 1000.0), "g": ("g", 1.0), "gm": ("g", 1.0),
    "gms": ("g", 1.0), "gr": ("g", 1.0), "gram": ("g", 1.0), "grams": ("g", 1.0),
    "mg": ("mg", 1.0), "mcg": ("mg", 0.001),
    "l": ("ml", 1000.0), "lt": ("ml", 1000.0), "ltr": ("ml", 1000.0),
    "litre": ("ml", 1000.0), "litres": ("ml", 1000.0), "liter": ("ml", 1000.0),
    "ml": ("ml", 1.0), "cl": ("ml", 10.0),
}

_GROUPING_RE = re.compile(r"(?<=\d),(?=\d)")          # 1,299 -> 1299
_UNIT_ALT = "|".join(sorted(UNIT_TO_BASE, key=len, reverse=True))
_MEASURE_RE = re.compile(rf"(\d+(?:[.,]\d+)?)\s*({_UNIT_ALT})\b", re.I)
_PRICE_RE = re.compile(r"(?:rs\.?|inr|₹|€|£)\s*(\d+(?:[.,]\d{1,2})?)", re.I)
_DIGIT_RUN_RE = re.compile(r"\d+")
_TOKEN_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)   # word-ish, >=3 letters, no digits


def fold(s: str) -> str:
    """Lower-case and fold OCR-confusable characters, for comparison only."""
    return str(s or "").translate(_CONFUSION_FOLD).lower()


def tokens(text: str) -> set[str]:
    """Word-ish tokens (>=3 letters). Digits are handled separately by `digit_runs` -
    mixing them in would let a changed number hide inside a high word-overlap score."""
    return {t.lower() for t in _TOKEN_RE.findall(str(text or ""))}


def digit_runs(text: str, min_len: int = 3) -> set[str]:
    """Every contiguous run of >=`min_len` digits.

    Short runs are excluded on purpose: 1-2 digit numbers ("2%", "5") collide by chance
    constantly and would swamp the signal. Three digits is where a run starts being
    specific enough that its presence or absence means something.
    """
    cleaned = _GROUPING_RE.sub("", str(text or ""))
    return {m.group() for m in _DIGIT_RUN_RE.finditer(cleaned) if len(m.group()) >= min_len}


def measures(text: str) -> set[tuple[str, float]]:
    """Every number+unit in the text, canonicalised to a base unit.

    Returns e.g. {('g', 500.0), ('ml', 330.0)}. This is what answers "did 200 g become
    258 g" without anyone having to decide which of them was 'the' net quantity.
    """
    out = set()
    for m in _MEASURE_RE.finditer(_GROUPING_RE.sub("", str(text or ""))):
        unit = m.group(2).lower()
        if unit not in UNIT_TO_BASE:
            continue
        base, mult = UNIT_TO_BASE[unit]
        try:
            value = float(m.group(1).replace(",", "."))
        except ValueError:
            continue
        out.add((base, round(value * mult, 3)))
    return out


def prices(text: str) -> set[float]:
    """Every currency-marked amount. Unlabelled - a pack's MRP, a shelf tag and a
    per-unit price all land here together, which is correct for a SET comparison: we
    are asking whether the same amounts are present, not which one is the MRP."""
    out = set()
    for m in _PRICE_RE.finditer(_GROUPING_RE.sub("", str(text or ""))):
        try:
            out.add(round(float(m.group(1).replace(",", ".")), 2))
        except ValueError:
            continue
    return out


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #


@dataclass
class SetDiff:
    """One set compared before/after, plus the fractions that summarise it."""

    label: str
    retained: set = _dc_field(default_factory=set)
    lost: set = _dc_field(default_factory=set)       # in original, gone from enhanced
    invented: set = _dc_field(default_factory=set)   # in enhanced, absent from original

    @property
    def original_count(self) -> int:
        return len(self.retained) + len(self.lost)

    @property
    def enhanced_count(self) -> int:
        return len(self.retained) + len(self.invented)

    @property
    def retention(self) -> float | None:
        """Fraction of the original's items that survived. None when the original had
        nothing to survive - which is 'not measurable', not 'perfect'."""
        if self.original_count == 0:
            return None
        return len(self.retained) / self.original_count

    @property
    def invention(self) -> float | None:
        """Fraction of the enhanced image's items that were not in the original."""
        if self.enhanced_count == 0:
            return None
        return len(self.invented) / self.enhanced_count

    def summary(self) -> str:
        r = f"{self.retention:.0%}" if self.retention is not None else "n/a"
        i = f"{self.invention:.0%}" if self.invention is not None else "n/a"
        return (f"{len(self.retained)}/{self.original_count} kept ({r}); "
                f"{len(self.invented)} new ({i} of the enhanced image's)")


def diff_sets(label: str, original: set, enhanced: set, fold_text: bool = False) -> SetDiff:
    """Compare two sets. `fold_text` applies OCR-confusion folding first, so "1OO" and
    "100" are treated as the same item rather than as a loss plus an invention."""
    if fold_text:
        o_map = {fold(str(x)): x for x in original}
        e_map = {fold(str(x)): x for x in enhanced}
        o_keys, e_keys = set(o_map), set(e_map)
        return SetDiff(
            label=label,
            retained={o_map[k] for k in o_keys & e_keys},
            lost={o_map[k] for k in o_keys - e_keys},
            invented={e_map[k] for k in e_keys - o_keys},
        )
    return SetDiff(label=label, retained=original & enhanced,
                   lost=original - enhanced, invented=enhanced - original)


def measures_close(a: tuple[str, float], b: tuple[str, float], tol: float = 0.01) -> bool:
    """Same base unit and within 1% - absorbs "0.5 kg" vs "500 g" float noise only."""
    if a[0] != b[0]:
        return False
    bigger = max(abs(a[1]), abs(b[1]), 1e-9)
    return abs(a[1] - b[1]) / bigger <= tol


def diff_measures(original: set, enhanced: set) -> SetDiff:
    """Measure comparison with unit-aware tolerance, so 500.0 g and 500.001 g are one
    item rather than a loss and an invention."""
    retained, lost = set(), set()
    remaining = set(enhanced)
    for o in original:
        hit = next((e for e in remaining if measures_close(o, e)), None)
        if hit is None:
            lost.add(o)
        else:
            retained.add(o)
            remaining.discard(hit)
    return SetDiff(label="quantities", retained=retained, lost=lost, invented=remaining)


def prominent_lines(lines, top_n: int = 5, min_chars: int = 3) -> list[str]:
    """The largest text lines, biggest first - the brand/product-name region without
    ever claiming to know which is which.

    The old design picked the single largest line and CALLED it the Brand. This returns
    several and labels none: the check built on it asks only "does this prominent
    string still appear somewhere in the other image's text", which is a question OCR
    can answer, unlike "which of these is the brand".
    """
    scored = []
    for ln in lines:
        text = (ln.text or "").strip()
        if len(text) < min_chars or not _TOKEN_RE.search(text):
            continue
        ys = [p[1] for p in ln.bbox]
        scored.append((max(ys) - min(ys), text))
    scored.sort(key=lambda t: -t[0])
    return [t for _, t in scored[:top_n]]


def contains_fuzzy(needle: str, haystack: str, threshold: float = 0.82) -> tuple[bool, float]:
    """Does `needle` appear in `haystack`, allowing OCR noise? Returns (found, score).

    Slides a window the length of the needle across the haystack and takes the best
    similarity, both OCR-folded. This is the containment test that replaces field-to-
    field brand comparison: it asks whether the original's prominent text SURVIVES
    anywhere in the enhanced image, which does not require either side's parser to have
    labelled it the same way.
    """
    import difflib

    n, h = fold(needle).replace(" ", ""), fold(haystack).replace(" ", "")
    if not n:
        return False, 0.0
    if n in h:
        return True, 1.0
    if not h:
        return False, 0.0

    best = 0.0
    step = max(1, len(n) // 4)
    for start in range(0, max(1, len(h) - len(n) + 1), step):
        window = h[start:start + len(n)]
        best = max(best, difflib.SequenceMatcher(None, n, window).ratio())
        if best >= threshold:
            return True, best
    return best >= threshold, best
