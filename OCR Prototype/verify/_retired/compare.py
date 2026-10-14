"""Shape-aware field comparison: did the enhanced image keep the original's value?

The naive check ("are these two strings equal") is wrong here and produces mostly
noise: OCR legitimately reads the same printed price as "75", "75.00" and "Rs 75",
and the same pack size as "400 g" and "400gm". Comparing those as strings reports a
corruption that never happened, and a reviewer who sees ten false alarms stops
reading the eleventh - which is the real one.

So the comparison is chosen from the SHAPE of the value, not the field's name
(ported from `ondc-intelligence/app/domain/consistency.py`, which solved the same
problem for the `product.verify` capability):

    value looks like...           comparison                 example
    ---------------------------   ------------------------   ----------------------
    a long run of digits (>=8)    exact digit equality       barcode, FSSAI licence
    a number followed by a unit   canonical measure          "1 kg" == "1000 g"
    a plain number                numeric equality           "75.00" == "75"
    a long block of prose         token-overlap ratio        ingredients, nutrition
    anything else (short text)    fuzzy string ratio         brand, name, country

Deliberately NOT ported: the GloVe semantic step. There it compared a caller's
free-text claim against what the image showed - two independently-worded strings, so
meaning-level similarity earned its 68 MB. Here both sides are OCR of THE SAME
PRINTED WORDS, so any real difference is a character-level corruption ("Amul" ->
"Arnul"), exactly what a fuzzy ratio is built to catch and what a semantic model
would smooth over. Dropping it also keeps this package free of the vector blob.

Digit fields are the one place where "close" is not good enough: a barcode one digit
off is a DIFFERENT product, not a near miss, so those compare exactly while
everything else gets a threshold.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------- thresholds

FUZZY_THRESHOLD = 0.82   # short text (brand, product name, manufacturer, country)
PROSE_THRESHOLD = 0.70   # long blocks (ingredients, nutrition) - OCR drops words in
                         # fine print far more often than it does in a 2-word brand,
                         # so holding prose to the same bar as a brand name would
                         # flag almost every ingredients list.
MEASURE_TOLERANCE = 0.01  # 1% - absorbs "0.5 kg" vs "500 g" float noise, nothing more

# Canonical unit multipliers, so "1 kg" == "1000 g" and "1 l" == "1000 ml".
UNIT_TO_BASE = {
    "kg": ("g", 1000.0), "g": ("g", 1.0), "gm": ("g", 1.0), "gms": ("g", 1.0),
    "gram": ("g", 1.0), "grams": ("g", 1.0), "mg": ("mg", 1.0), "mcg": ("mg", 0.001),
    "l": ("ml", 1000.0), "ltr": ("ml", 1000.0), "litre": ("ml", 1000.0),
    "liter": ("ml", 1000.0), "ml": ("ml", 1.0),
}

# Values that mean "nothing here", so they never count as a real reading.
NULLISH = {"", "-", "n/a", "na", "none", "not found", "not visible", "null", "unknown",
           "not applicable", "not printed", "not legible"}

# Thousands separators, removed before a number is parsed. Indian price fields carry
# them as a rule ("1,299"), and a plain number regex stops at the first comma - so an
# MRP of "1,299" against "1299" would compare as 1 vs 1299.
_GROUPING_RE = re.compile(r"(?<=\d),(?=\d)")


# ---------------------------------------------------------------- verdicts


class Status(str, Enum):
    """What happened to one field between the original and the enhanced image."""

    MATCH = "MATCH"        # present in both, and they agree
    CHANGED = "CHANGED"    # present in both, and they DISAGREE - the corruption case
    LOST = "LOST"          # readable in the original, gone from the enhanced image
    ADDED = "ADDED"        # not readable in the original, present in the enhanced one
    ABSENT = "ABSENT"      # neither image showed it - nothing to verify


# Below this, a field's extracted value is too weakly evidenced to reject an image
# over. `stage2/heuristic.py` scores its own extractors honestly and the spread is
# wide: a labelled FSSAI match is 0.95x the OCR line confidence, a labelled MRP 0.9x,
# but Brand-by-"biggest text on the pack" is only 0.4x and Product-Name-by-top-lines
# 0.35x. Those last two are explicitly layout GUESSES, and two guesses disagreeing is
# not evidence that the enhancer changed anything - it is evidence that the guess is
# unstable. Comparisons below this bar still surface as findings; they just route to
# REVIEW instead of REJECT.
CONFIDENCE_FLOOR = 0.5


@dataclass
class FieldComparison:
    field: str
    original: str | None
    enhanced: str | None
    status: Status
    score: float | None      # 0..1 similarity where one was computed, else None
    method: str              # which comparison ran, for the audit trail
    note: str = ""           # human-readable "why", shown in the UI

    # How well-evidenced each side's value was, straight from ProductExtraction.Field
    # (the extractor's own confidence and which rule produced it).
    original_confidence: float = 1.0
    enhanced_confidence: float = 1.0
    original_source: str = ""
    enhanced_source: str = ""

    @property
    def weakly_evidenced(self) -> bool:
        """True when either side's value is a low-confidence guess - see
        CONFIDENCE_FLOOR. Only meaningful for CHANGED, where both sides read
        something; presence outcomes are about absence, which has no confidence."""
        if self.status is not Status.CHANGED:
            return False
        return min(self.original_confidence, self.enhanced_confidence) < CONFIDENCE_FLOOR


# ---------------------------------------------------------------- parsing helpers


def norm(s: object) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def is_nullish(s: object) -> bool:
    return norm(s) in NULLISH


def digits(s: object) -> str:
    """Digit string, tolerating spaces and a trailing '.0' from float-ish values."""
    s = re.sub(r"\.0+$", "", str(s or "").strip())
    return re.sub(r"\D", "", s)


def as_number(s: object) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", _GROUPING_RE.sub("", str(s or "")))
    return float(m.group()) if m else None


def parse_measure(s: object) -> tuple[str, float] | None:
    """('g', 400.0) from '400 gm', canonicalised to a base unit; None if no unit."""
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([a-zA-Z]+)", _GROUPING_RE.sub("", str(s or "")))
    if not m:
        return None
    unit = m.group(2).lower()
    if unit not in UNIT_TO_BASE:
        return None
    base, mult = UNIT_TO_BASE[unit]
    return base, float(m.group(1)) * mult


def partial_ratio(a: str, b: str) -> float:
    """Best score of the shorter string against any window of the longer one, so a
    front-of-pack name still matches a fuller printed title with a trailing size."""
    short, long = sorted((a, b), key=len)
    if not short:
        return 0.0
    best = 0.0
    for _, j, _ in difflib.SequenceMatcher(None, short, long).get_matching_blocks():
        window = long[j:j + len(short)]
        best = max(best, difflib.SequenceMatcher(None, short, window).ratio())
    return best


def fuzzy_ratio(a: object, b: object) -> float:
    """0..1 similarity robust to word order, subsets, and truncation."""
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    ta, tb = set(a.split()), set(b.split())
    inter = " ".join(sorted(ta & tb))
    ra, rb = " ".join(sorted(ta)), " ".join(sorted(tb))
    scores = [difflib.SequenceMatcher(None, a, b).ratio(),
              difflib.SequenceMatcher(None, ra, rb).ratio(),
              partial_ratio(a, b)]
    if inter:
        scores.append(difflib.SequenceMatcher(None, inter, ra).ratio())
        scores.append(difflib.SequenceMatcher(None, inter, rb).ratio())
    return max(scores)


def token_overlap(a: object, b: object) -> float:
    """Jaccard overlap on word sets - the right measure for a long printed block.

    An ingredients list is a SET of items whose OCR line order genuinely varies
    between two reads of the same panel (line breaks land differently once the image
    is re-rendered). A sequence-based ratio punishes that reordering as if words were
    lost; Jaccard only asks which words are present, which is the question that
    actually matters for "did the ingredients survive".
    """
    ta = {t for t in re.findall(r"[a-z0-9]+", norm(a)) if len(t) > 2}
    tb = {t for t in re.findall(r"[a-z0-9]+", norm(b)) if len(t) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ---------------------------------------------------------------- the comparison


def _looks_like_long_digits(*values: str) -> bool:
    """A barcode/licence-shaped value: one long UNBROKEN digit run, in a SHORT value.

    Both halves of that are load-bearing, and both were learned from the same bug. The
    original test was `len(digits(v)) >= 8`, and `digits()` strips every non-digit and
    concatenates what's left - so an ingredients paragraph reading "Sugar 6%, Corn
    Syrup 1%, Maize Starch 2%, ... (E100, E124)" collapsed to the 19-digit string
    "6121422420100124420" and was compared as a BARCODE. Two ingredient lists then
    "differed" with the message "a code one digit off is a different product", which is
    nonsense for prose and buried the real finding.

    A genuine code is a contiguous run (`\\d{8,}`), and it is essentially the whole
    value - a 13-digit EAN or a 14-digit FSSAI number is ~15 characters, not 300. The
    length cap is what stops a paragraph that happens to quote a licence number from
    being treated as one, which the run test alone would still allow.
    """
    for v in values:
        s = str(v or "")
        if len(s) <= 40 and re.search(r"\d{8,}", s):
            return True
    return False


def _looks_like_prose(*values: str) -> bool:
    return any(len(str(v).split()) >= 8 for v in values)


def compare_values(original: str | None, enhanced: str | None) -> tuple[Status, float | None, str, str]:
    """Compare one field's two readings. Returns (status, score, method, note).

    Presence is settled before similarity: a field can only be CHANGED if both sides
    actually read something. LOST/ADDED are their own outcomes precisely because they
    have different causes and different fixes than a value that simply differs.
    """
    o_null, e_null = is_nullish(original), is_nullish(enhanced)

    if o_null and e_null:
        return Status.ABSENT, None, "presence", "neither image showed this field"
    if o_null:
        return (Status.ADDED, None, "presence",
                "not readable in the original - either the enhancement made existing "
                "print legible (fine) or the model invented it (not fine); needs a look")
    if e_null:
        return (Status.LOST, None, "presence",
                "readable in the original but gone from the enhanced image")

    o, e = str(original), str(enhanced)

    # --- long digit runs: barcode, FSSAI licence. Exact or nothing.
    if _looks_like_long_digits(o, e):
        do, de = digits(o), digits(e)
        if do == de:
            return Status.MATCH, 1.0, "digits:exact", "digit-for-digit identical"
        # How close, purely so a reviewer can tell a 1-digit OCR slip from a
        # completely different number - it does NOT soften the verdict.
        score = difflib.SequenceMatcher(None, do, de).ratio()
        return (Status.CHANGED, score, "digits:exact",
                f"digits differ ({do} vs {de}) - a code one digit off is a different "
                f"product, not a near miss")

    # --- number + unit: canonical measure, so "1 kg" == "1000 g".
    mo, me = parse_measure(o), parse_measure(e)
    if mo and me:
        if mo[0] != me[0]:
            return (Status.CHANGED, 0.0, "measure",
                    f"different units of measure ({mo[0]} vs {me[0]})")
        bigger = max(abs(mo[1]), abs(me[1]), 1e-9)
        rel = abs(mo[1] - me[1]) / bigger
        if rel <= MEASURE_TOLERANCE:
            return Status.MATCH, 1.0, "measure", f"same quantity ({mo[1]:g} {mo[0]})"
        return (Status.CHANGED, max(0.0, 1.0 - rel), "measure",
                f"quantity differs ({mo[1]:g} {mo[0]} vs {me[1]:g} {me[0]})")

    # --- plain number: "75.00" == "75".
    no, ne = as_number(o), as_number(e)
    if no is not None and ne is not None and not any(c.isalpha() for c in o + e):
        if abs(no - ne) < 1e-6:
            return Status.MATCH, 1.0, "number", f"same value ({no:g})"
        return (Status.CHANGED, 0.0, "number", f"value differs ({no:g} vs {ne:g})")

    # --- long prose: ingredients, nutrition panel.
    if _looks_like_prose(o, e):
        score = token_overlap(o, e)
        if score >= PROSE_THRESHOLD:
            return (Status.MATCH, score, "prose:token-overlap",
                    f"{score:.0%} of the words are shared")
        return (Status.CHANGED, score, "prose:token-overlap",
                f"only {score:.0%} of the words are shared - text was added, dropped "
                f"or rewritten")

    # --- everything else: short text.
    score = fuzzy_ratio(o, e)
    if score >= FUZZY_THRESHOLD:
        return Status.MATCH, score, "text:fuzzy", f"{score:.0%} similar"
    return (Status.CHANGED, score, "text:fuzzy",
            f"only {score:.0%} similar - the printed text does not agree")


def compare_field(
    field: str, original: str | None, enhanced: str | None,
    original_confidence: float = 1.0, enhanced_confidence: float = 1.0,
    original_source: str = "", enhanced_source: str = "",
) -> FieldComparison:
    status, score, method, note = compare_values(original, enhanced)
    cmp_ = FieldComparison(
        field=field,
        original=None if is_nullish(original) else str(original),
        enhanced=None if is_nullish(enhanced) else str(enhanced),
        status=status, score=score, method=method, note=note,
        original_confidence=original_confidence, enhanced_confidence=enhanced_confidence,
        original_source=original_source, enhanced_source=enhanced_source,
    )
    if cmp_.weakly_evidenced:
        cmp_.note += (f" — but both readings are weak guesses "
                      f"({original_source or 'unknown'} @ {original_confidence:.2f}, "
                      f"{enhanced_source or 'unknown'} @ {enhanced_confidence:.2f}), so "
                      f"this may be the extractor being unstable rather than the image "
                      f"being wrong")
    return cmp_
