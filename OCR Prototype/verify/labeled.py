"""The few fields a pack LABELS explicitly - and nothing else.

This replaces `stage2/heuristic.py` for the verifier, and the difference is what it
refuses to do. That module has ~25 extractors; roughly five of them are label-gated
regexes and the rest are guesses - "the biggest line is the Brand", "the longest
sentence is the Description", "any line containing 'calcium' means Mineral_Source". The
guesses carry its own lowest confidences (0.35-0.4) by its own admission, and they were
the source of most of the verifier's false findings, because two OCR reads of the same
pack routinely produce different guesses.

So this module keeps ONLY the extractors where the pack itself tells us what the value
is - the label is the prerequisite, exactly the discipline `heuristic.py`'s docstring
describes for its regex tier and then abandons for its layout tier:

    FSSAI licence   "Lic. No. <14 digits>"      the label is mandatory
    MRP             "MRP Rs. 75"                the label is mandatory
    Net quantity    "Net Wt. 500 g"             the label is mandatory
    Pack quantity   "Pack of 6"                 the phrase is the label

A field with no label present is left absent. That is not a gap - unlabelled values are
still compared, as unattributed members of the sets in `verify.signals` (every
number+unit on the pack, every price, every digit run). We simply decline to put a NAME
on something the pack did not name, because naming it wrongly is what broke the old
design.

Nothing here reads Brand, Product Name, Manufacturer or any of the keyword-inferred
fields. Brand-ish text is handled by containment (`signals.prominent_lines` +
`signals.contains_fuzzy`), which asks whether the text survived rather than what it was
called.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# FSSAI: exactly 14 digits, and only when its label is present, so a barcode or a phone
# number of similar length can never be mistaken for one. Digit-grouping tolerated -
# OCR frequently splits the number ("1151 5006 000531").
_FSSAI_RE = re.compile(
    r"(?:fssai|lic\.?\s*no\.?|licence\s*no\.?|license\s*no\.?)\b[^0-9]{0,14}"
    r"((?:\d[\s-]*){14})",
    re.I,
)

# MRP: the label is required. A bare "75" anywhere on a pack is not a price.
_MRP_RE = re.compile(
    r"(?:m\.?\s?r\.?\s?p\.?|maximum\s+retail\s+price)\b[^0-9₹€£]{0,18}"
    r"(?:rs\.?|inr|₹|€|£)?\s*(\d{1,6}(?:[.,]\d{1,2})?)",
    re.I,
)

_UNIT = (r"kg|kgs|g|gm|gms|gr|grams?|mg|ml|l|lt|ltr|litres?|liters?|cl|"
         r"pcs?|pieces?|units?|nos?")
_NETQTY_RE = re.compile(
    r"net\s*(?:qty|quantity|wt\.?|weight|content[s]?|vol\.?|volume)\b[^0-9]{0,12}"
    r"(\d+(?:[.,]\d+)?)\s*(" + _UNIT + r")\b",
    re.I,
)

_PACKQTY_RE = re.compile(r"pack\s*of\s*(\d{1,3})\b", re.I)


@dataclass
class LabeledValue:
    value: str
    confidence: float   # the OCR line's own confidence - never a hand-tuned constant
    evidence: str       # the raw OCR line the value came from, for the audit trail


# Ordered as displayed. Each is (key, display label, regex, formatter).
def _fmt_digits(m: re.Match) -> str:
    return re.sub(r"\D", "", m.group(1))


def _fmt_amount(m: re.Match) -> str:
    return m.group(1).replace(",", ".")


def _fmt_measure(m: re.Match) -> str:
    return f"{m.group(1).replace(',', '.')} {m.group(2).lower()}"


_EXTRACTORS = [
    ("fssai", "FSSAI licence no.", _FSSAI_RE, _fmt_digits),
    ("mrp", "MRP", _MRP_RE, _fmt_amount),
    ("net_quantity", "Net quantity (labelled)", _NETQTY_RE, _fmt_measure),
    ("pack_quantity", "Pack quantity", _PACKQTY_RE, lambda m: m.group(1)),
]

LABELS = {key: label for key, label, _, _ in _EXTRACTORS}
KEYS = [key for key, _, _, _ in _EXTRACTORS]


def extract(lines) -> dict[str, LabeledValue]:
    """Run every label-gated extractor over a product's OCR lines.

    Takes OCR line objects (needing `.text` and `.confidence`) rather than a flat blob,
    so each value carries the confidence of the line it came from. A value read off a
    line RapidOCR itself scored 0.4 should not be weighed like one from a 0.99 line,
    and only the per-line view preserves that.

    First match wins per field. These patterns require their label, so a second match
    is nearly always the same value re-detected on a wrapped line, not a competing
    candidate worth arbitrating.
    """
    out: dict[str, LabeledValue] = {}
    for ln in lines:
        text = re.sub(r"\s+", " ", (getattr(ln, "text", "") or "")).strip()
        if not text:
            continue
        for key, _label, pattern, fmt in _EXTRACTORS:
            if key in out:
                continue
            m = pattern.search(text)
            if m:
                value = fmt(m)
                if value:
                    out[key] = LabeledValue(
                        value=value,
                        confidence=float(getattr(ln, "confidence", None) or 0.8),
                        evidence=text[:120],
                    )
    return out


def fssai_plausible(digits: str) -> bool:
    """An FSSAI licence number is 14 digits and starts with 1 or 2 (state/central
    licence prefixes). A cheap structural sanity check - it does not prove the licence
    exists, only that the digits are shaped like one, which is enough to tell a real
    read from a mis-scan of a nearby number."""
    return len(digits) == 14 and digits[0] in "12"
