"""The Layout / Field Parser - regex + layout heuristics, no model, no network.

This is Stage 2's workhorse and the box labelled "Regex (Parsing with prerequisite) /
Heuristic (Position/Size/Proximity finds Brands)" in the architecture. It takes the product's
OCR lines (text + boxes + confidence) and the decoded barcode, and fills the ProductExtraction
fields. Each field is won by the cheapest reliable signal:

  * KEYWORD + REGEX for the fields a pack labels explicitly - MRP ("MRP ₹75"), FSSAI ("Lic.
    No. <14 digits>"), Net Quantity ("Net Qty 330 ml"), Manufacturer ("Marketed by: ..."),
    Country of Origin ("Country of Origin: India"). The label is the prerequisite: we read
    the number/name that SITS WITH its label, not any lookalike elsewhere on the pack (the
    same discipline the barcode parser learned - an FSSAI number is not an MRP).
  * The BARCODE for Upc/Ean (already decoded + checksum-verified upstream) and, when Open
    Food Facts knew it, its canonical brand - the single most reliable Brand signal there is.
  * LAYOUT (position/size) for Brand / Product Name when nothing labels them: the brand is
    typically the largest, high-on-the-pack text that ISN'T back-panel fine print.

Everything is confidence-scored so the downstream routing step knows which fields to trust
and which to send to a VLM or a human.
"""

from __future__ import annotations

import re

from .schema import PositionedLine, ProductExtraction

# ---------------------------------------------------------------- patterns

# MRP: the label is required (a bare "75" anywhere is not an MRP). Tolerate the many ways it
# is written - "M.R.P", "MRP Rs.", "₹", "INR", "/-", "(incl. of all taxes)".
_MRP_RE = re.compile(
    r"(?:m\.?\s?r\.?\s?p\.?|maximum\s+retail\s+price)\b[^0-9₹]{0,15}"
    r"(?:rs\.?|inr|₹|₹)?\s*"
    r"(\d{1,5}(?:\.\d{1,2})?)",
    re.I,
)
# A price token with a currency mark but no MRP word - weaker, used only as a fallback.
_PRICE_RE = re.compile(r"(?:rs\.?|inr|₹|₹)\s*(\d{1,5}(?:\.\d{1,2})?)", re.I)

# Net quantity: a number + a unit. Prefer one that sits with a "net ..." label; otherwise the
# most prominent size on the pack (front face) is a good guess.
_UNIT = r"(?:kg|kgs|g|gm|gms|grams?|mg|ml|l|ltr|litres?|liters?|pcs?|pieces?|units?|nos?|tea\s?bags?|sachets?|N)"
_NETQTY_LABELLED_RE = re.compile(
    r"net\s*(?:qty|quantity|wt\.?|weight|content|vol\.?|volume)\b[^0-9]{0,10}"
    r"(\d+(?:\.\d+)?)\s*(" + _UNIT + r")\b",
    re.I,
)
_NETQTY_ANY_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(" + _UNIT + r")\b", re.I)

# FSSAI: the 14-digit licence number, only when its label is present (so we never grab a
# barcode or phone number of similar length).
_FSSAI_RE = re.compile(
    r"(?:fssai|lic\.?\s*no\.?|licence\s*no\.?|license\s*no\.?)\b[^0-9]{0,12}(\d{14})", re.I
)

# Manufacturer / marketer: grab the entity named after the "…by" label.
_MFR_RE = re.compile(
    r"(?:manufactured|mfd|mfg|marketed|mktd|packed|produced)\.?\s*"
    r"(?:&\s*(?:marketed|packed|distributed))?\s*by\s*[:\-]?\s*(.+)",
    re.I,
)

# Country of origin.
_COUNTRY_RE = re.compile(
    r"(?:country\s*of\s*origin|made\s+in|product\s+of|origin)\b[^a-z]{0,6}([a-z][a-z .]{2,30})",
    re.I,
)

# Back-panel vocabulary: a line containing any of these is fine print, NOT a brand/product
# name, so the layout heuristic skips it when hunting for the big front-of-pack text.
_BACK_PANEL_WORDS = (
    "ingredient", "nutrition", "manufactured", "marketed", "packed", "fssai", "lic",
    "customer", "care", "consumer", "storage", "store in", "best before", "batch", "mfg",
    "www.", ".com", "email", "phone", "toll", "regd", "office", "address", "net wt",
    "net qty", "allergen", "contains", "shelf life", "expiry", "recycle", "@",
)


# ---------------------------------------------------------------- field extractors


def _clean_entity(text: str) -> str:
    """Trim a captured company/country string to something reasonable: stop at the first hard
    delimiter and drop trailing punctuation. Keeps 'Pushpam Foods & Beverages Pvt. Ltd',
    drops the address that often trails it on the same OCR line."""
    text = re.split(r"[,;|]|\s{3,}|\d{5,}", text, maxsplit=1)[0]
    return text.strip(" .,:;-–").strip()


def _extract_mrp(lines: list[PositionedLine], out: ProductExtraction) -> None:
    for ln in lines:
        m = _MRP_RE.search(ln.text)
        if m:
            out.set("Mrp", m.group(1), 0.9 * (ln.confidence or 0.8), "regex:mrp-labelled")
            return
    # fallback: a currency-marked price, less certain (could be a "unit sale price" etc.)
    for ln in lines:
        m = _PRICE_RE.search(ln.text)
        if m:
            out.set("Mrp", m.group(1), 0.5 * (ln.confidence or 0.8), "regex:price-mark")
            return


def _extract_net_quantity(lines: list[PositionedLine], out: ProductExtraction) -> None:
    for ln in lines:  # strongly prefer a labelled "Net Qty ..."
        m = _NETQTY_LABELLED_RE.search(ln.text)
        if m:
            out.set("Net Quantity", f"{m.group(1)} {m.group(2).lower()}",
                    0.9 * (ln.confidence or 0.8), "regex:net-qty-labelled")
            return
    # fallback: the most prominent number+unit on the pack (usually the front-face size).
    best = None
    for ln in lines:
        m = _NETQTY_ANY_RE.search(ln.text)
        if m and (best is None or ln.height_frac > best[0]):
            best = (ln.height_frac, f"{m.group(1)} {m.group(2).lower()}", ln.confidence or 0.8)
    if best:
        out.set("Net Quantity", best[1], 0.6 * best[2], "layout:prominent-size")


def _extract_fssai(lines: list[PositionedLine], out: ProductExtraction) -> None:
    for ln in lines:
        m = _FSSAI_RE.search(re.sub(r"\s+", " ", ln.text))
        if m:
            out.set("Fssai No", m.group(1), 0.95 * (ln.confidence or 0.8), "regex:fssai")
            return


def _extract_manufacturer(lines: list[PositionedLine], out: ProductExtraction) -> None:
    for ln in lines:
        m = _MFR_RE.search(ln.text)
        if m:
            entity = _clean_entity(m.group(1))
            if len(entity) >= 3:
                out.set("Manufacturer", entity, 0.85 * (ln.confidence or 0.8), "regex:mktd-by")
                return


def _extract_country(lines: list[PositionedLine], out: ProductExtraction) -> None:
    for ln in lines:
        m = _COUNTRY_RE.search(ln.text)
        if m:
            country = _clean_entity(m.group(1)).title()
            if 3 <= len(country) <= 30:
                out.set("Country Of Origin", country, 0.85 * (ln.confidence or 0.8), "regex:origin")
                return


def _extract_brand_and_name(lines: list[PositionedLine], out: ProductExtraction, barcode) -> None:
    # Best brand signal: the barcode's canonical brand from Open Food Facts, when we have it.
    if barcode is not None and getattr(barcode, "canonical_brand", None):
        out.set("Brand", barcode.canonical_brand, 0.9, "barcode:off-lookup")
    if barcode is not None and getattr(barcode, "canonical_name", None):
        out.set("Product Name", barcode.canonical_name, 0.85, "barcode:off-lookup")

    # Layout fallback: the biggest, upper, non-fine-print line is the brand / product name.
    def is_front_text(ln: PositionedLine) -> bool:
        low = ln.text.lower()
        return (
            len(ln.text.strip()) >= 3
            and any(c.isalpha() for c in ln.text)
            and not any(w in low for w in _BACK_PANEL_WORDS)
        )

    front = sorted((ln for ln in lines if is_front_text(ln)), key=lambda l: -l.height_frac)
    if front:
        # The single largest line -> brand (if not already set from the barcode).
        out.set("Brand", front[0].text, 0.4 * (front[0].confidence or 0.8), "layout:largest-text")
        # The largest 1-2 lines joined -> a product-name guess.
        name = " ".join(l.text for l in front[:2])
        out.set("Product Name", name, 0.35 * (front[0].confidence or 0.8), "layout:top-lines")


# ---------------------------------------------------------------- public entry


def parse(lines: list[PositionedLine], barcode=None) -> ProductExtraction:
    """Run every field extractor over a product's OCR lines (+ its decoded barcode) and
    return the filled ProductExtraction. Order matters only for readability - each extractor
    writes its own field, and `ProductExtraction.set` keeps the higher-confidence answer."""
    out = ProductExtraction()

    # Upc / Ean comes straight from the verified barcode decode - the one field with its own
    # checksum, so by far the most trustworthy thing Stage 2 emits.
    if barcode is not None and getattr(barcode, "code", None) and barcode.valid_checksum:
        out.set("Upc / Ean", barcode.code, 1.0, "barcode:decoded")

    _extract_mrp(lines, out)
    _extract_net_quantity(lines, out)
    _extract_fssai(lines, out)
    _extract_manufacturer(lines, out)
    _extract_country(lines, out)
    _extract_brand_and_name(lines, out, barcode)
    return out
