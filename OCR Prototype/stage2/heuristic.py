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

# ---------------------------------------------------------------- long-text block headers
#
# These four fields (Ingredients/Nutritional_Details/Usage_Details/Storage_Instructions) are
# printed as a LABELLED BLOCK, not a single labelled value - "INGREDIENTS: Water, Sugar, ..."
# runs across several OCR lines. `_extract_block` below finds the header line and appends
# whatever follows until something that looks like a NEW section starts. This is inherently
# approximate (OCR line order across a product's several photographed faces is only
# piecewise reliable) - it is the softest-signal extractor in this file, unlike the
# labelled-regex fields above which only ever fire on an exact, unambiguous match.

_INGREDIENTS_HEADER_RE = re.compile(r"^ingredients?\s*[:\-]?\s*(.*)$", re.I)
_NUTRITION_HEADER_RE = re.compile(
    r"^nutrition(?:al)?\s*(?:information|facts|value)?\s*[:\-]?\s*(.*)$", re.I
)
_USAGE_HEADER_RE = re.compile(
    r"^(?:directions?\s*(?:for\s+)?use|how\s+to\s+use|preparation|usage\s*(?:instructions?)?)"
    r"\s*[:\-]?\s*(.*)$",
    re.I,
)
_STORAGE_HEADER_RE = re.compile(r"^(?:storage(?:\s+instructions?)?|store\s+in)\s*[:\-]?\s*(.*)$", re.I)

# Other new-field patterns, same discipline as the originals above: labelled where the pack
# labels it, keyword-triggered only where a label never appears (flagged as lower-confidence).
_NO_PRESERVATIVE_RE = re.compile(r"no\s+added\s+preservatives?|preservative[\s\-]?free", re.I)
_HAS_PRESERVATIVE_RE = re.compile(r"contains?\s+preservatives?|preservatives?\s*[:(]", re.I)
_RTC_RE = re.compile(r"\bready\s*to\s*cook\b", re.I)
_RTE_RE = re.compile(r"\bready\s*to\s*eat\b", re.I)
_HERBAL_RE = re.compile(r"\bherbal\b|\bayurvedic\b", re.I)
_FLAVOUR_LABEL_RE = re.compile(r"flavou?rs?\s*[:\-]\s*([a-z][a-z \-]{2,30})", re.I)
_FLAVOUR_NAME_RE = re.compile(r"([A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+){0,2})\s+flavou?r\b")
_CAFFEINE_RE = re.compile(
    r"caffeine\b[^0-9]{0,15}(\d+(?:\.\d+)?\s*mg(?:\s*/\s*\d+\s*ml)?(?:\s*(?:can|serve|bottle))?)",
    re.I,
)
_AGE_RE = re.compile(
    r"(?:recommended\s+age)\s*[:\-]?\s*(\d{1,2}\s*\+|\d{1,2}\s*(?:years?|yrs?)\s*(?:and\s*)?"
    r"(?:above|older)?)",
    re.I,
)
_BABY_WEIGHT_RE = re.compile(r"(\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?\s*kg)", re.I)
_ABSORPTION_RE = re.compile(r"absorption[^0-9]{0,15}(\d+(?:\.\d+)?\s*(?:hrs?|hours?))", re.I)
_DIMENSION_RE = re.compile(
    r"(\d+(?:\.\d+)?\s*[xX×]\s*\d+(?:\.\d+)?(?:\s*[xX×]\s*\d+(?:\.\d+)?)?\s*(?:cm|mm|inch(?:es)?|in)\b)",
    re.I,
)
_PACK_QTY_RE = re.compile(r"pack\s*of\s*(\d+)", re.I)
_NON_VEG_RE = re.compile(r"\bnon[\s\-]?veg(?:etarian)?\b", re.I)
_VEG_RE = re.compile(r"\bvegetarian\b|\bveg\b", re.I)

# Best-effort keyword lookups for fields with no reliable label pattern at all - lowest
# confidence in this file by design; a miss here just leaves the field null, which the
# schema's `set()` treats as an honest abstention, not a wrong guess.
_BULLET_KEYWORDS = (
    "no added", "no artificial", "zero", "100%", "high in", "rich in", "source of",
    "low fat", "sugar free", "gluten free", "vegan", "fortified", "clinically proven",
    "recommended by", "no.1", "no. 1", "#",
)
_THEME_KEYWORDS = (
    "diwali", "christmas", "rakhi", "raksha bandhan", "holi", "valentine", "birthday",
    "wedding", "festive", "new year", "summer edition", "monsoon edition",
)
_HAIR_TYPE_KEYWORDS = (
    "dry hair", "oily hair", "normal hair", "damaged hair", "frizzy hair", "curly hair",
    "all hair types",
)
_MINERAL_KEYWORDS = ("calcium", "zinc", "iron", "magnesium", "himalayan", "phosphate")


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


# ---------------------------------------------------------------- long-text blocks


def _looks_like_new_section(text: str) -> bool:
    """Does this line look like the START of a different labelled section, so a block
    extractor below should stop appending here rather than swallow it? Short ALL-CAPS label
    lines and anything the labelled extractors above would themselves claim both count;
    ordinary sentence continuations don't."""
    stripped = text.strip()
    if not stripped:
        return True
    if (_MRP_RE.search(stripped) or _FSSAI_RE.search(stripped) or _MFR_RE.search(stripped)
            or _COUNTRY_RE.search(stripped) or _NETQTY_LABELLED_RE.search(stripped)):
        return True
    words = stripped.split()
    if len(words) <= 6 and stripped.upper() == stripped and any(c.isalpha() for c in stripped):
        return True
    return False


def _extract_block(lines: list[PositionedLine], header_re: re.Pattern, max_lines: int = 8) -> str | None:
    """Find the first line matching `header_re` and append subsequent lines until one looks
    like a new section, `max_lines` is reached, or the lines run out. See the module-level
    comment above `_INGREDIENTS_HEADER_RE` for why this is approximate by construction."""
    for i, ln in enumerate(lines):
        m = header_re.match(ln.text.strip())
        if not m:
            continue
        collected = []
        remainder = m.group(1) if m.lastindex else ""
        if remainder and remainder.strip():
            collected.append(remainder.strip())
        for nxt in lines[i + 1: i + 1 + max_lines]:
            if _looks_like_new_section(nxt.text):
                break
            collected.append(nxt.text.strip())
        if collected:
            return re.sub(r"\s+", " ", " ".join(collected)).strip(" .,:;-")
    return None


def _extract_long_text_fields(lines: list[PositionedLine], out: ProductExtraction) -> None:
    for field, header_re in (
        ("Ingredients", _INGREDIENTS_HEADER_RE),
        ("Nutritional_Details", _NUTRITION_HEADER_RE),
        ("Usage_Details", _USAGE_HEADER_RE),
        ("Storage_Instructions", _STORAGE_HEADER_RE),
    ):
        block = _extract_block(lines, header_re)
        if block:
            out.set(field, block, 0.5, f"heuristic:block:{field.lower()}")


def _extract_description(lines: list[PositionedLine], out: ProductExtraction) -> None:
    """Weakest-signal field in this file: no label ever marks "this is the marketing
    blurb", so this just picks the longest sentence-shaped front-of-pack line that nothing
    else here has claimed. Left in place because ground truth shows it is usually present
    and short-ish, but treat this one skeptically."""
    claimed = {f.value for f in out.fields.values()}
    candidates = [
        ln for ln in lines
        if len(ln.text.split()) >= 8
        and any(c.islower() for c in ln.text)
        and ln.text.strip() not in claimed
        and not any(w in ln.text.lower() for w in _BACK_PANEL_WORDS)
    ]
    if candidates:
        best = max(candidates, key=lambda l: len(l.text))
        out.set("Product_Description", best.text.strip(), 0.4 * (best.confidence or 0.8),
                "heuristic:longest-front-sentence")


def _extract_bullets(lines: list[PositionedLine], out: ProductExtraction, max_bullets: int = 6) -> None:
    for ln in lines:
        text = ln.text.strip()
        words = text.split()
        if not (2 <= len(words) <= 10):
            continue
        if any(kw in text.lower() for kw in _BULLET_KEYWORDS):
            out.add_bullet(text)
        if len(out.bullets) >= max_bullets:
            break


# ---------------------------------------------------------------- labelled / keyword fields


def _extract_preservatives(lines: list[PositionedLine], out: ProductExtraction) -> None:
    for ln in lines:
        if _NO_PRESERVATIVE_RE.search(ln.text):
            out.set("Preservatives", "No", 0.8 * (ln.confidence or 0.8), "regex:no-preservative-claim")
            return
    for ln in lines:
        if _HAS_PRESERVATIVE_RE.search(ln.text):
            out.set("Preservatives", "Yes", 0.6 * (ln.confidence or 0.8), "regex:preservative-claim")
            return


def _extract_ready_flags(lines: list[PositionedLine], out: ProductExtraction) -> None:
    for ln in lines:
        if _RTC_RE.search(ln.text):
            out.set("Ready_to_cook", "Yes", 0.75 * (ln.confidence or 0.8), "regex:ready-to-cook")
            break
    for ln in lines:
        if _RTE_RE.search(ln.text):
            out.set("Ready_to_eat", "Yes", 0.75 * (ln.confidence or 0.8), "regex:ready-to-eat")
            break


def _extract_herbal(lines: list[PositionedLine], out: ProductExtraction) -> None:
    for ln in lines:
        if _HERBAL_RE.search(ln.text):
            out.set("Herbal_or_Ayurvedic", "Yes", 0.7 * (ln.confidence or 0.8), "regex:herbal-claim")
            return


def _extract_flavour(lines: list[PositionedLine], out: ProductExtraction) -> None:
    for ln in lines:
        m = _FLAVOUR_LABEL_RE.search(ln.text)
        if m:
            out.set("Flavours_or_Spices", m.group(1).strip().title(), 0.7 * (ln.confidence or 0.8),
                    "regex:flavour-label")
            return
    for ln in lines:
        m = _FLAVOUR_NAME_RE.search(ln.text)
        if m:
            out.set("Flavours_or_Spices", m.group(1).strip(), 0.5 * (ln.confidence or 0.8),
                    "regex:flavour-name")
            return


def _extract_caffeine(lines: list[PositionedLine], out: ProductExtraction) -> None:
    for ln in lines:
        m = _CAFFEINE_RE.search(ln.text)
        if m:
            out.set("Caffeine_Content", m.group(1).strip(), 0.75 * (ln.confidence or 0.8),
                    "regex:caffeine-content")
            return


def _extract_age_and_baby(lines: list[PositionedLine], out: ProductExtraction) -> None:
    for ln in lines:
        m = _AGE_RE.search(ln.text)
        if m:
            out.set("Recommended_Age", m.group(1).strip(), 0.6 * (ln.confidence or 0.8),
                    "regex:recommended-age")
            break
    for ln in lines:
        if "weight" in ln.text.lower() or "kg" in ln.text.lower():
            m = _BABY_WEIGHT_RE.search(ln.text)
            if m:
                out.set("Baby_Weight", m.group(1).strip(), 0.55 * (ln.confidence or 0.8),
                        "regex:baby-weight-range")
                break


def _extract_absorption(lines: list[PositionedLine], out: ProductExtraction) -> None:
    for ln in lines:
        m = _ABSORPTION_RE.search(ln.text)
        if m:
            out.set("Absorption_Duration", m.group(1).strip(), 0.6 * (ln.confidence or 0.8),
                    "regex:absorption-duration")
            return


def _extract_dimension(lines: list[PositionedLine], out: ProductExtraction) -> None:
    for ln in lines:
        m = _DIMENSION_RE.search(ln.text)
        if m:
            out.set("Dimension", m.group(1).strip(), 0.6 * (ln.confidence or 0.8), "regex:dimension")
            return


def _extract_pack_quantity(lines: list[PositionedLine], out: ProductExtraction) -> None:
    for ln in lines:
        m = _PACK_QTY_RE.search(ln.text)
        if m:
            out.set("Pack_Quantity", m.group(1), 0.7 * (ln.confidence or 0.8), "regex:pack-of-n")
            return


def _extract_diet_type(lines: list[PositionedLine], out: ProductExtraction) -> None:
    for ln in lines:  # non-veg checked first: "vegetarian" is a substring pitfall of "non-veg"
        if _NON_VEG_RE.search(ln.text):
            out.set("Diet_Type", "Non-Vegetarian", 0.7 * (ln.confidence or 0.8), "regex:non-veg-mark")
            return
    for ln in lines:
        if _VEG_RE.search(ln.text):
            out.set("Diet_Type", "Vegetarian", 0.6 * (ln.confidence or 0.8), "regex:veg-mark")
            return


def _extract_keyword_field(
    lines: list[PositionedLine], out: ProductExtraction, field_name: str, keywords: tuple[str, ...],
) -> None:
    """Lowest-confidence extractors in this file: no label pattern exists for these fields at
    all, just a closed keyword list. A miss just leaves the field null (an honest abstention),
    never a fabricated guess."""
    for ln in lines:
        low = ln.text.lower()
        for kw in keywords:
            if kw in low:
                out.set(field_name, kw.title(), 0.4 * (ln.confidence or 0.8), f"keyword:{field_name.lower()}")
                return


# ---------------------------------------------------------------- public entry


def parse(
    text_lines: list[PositionedLine], layout_lines: list[PositionedLine] | None = None,
    barcode=None,
) -> ProductExtraction:
    """Run every field extractor over a product's OCR lines (+ its decoded barcode) and
    return the filled ProductExtraction. Order matters only for readability - each extractor
    writes its own field, and `ProductExtraction.set` keeps the higher-confidence answer.

    Two line pools, because they're not interchangeable (see pipeline.collect_lines_and_barcode
    and ocr.router.read_dual for why):
      * `text_lines` - whichever engine transcribes best (the local VLM when it's running,
        else RapidOCR). Every regex/keyword/block extractor reads from here - they only need
        accurate TEXT, not real geometry.
      * `layout_lines` - ALWAYS RapidOCR's read, the only source with real per-line boxes.
        Used ONLY by `_extract_brand_and_name`'s position/size fallback, which is meaningless
        without genuine geometry. Defaults to `text_lines` if not given (e.g. a caller that
        only has one engine's output and accepts the layout heuristic running on it too).
    """
    layout_lines = layout_lines if layout_lines is not None else text_lines
    out = ProductExtraction()

    # Upc / Ean comes straight from the verified barcode decode - the one field with its own
    # checksum, so by far the most trustworthy thing Stage 2 emits.
    if barcode is not None and getattr(barcode, "code", None) and barcode.valid_checksum:
        out.set("Upc / Ean", barcode.code, 1.0, "barcode:decoded")

    _extract_mrp(text_lines, out)
    _extract_net_quantity(text_lines, out)
    _extract_fssai(text_lines, out)
    _extract_manufacturer(text_lines, out)
    _extract_country(text_lines, out)
    _extract_brand_and_name(layout_lines, out, barcode)

    # Ground-truth-format fields (see schema.to_ground_truth_dict) - long-text blocks first
    # so _extract_description can see what they've already claimed and not duplicate it.
    _extract_long_text_fields(text_lines, out)
    _extract_bullets(text_lines, out)
    _extract_preservatives(text_lines, out)
    _extract_ready_flags(text_lines, out)
    _extract_herbal(text_lines, out)
    _extract_flavour(text_lines, out)
    _extract_caffeine(text_lines, out)
    _extract_age_and_baby(text_lines, out)
    _extract_absorption(text_lines, out)
    _extract_dimension(text_lines, out)
    _extract_pack_quantity(text_lines, out)
    _extract_diet_type(text_lines, out)
    _extract_keyword_field(text_lines, out, "Theme_or_Occasion_Type", _THEME_KEYWORDS)
    _extract_keyword_field(text_lines, out, "Hair_Type", _HAIR_TYPE_KEYWORDS)
    _extract_keyword_field(text_lines, out, "Mineral_Source", _MINERAL_KEYWORDS)
    _extract_description(text_lines, out)
    return out
