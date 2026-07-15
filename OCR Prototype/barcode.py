"""Barcode decode -> validate -> lookup -> cross-check.

Two genuinely different signals get combined here:

  1. DECODING the bars (cv2.barcode) - classical CV, deterministic, no model. Either gets
     the exact right 13 digits or fails outright; no "close enough". Fragile to angle/glare/
     resolution, so it alone only succeeds on ~8% of real photos (measured).

  2. The PRINTED digits under the barcode, read as ordinary text by whichever OCR engine
     already ran. On a clean, straight-on shot these usually agree with the decoded bars -
     so OCR digits are used as a fallback source of barcode candidates, not a replacement.

Before trusting either source, the EAN-13/UPC-A CHECKSUM is verified - a free, instant,
deterministic check (no network) that catches OCR typos and catalog-data noise. 129 of this
dataset's "UPC/EAN" ground-truth values are 6-11 digits, which cannot be a valid barcode at
all; checksum validation rejects those rather than something downstream trusting garbage.

Only a checksum-VALID code is ever looked up. An invalid code is reported as invalid and
the pipeline stops there - we do not query an API with a number we already know is wrong.

Lookup is Open Food Facts (free, no key). Measured coverage: 15/30 (50%) on this dataset's
Indian grocery sample - strong for food, near-zero for non-food (detergents, stationery),
because it is specifically a FOOD database.
"""

from __future__ import annotations

import difflib
import json
import re
import sqlite3
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import certifi
import cv2
import numpy as np

from json_metrics import str_sim

# How much of the barcode's own digit string must show up, verbatim, in the OCR text
# before we call the decode "confirmed" by the print underneath the bars. Not a fuzzy
# text similarity (str_sim) - a barcode is either printed correctly under the bars or
# it isn't, so this is a longest-common-substring fraction, not a token/character blend.
TEXT_MATCH_THRESHOLD = 0.4

# python.org's macOS build ships with no CA bundle wired up ("CERTIFICATE_VERIFY_FAILED").
# Point the default TLS context at certifi's bundle; verification still happens, it just
# now knows where the roots are. Same fix as benchmark_models.py's EasyOCR downloader.
ssl._create_default_https_context = lambda *a, **k: ssl.create_default_context(
    cafile=certifi.where()
)

DB = Path(__file__).parent / "barcode_cache.sqlite"
USER_AGENT = "OCR-Prototype/1.0 (research prototype; not for production traffic)"


@dataclass
class BarcodeResult:
    code: str | None = None
    valid_checksum: bool = False
    source: str = ""  # "bars" | "ocr_digits" | "" (nothing found)
    looked_up: bool = False
    lookup_source: str = ""  # "Open Food Facts" | "UPCitemdb" | "" (not found in either)
    canonical_name: str | None = None
    canonical_brand: str | None = None
    name_match: float | None = None   # fuzzy similarity: OCR brand/name vs canonical
    brand_match: float | None = None
    points: np.ndarray | None = None  # quadrangle cv2 located the bars at, for a UI crop
    text_match: float | None = None   # fraction of `code`'s digits found in the OCR text
    text_match_ok: bool | None = None  # text_match >= TEXT_MATCH_THRESHOLD
    trail: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- checksum


def checksum_valid(code: str) -> bool:
    """EAN-13 / UPC-A check digit. UPC-A (12 digits) is EAN-13 with a leading 0 - the same
    algorithm covers both. This is the free, instant gate before anything touches the network.
    """
    if len(code) == 12:
        code = "0" + code
    if len(code) != 13 or not code.isdigit():
        return False
    digits = [int(c) for c in code]
    total = sum(d if i % 2 == 0 else d * 3 for i, d in enumerate(digits[:12]))
    return (10 - total % 10) % 10 == digits[12]


# ---------------------------------------------------------------- decode


def decode_bars(rgb: np.ndarray) -> tuple[str | None, np.ndarray | None]:
    """Read the actual black/white bars. cv2's built-in decoder - no model, no network.

    cv2's signature is easy to misread: detectAndDecode(img) -> retval, points, straight_code.
    `retval` IS the decoded string (empty if nothing found) - it is not a success boolean.
    `points` is the quadrangle corners cv2 located the bars at - returned here too so a
    caller (e.g. the Streamlit UI) can crop and show just the barcode, not the whole product.
    """
    try:
        det = cv2.barcode.BarcodeDetector()
        retval, points, _straight = det.detectAndDecode(rgb)
        if retval and retval.isdigit():
            return retval, points
    except Exception:
        pass
    return None, None


def crop_region(rgb: np.ndarray, points: np.ndarray | None, pad: int = 15) -> np.ndarray | None:
    """Bounding-box crop of the quadrangle `decode_bars` located, padded a little so the
    quiet zone around the bars stays visible. None if there's nothing to crop (no bars
    detector hit - e.g. the code came from OCR digit text instead)."""
    if points is None:
        return None
    pts = np.asarray(points).reshape(-1, 2)
    if pts.size == 0:
        return None
    h, w = rgb.shape[:2]
    x0 = max(int(pts[:, 0].min()) - pad, 0)
    y0 = max(int(pts[:, 1].min()) - pad, 0)
    x1 = min(int(pts[:, 0].max()) + pad, w)
    y1 = min(int(pts[:, 1].max()) + pad, h)
    if x1 <= x0 or y1 <= y0:
        return None
    return rgb[y0:y1, x0:x1]


def text_match(code: str, ocr_text: str) -> float:
    """How much of the barcode's digit string shows up, verbatim, in the OCR text.
    Digit-only comparison (OCR text stripped to non-digits) via longest common substring,
    scored as a fraction of the barcode's own length. This is the "does the printed number
    under the bars agree with what OCR read" sanity check, independent of the Open Food
    Facts lookup (which only covers ~50% of codes)."""
    digits = re.sub(r"[^\d]", "", ocr_text)
    if not code or not digits:
        return 0.0
    match = difflib.SequenceMatcher(None, code, digits, autojunk=False) \
        .find_longest_match(0, len(code), 0, len(digits))
    return match.size / len(code)


def candidates_from_text(text: str) -> list[str]:
    """Digit runs long enough to be a barcode (11-14, to tolerate an OCR miss/extra digit
    at the boundary), longest first so the likeliest real barcode is tried before a shorter
    coincidental run."""
    runs = re.findall(r"\d{11,14}", re.sub(r"[^\d]", " ", text))
    return sorted(set(runs), key=len, reverse=True)


def find_barcode(
    rgb: np.ndarray, ocr_text: str = ""
) -> tuple[str | None, str, np.ndarray | None, list[str]]:
    """Try decoding the bars first; fall back to a checksum-valid digit run from OCR text.
    Returns (code_or_None, source, points_or_None, trail)."""
    trail = []

    bars, points = decode_bars(rgb)
    if bars:
        trail.append(f"decoded from bars: {bars}")
        return bars, "bars", points, trail
    trail.append("bar decode: no barcode detected in image")

    for cand in candidates_from_text(ocr_text):
        if checksum_valid(cand):
            trail.append(f"OCR digits, checksum-valid: {cand}")
            return cand, "ocr_digits", None, trail
    if ocr_text:
        trail.append("OCR digits: no checksum-valid run found")

    return None, "", None, trail


# ---------------------------------------------------------------- lookup


def _cache() -> sqlite3.Connection:
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS lookups (code TEXT PRIMARY KEY, payload TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS upcitemdb_lookups (code TEXT PRIMARY KEY, payload TEXT)")
    return c


def _lookup_off(code: str) -> dict | None:
    """Open Food Facts, free, no key. Cached permanently - a barcode's canonical data does
    not change between runs, so never fetch it twice."""
    with _cache() as c:
        row = c.execute("SELECT payload FROM lookups WHERE code = ?", (code,)).fetchone()
    if row:
        cached = json.loads(row[0])
        return cached if cached else None

    req = urllib.request.Request(
        f"https://world.openfoodfacts.org/api/v2/product/{code}.json",
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.load(r)
    except (urllib.error.URLError, TimeoutError, OSError):
        return None  # network hiccup: do not cache, so a later run can retry

    result = None
    if data.get("status") == 1:
        p = data.get("product", {})
        result = {
            "name": p.get("product_name") or None,
            "brand": (p.get("brands") or "").split(",")[0].strip() or None,
            "categories": p.get("categories"),
        }

    with _cache() as c:
        c.execute("INSERT OR REPLACE INTO lookups (code, payload) VALUES (?, ?)",
                  (code, json.dumps(result or {})))
    return result


def _lookup_upcitemdb(code: str) -> dict | None:
    """UPCitemdb's free 'trial' tier: no key, no signup, public and documented (unlike GS1
    India's own GTIN-validation tool, which pairs its lookup call with an invisible reCAPTCHA
    on the page - a deliberate anti-automation signal that a script has no business working
    around, even though the request itself is easy to replicate). Checked against 4 real
    barcodes from this dataset and got 0 hits - it is a US/western-retail-weighted database
    (a US UPC looked up alongside these came back with Walmart/Target/CVS listings), so this
    is a low-probability bonus source, not something to expect India coverage from.
    """
    with _cache() as c:
        row = c.execute(
            "SELECT payload FROM upcitemdb_lookups WHERE code = ?", (code,)
        ).fetchone()
    if row:
        cached = json.loads(row[0])
        return cached if cached else None

    req = urllib.request.Request(
        f"https://api.upcitemdb.com/prod/trial/lookup?upc={code}",
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.load(r)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None

    result = None
    items = data.get("items") or []
    if items:
        it = items[0]
        result = {
            "name": it.get("title") or None,
            "brand": it.get("brand") or None,
            "categories": it.get("category"),
        }

    with _cache() as c:
        c.execute("INSERT OR REPLACE INTO upcitemdb_lookups (code, payload) VALUES (?, ?)",
                  (code, json.dumps(result or {})))
    return result


def lookup(code: str) -> tuple[dict | None, str]:
    """Try Open Food Facts first (the better-fit source when it hits - food-specific,
    reasonable India coverage on this dataset), then UPCitemdb as a free bonus fallback.
    Returns (data_or_None, source_name) so callers/trail can say where a hit came from."""
    data = _lookup_off(code)
    if data is not None:
        return data, "Open Food Facts"
    data = _lookup_upcitemdb(code)
    if data is not None:
        return data, "UPCitemdb"
    return None, ""


# ---------------------------------------------------------------- pipeline


def read_and_validate(
    rgb: np.ndarray, ocr_text: str, ocr_brand: str | None = None, ocr_name: str | None = None
) -> BarcodeResult:
    """Full pipeline: decode -> checksum -> lookup -> cross-check against what OCR read.

    Cross-checking is the "catch spelling mistakes" half of this: if the barcode's
    canonical brand disagrees with what OCR read, that is a strong, independent signal
    something is wrong - far stronger than trusting OCR alone.
    """
    code, source, points, trail = find_barcode(rgb, ocr_text)

    if code is None:
        return BarcodeResult(trail=trail)

    valid = checksum_valid(code)
    trail.append(f"checksum: {'VALID' if valid else 'INVALID - stopping, no lookup attempted'}")
    if not valid:
        return BarcodeResult(code=code, valid_checksum=False, source=source, points=points, trail=trail)

    # Does the number printed under the bars agree with what OCR read? Independent of the
    # Open Food Facts lookup below (which only covers ~50% of codes), and the only cross-check
    # available at all for the ~50% that aren't in that database.
    match = text_match(code, ocr_text)
    match_ok = match >= TEXT_MATCH_THRESHOLD
    trail.append(f"text cross-check: {match:.0%} of barcode digits found in OCR text "
                 f"(threshold {TEXT_MATCH_THRESHOLD:.0%}) -> {'OK' if match_ok else 'weak match'}")

    data, lookup_source = lookup(code)
    if data is None:
        trail.append("lookup: not found in Open Food Facts or UPCitemdb")
        return BarcodeResult(code=code, valid_checksum=True, source=source, points=points,
                              text_match=match, text_match_ok=match_ok, trail=trail)

    trail.append(f"lookup: FOUND in {lookup_source} - {data.get('brand')} / {data.get('name')}")
    result = BarcodeResult(
        code=code, valid_checksum=True, source=source, looked_up=True, points=points,
        lookup_source=lookup_source,
        canonical_name=data.get("name"), canonical_brand=data.get("brand"),
        text_match=match, text_match_ok=match_ok, trail=trail,
    )

    if ocr_brand and result.canonical_brand:
        result.brand_match = str_sim(ocr_brand, result.canonical_brand)
        trail.append(f"brand cross-check: OCR={ocr_brand!r} vs canonical={result.canonical_brand!r} "
                      f"-> similarity {result.brand_match:.2f}")
    if ocr_name and result.canonical_name:
        result.name_match = str_sim(ocr_name, result.canonical_name)
        trail.append(f"name cross-check: OCR={ocr_name!r} vs canonical={result.canonical_name!r} "
                      f"-> similarity {result.name_match:.2f}")

    return result
