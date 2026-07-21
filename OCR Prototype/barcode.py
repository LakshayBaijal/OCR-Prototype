"""Barcode decode -> validate -> lookup -> cross-check.

Two genuinely different signals get combined here:

  1. DECODING the bars - classical CV, deterministic, no model. Either gets the exact right
     13 digits or fails outright; no "close enough". `decode_bars_robust` pushes this HARD
     (latency traded for reliability, since the barcode is the one field that must be exactly
     right and un-reviewed): two decoders (zbar + cv2), then whole-image upscale + Otsu, then
     a gradient localiser that finds the bars and decodes each region upscaled and at every
     90 degrees. Measured to nearly double the bar-decode rate on this dataset's low-res
     catalog images (14 -> 23 of 100) at ZERO cost to precision. Every decode is checksum-
     gated before it is trusted (a plain cv2 decode is NOT internally checksum-checked -
     measured it returning a flat-out wrong 8-digit "92113800" on a pack whose real code was
     "8901414001626"; that is now rejected). The one thing this can't beat is resolution: a
     ~60px barcode in a 420px thumbnail is below ~1px per module and its bars are simply gone.

  2. The PRINTED digits under the barcode, read as ordinary text by whichever OCR engine
     already ran. On a clean, straight-on shot these usually agree with the decoded bars -
     so OCR digits are used as a fallback source of barcode candidates, not a replacement.

Before trusting any of the above, the EAN-13/UPC-A CHECKSUM is verified - a free, instant,
deterministic check (no network) that catches OCR typos, a wrong bar decode, and catalog-data
noise. 129 of this dataset's "UPC/EAN" ground-truth values are 6-11 digits, which cannot be a
valid barcode at all; checksum validation rejects those rather than something downstream
trusting garbage.

Only a checksum-VALID code is ever looked up. An invalid code is reported as invalid and
the pipeline stops there - we do not query an API with a number we already know is wrong.

Lookup is Open Food Facts (free, no key). Measured coverage: 15/30 (50%) on this dataset's
Indian grocery sample - strong for food, near-zero for non-food (detergents, stationery),
because it is specifically a FOOD database.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
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

# pyzbar wraps the native `libzbar` C library via ctypes and resolves it the moment the
# module is imported - not lazily. On macOS, Python's library search does not look in
# Homebrew's install prefix (/opt/homebrew on Apple Silicon, /usr/local on Intel), so a
# plain `import pyzbar` fails with "Unable to find zbar shared library" even right after
# `brew install zbar` succeeds and the .dylib is sitting on disk. Adding Homebrew's lib dir
# to DYLD_LIBRARY_PATH BEFORE the import (verified: setting it from inside the running
# process, not just as a shell env var, is sufficient - Python's own loader reads it at
# import time) fixes that. Harmless if zbar isn't installed at all: the import below still
# fails, caught by the except, and callers fall back to cv2 only.
for _prefix in ("/opt/homebrew/lib", "/usr/local/lib"):
    if os.path.isdir(_prefix) and _prefix not in os.environ.get("DYLD_LIBRARY_PATH", ""):
        os.environ["DYLD_LIBRARY_PATH"] = _prefix + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")

try:
    from pyzbar import pyzbar as _pyzbar
    from pyzbar.pyzbar import ZBarSymbol as _ZBarSymbol

    # Only the 1-D retail-product symbologies. Restricting to these keeps zbar focused on
    # what a grocery pack carries AND silences the noisy "databar" assertion warnings zbar
    # prints to stderr when it speculatively tries to decode GS1 DataBar on binarized input.
    _PRODUCT_SYMBOLS = [_ZBarSymbol.EAN13, _ZBarSymbol.UPCA, _ZBarSymbol.EAN8]
except (ImportError, OSError):
    _pyzbar = None
    _PRODUCT_SYMBOLS = None

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
    # Set only when `code` is None: the longest digit run OCR saw near barcode length,
    # UNMODIFIED, even though it failed the checksum. Never auto-corrected, never looked up,
    # never treated as `code` - it exists so a human reviewer has something concrete to check
    # against the physical pack instead of nothing at all. See find_barcode()'s docstring for
    # why auto-correcting it is not safe.
    unverified_candidate: str | None = None
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


def decode_bars_zbar(rgb: np.ndarray) -> tuple[str | None, np.ndarray | None]:
    """Read the actual black/white bars with ZBar (via pyzbar) - a mature, purpose-built
    barcode-reading library, not OpenCV's general-purpose add-on module.

    Measured meaningfully more reliable than `decode_bars` (cv2) on this dataset: on a
    100-product sample, zbar found 12 valid decodes to cv2's 10, with ZERO wrong reads
    (cv2 had one); 5 of zbar's hits were on photos cv2 found nothing at all on, and every
    one of those 5 matched the ground-truth barcode exactly. Both decoders are still tried
    (see find_barcode) since their failure cases don't fully overlap - the ~3 products cv2
    alone caught are worth keeping.

    Restricted to EAN13/UPCA symbol types: zbar also reads QR codes, Code128, etc., which
    are not product barcodes here. Returns None, None if the `pyzbar`/`libzbar` dependency
    isn't installed (see the module-level import at the top of this file) - never raises.
    """
    if _pyzbar is None:
        return None, None
    for sym in _pyzbar.decode(rgb, symbols=_PRODUCT_SYMBOLS):
        data = sym.data.decode(errors="ignore")
        if data.isdigit():
            points = np.array([[p.x, p.y] for p in sym.polygon])
            return data, points
    return None, None


def _localize_barcode_regions(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Candidate barcode bounding boxes, via the classic gradient/morphology localiser.

    A 1-D barcode is a patch of strong gradient in ONE direction (across the bars) and weak
    in the other (along them). Subtracting the two Sobel gradients isolates exactly that,
    then a morphological close + open welds the bars into one solid blob whose bounding box
    is the barcode. Done for BOTH gradient orientations so a rotated/vertical barcode (common
    on cans - measured on a Kingfisher can) is found too. Returns the largest few boxes; the
    caller crops and decodes each, so extra non-barcode boxes cost time, never correctness.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1))
    gy = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=-1))
    boxes: list[tuple[int, int, int, int]] = []
    for grad, kernel_wh in ((cv2.subtract(gx, gy), (21, 7)), (cv2.subtract(gy, gx), (7, 21))):
        blurred = cv2.blur(grad, (9, 9))
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_wh)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:4]:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > 400:
                boxes.append((x, y, w, h))
    return boxes


def _decode_upscaled(img: np.ndarray) -> str | None:
    """Try zbar on Otsu-binarised, upscaled, every-90-degrees variants of `img`.

    The two levers that matter on this dataset's low-resolution catalog images (measured):
      * UPSCALE - a barcode only ~1-2 px per module is below what zbar can track; cubic
        upscaling to 3-4x gives it room. (Below ~1px/module the information is genuinely
        gone and nothing recovers it - e.g. a 60px barcode in a 420px thumbnail.)
      * OTSU binarisation - snaps blurred grey bars to clean black/white. This is the
        decisive step: measured an Epigamia barcode that NEVER decodes as a plain upscale
        but decodes reliably (3/3) once Otsu'd at 4x. Determinism confirmed.
    Rotations cover barcodes printed sideways (cans). Returns the first decode (zbar
    validates the EAN-13/UPC-A check digit itself, so this is already checksum-clean).
    """
    for scale in (2, 3, 4):
        big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(big, cv2.COLOR_RGB2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_rgb = cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB)
        for rot in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
            variant = cv2.rotate(otsu_rgb, rot) if rot is not None else otsu_rgb
            code, _ = decode_bars_zbar(variant)
            if code:
                return code
    return None


def decode_bars_robust(rgb: np.ndarray) -> tuple[str | None, np.ndarray | None]:
    """Decode the bars, trying HARD - the deterministic, trustworthy signal, pushed as far
    as it goes before the pipeline falls back to reading printed digits.

    Escalates cheapest-first (most images stop at step 1; only failures pay for the rest):
      1. both decoders on the image as-is (zbar, then cv2) - the fast path.
      2. whole-image upscale + Otsu + rotations (`_decode_upscaled`).
      3. localise candidate barcode regions, and run step 2 on each tight crop - this is what
         recovers a small barcode that is lost in the noise of the full frame.

    Every result is checksum-validated here (zbar already validates internally; cv2 does
    not, and has been measured returning a flat-out wrong code, so the gate matters). Returns
    (code, points, source): source is "direct" (decoded the image as-is - trustworthy),
    "upscaled" or "localized" (decoded only after aggressive upscaling/localising - which can
    HALLUCINATE a checksum-valid code out of busy graphics, measured "080100008021" invented
    from a marketing panel, so the caller must corroborate these against the printed digits
    before trusting them). `points` is a real quadrangle for a direct decode, the localiser's
    box for "localized", else None.

    The escalation is deterministic but slow (steps 2-3 run dozens of zbar passes), so the
    result is cached on the SHA-256 of the exact pixels - the same image never pays twice.
    """
    cached = _bars_cache_get(rgb)
    if cached is not None:
        return cached
    result = _decode_bars_robust_compute(rgb)
    _bars_cache_put(rgb, result)
    return result


def _decode_bars_robust_compute(rgb: np.ndarray) -> tuple[str | None, np.ndarray | None, str]:
    for fn in (decode_bars_zbar, decode_bars):
        code, points = fn(rgb)
        if code and checksum_valid(code):
            return code, points, "direct"

    code = _decode_upscaled(rgb)
    if code and checksum_valid(code):
        return code, None, "upscaled"

    for (x, y, w, h) in _localize_barcode_regions(rgb):
        pad = 8
        crop = rgb[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        if crop.size == 0:
            continue
        code = _decode_upscaled(crop)
        if code and checksum_valid(code):
            box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            return code, box, "localized"

    return None, None, ""


def _bars_cache_get(rgb: np.ndarray) -> tuple[str | None, np.ndarray | None, str] | None:
    """Cached (code, points, source) for these exact pixels, or None for a cache MISS
    (distinct from a cached miss, stored as (None, None, ""))."""
    key = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
    with _cache() as c:
        row = c.execute(
            "SELECT code, points, source FROM bars_decode_v2 WHERE key = ?", (key,)
        ).fetchone()
    if row is None:
        return None
    code = row[0] or None
    points = np.array(json.loads(row[1])) if row[1] else None
    return code, points, row[2] or ""


def _bars_cache_put(rgb: np.ndarray, result: tuple[str | None, np.ndarray | None, str]) -> None:
    key = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
    code, points, source = result
    pts_json = json.dumps(np.asarray(points).tolist()) if points is not None else None
    with _cache() as c:
        c.execute(
            "INSERT OR REPLACE INTO bars_decode_v2 (key, code, points, source) VALUES (?, ?, ?, ?)",
            (key, code, pts_json, source),
        )


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
    coincidental run.

    Deliberately NOT widened further (e.g. re-reading every digit-dense line on the pack at
    higher sensitivity when this fails). Tried it, measured it: an FSSAI licence number
    ("Lic No 1001304200197", 13 digits) and a phone number with country code ("91 73585
    55587", 12 digits) both sit in this exact length window, and a corrupted re-read of
    either can coincidentally pass the EAN-13 checksum - which the checksum is not immune
    to; it only catches ~90% of RANDOM errors. On a 60-product sample that "rescue" produced
    2 checksum-valid codes and both were wrong (a licence number, not the real barcode). A
    false "checksum valid" is worse than reporting nothing: it would feed a wrong code into
    the Open Food Facts lookup and brand cross-check as if verified. Left out on purpose.
    """
    runs = re.findall(r"\d{11,14}", re.sub(r"[^\d]", " ", text))
    return sorted(set(runs), key=len, reverse=True)


def barcode_line_runs(text: str) -> list[str]:
    """Digit runs from lines that are PURE digits - no words anywhere on the line.

    This is the shape a barcode's own printed number takes on a pack: a run of digits sitting
    alone (`8901052034178`), sometimes with a stray guard glyph like `>`. Every OTHER long
    number on a pack carries a text label on the SAME OCR line - "Lic. No. 100140310010025",
    "FREE NUMBER: 1800 345 1720", "Survey No. 460/2" - so requiring ZERO alphabetic characters
    on the line cleanly separates the real barcode digits from licence/phone/address numbers
    that merely happen to be barcode-length.

    Also tries MERGING 2-3 consecutive short (<11-digit) pure-digit lines. An EAN-13's own
    printed human-readable text is conventionally rendered as three groups with a visible gap
    between them - one leading digit, then two sixes - mirroring the bars' own left-guard /
    centre-guard / right-guard structure. RapidOCR's line detector treats that gap as a line
    break, so the barcode's digits often arrive as 2-3 individually-too-short lines rather
    than one: measured a real "8906059638503" split into lines "8906059" (7 digits) and
    "638503" (6 digits), NEITHER of which alone reaches the 11-digit floor, so the true
    barcode was never even considered before this. Concatenating adjacent short digit-only
    lines reassembles it exactly. Safe in the way that matters here: it only reassembles text
    OCR already read correctly but split apart, never invents a digit, and the result is still
    checksum-gated like every other candidate before it is ever trusted.

    Returned closest-to-13-digits first (a barcode is 12-13 digits).
    """
    pure_digit_lines: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or any(c.isalpha() for c in s):
            continue
        digits = re.sub(r"[^\d]", "", s)
        if digits:
            pure_digit_lines.append(digits)

    runs: set[str] = set()
    for i, d in enumerate(pure_digit_lines):
        if 11 <= len(d) <= 14:
            runs.add(d)
        if len(d) >= 11:
            continue
        merged = d
        for nxt in pure_digit_lines[i + 1 : i + 3]:
            if len(nxt) >= 11:
                break  # a full number of its own, not a fragment - stop merging
            merged += nxt
            if 11 <= len(merged) <= 14:
                runs.add(merged)
            if len(merged) > 14:
                break

    return sorted(runs, key=lambda r: abs(len(r) - 13))


def is_restricted_prefix(code: str) -> bool:
    """True if `code`'s GS1 prefix is 'restricted distribution' - in-store / variable-measure
    codes (leading '2', i.e. 20-29, and the '02'/'04' prefixes). By the GS1 standard these are
    NEVER assigned to real, registered retail products, so a supposed product barcode with one
    of these prefixes is not a real barcode. Used to reject a RECOVERED code that only happens
    to pass the checksum - e.g. a batch/lot number recovered into '2100009309000' - without
    touching genuinely-decoded codes."""
    return code[:1] == "2" or code[:2] in ("02", "04")


def recover_dropped_leading_digit(run: str) -> str | None:
    """If `run` is 12 digits, return the 13-digit EAN-13 formed by prepending the UNIQUE digit
    that makes the checksum valid (and whose prefix is a real retail one); else None.

    Safe in a way middle-digit correction is NOT. The leading position carries weight 1
    (coprime to 10) and the check digit is one of the 12 digits we already hold fixed, so
    exactly one prepended digit satisfies the checksum - always exactly one (verified over
    2000 random 12-digit strings), never a choice between several. It targets one specific,
    common OCR failure: the lone digit in a barcode's quiet zone, set apart from the bar block,
    gets dropped (measured: RapidOCR read a real "8901052034178" as "901052034178"). Two gates
    keep it from fabricating barcodes: it is applied ONLY to `barcode_line_runs` (pure-digit
    lines, never licence/phone numbers), and the recovered code must have a real retail GS1
    prefix - a batch code that recovers to a restricted-distribution prefix like "21000..." is
    rejected (measured false positive: run "100009309000" -> "2100009309000")."""
    if len(run) != 12:
        return None
    for x in "0123456789":
        cand = x + run
        if checksum_valid(cand) and not is_restricted_prefix(cand):
            return cand
    return None


def find_barcode(
    rgb: np.ndarray, ocr_text: str = ""
) -> tuple[str | None, str, np.ndarray | None, list[str], str | None]:
    """Find the barcode, cheapest signal first. Returns
    (code_or_None, source, points_or_None, trail, unverified_candidate_or_None), source one of
    "bars" | "ocr_digits" | "ocr_digits_recovered" | "".

    Order:
      1. decode the actual bars - zbar first, then cv2 (`decode_bars_zbar` / `decode_bars`;
         see their docstrings for why both, and why zbar goes first). EITHER decoder can
         return a wrong value (measured: cv2 returned a flat "92113800" on a pack whose real
         code was "8901414001626"), so a bars result is trusted only once it ALSO passes the
         checksum - an invalid one is discarded and the search continues, it does not stop
         the whole pipeline.
      2. the pack's own printed barcode line (`barcode_line_runs`: a pure-digit line): accept
         it if the checksum passes, or if it is 12 digits and a UNIQUE dropped-leading-digit
         recovery makes it pass (`recover_dropped_leading_digit`).
      3. any checksum-valid digit run anywhere in the text (old fallback).

    Middle-digit correction is deliberately NOT attempted. Flipping one digit until the
    checksum passes is not a fix: for ANY 13-digit string, exactly one flip at EVERY one of the
    13 positions makes the checksum pass (mod-10, weights 1 and 3, both coprime to 10) - 13
    different "valid" codes with no way to tell which is real. Measured: RapidOCR read a real
    "8904055800108" as "8906055800108" and single-digit correction produced 13 valid strings,
    one correct. Recovering a DROPPED LEADING digit (step 2) is different and safe: it has
    exactly ONE solution, and is gated to pure-digit barcode lines so a licence/phone number
    can't be turned into a fabricated barcode. When nothing passes, the best barcode-line run
    is surfaced UNMODIFIED as `unverified_candidate` for a human to check - never a guess.
    """
    trail = []

    # Deterministic bar decode, pushed hard (both decoders, then upscale+Otsu+rotate, then
    # localised regions - see decode_bars_robust). Checksum-gated inside, so a wrong cv2
    # read is discarded rather than trusted.
    bars, points, decode_src = decode_bars_robust(rgb)
    if bars:
        # No decoder is infallible: measured a degraded pack where a DIRECT zbar decode
        # returned a checksum-valid but WRONG "0080100008021" (an OFF lookup for a padlock),
        # and an escalated decode inventing a code out of promo-panel graphics. But blanket-
        # requiring the printed digits to corroborate EVERY decode over-rejects - many real
        # barcodes have their printed number OCR-garbled (the correct milk-carton decode
        # corroborates at only 46%), and gating on that cost 10 of 28 real decodes for no FP
        # benefit. Gating BOTH direct+escalated equally also over-rejected (cost 7 more real
        # decodes for no FP gain) - the reliable discriminator is the PREFIX, not the decode
        # path: every real product here is EAN-13 "890…" (India) or another real country code
        # (leading 3-9); both observed phantoms were leading-0 UPC-A. So: a real-looking prefix
        # is trusted outright (direct or escalated, garbled print or not - that's where the
        # recall lives); only an implausible leading 0/1/2 needs the printed digits to agree.
        suspicious = bars[:1] in ("0", "1", "2")
        corroboration = text_match(bars, ocr_text)
        if not suspicious or corroboration >= 0.5:
            note = decode_src if not suspicious else f"{decode_src}, corroborated {corroboration:.0%}"
            trail.append(f"decoded from bars ({note}): {bars}")
            return bars, "bars", points, trail, None
        trail.append(f"bars {decode_src}-decoded {bars} (leading {bars[:1]} implausible, printed "
                     f"digits match only {corroboration:.0%}); discarding as likely wrong")
    trail.append("bar decode: no valid barcode found in image (tried upscale + localise)")

    # VERIFIED path: a checksum-valid EAN-13 on the pack's OWN barcode line (pure digits, no
    # words). Preferred over any labelled number ("Lic. No. ...", "FREE NUMBER: ...") because
    # those merely happen to be barcode-length and can pass the checksum by coincidence.
    #
    # Require exactly 13 digits. Indian retail barcodes are EAN-13 (13 digits, "890..."); a
    # bare 12-digit run here is far more likely a truncated FSSAI/batch fragment than a real
    # UPC-A - measured false positive: a "100120220217" fragment passed the 12-digit (UPC-A)
    # checksum and was wrongly reported as a product barcode. 12-digit runs are instead handled
    # by the UNVERIFIED leading-digit-recovery path below, never auto-trusted.
    line_runs = barcode_line_runs(ocr_text)
    for run in line_runs:
        if len(run) == 13 and checksum_valid(run):
            trail.append(f"barcode digit line, checksum-valid: {run}")
            return run, "ocr_digits", None, trail, None

    # Fallback VERIFIED path: a checksum-valid EAN-13 from anywhere in the text. Kept for the
    # case where OCR merged the barcode digits onto a line with other text, but it IS a known
    # source of the occasional false positive, so it runs only after the pure-digit lines
    # above yield nothing, and is likewise restricted to exactly 13 digits.
    candidates = candidates_from_text(ocr_text)
    for cand in candidates:
        if len(cand) == 13 and checksum_valid(cand):
            trail.append(f"OCR digits, checksum-valid: {cand}")
            return cand, "ocr_digits", None, trail, None
    if ocr_text:
        trail.append("OCR digits: no checksum-valid 13-digit run found")

    # Nothing verified. Build the best UNVERIFIED candidate for a human to check. Recovering a
    # dropped leading digit is offered HERE, not as a verified code: it is right when OCR
    # merely lost the lone quiet-zone digit (measured: "901052034178" -> "8901052034178"), but
    # WRONG when the barcode line was misread elsewhere and the missing digit isn't at the
    # front (measured: a run already starting "890109..." recovered to a bogus Brazil-prefix
    # "7890109500401"). Checksum-valid-after-recovery is a strong hint, never a guarantee - so
    # it is surfaced for review, not trusted.
    # The candidate is drawn ONLY from `line_runs` (the pack's own pure-digit barcode lines),
    # never from `candidates_from_text` (labelled numbers). Surfacing an FSSAI/phone number as
    # a "barcode candidate" is worse than surfacing nothing - measured on a can whose barcode
    # was too low-res for OCR: the fallback offered its licence number "10018033000336", which
    # reads as a confidently wrong barcode. If no real barcode line was read, we say so.
    unverified = None
    for run in line_runs:
        recovered = recover_dropped_leading_digit(run)
        if recovered:
            unverified = recovered
            trail.append(
                f"unverified candidate for manual review: {recovered} "
                f"(recovered a possible dropped leading digit from barcode line {run}; "
                f"checksum valid but NOT auto-trusted)"
            )
            break
    if unverified is None and line_runs:
        unverified = line_runs[0]
        trail.append(
            f"unverified candidate for manual review: {unverified} (checksum invalid)"
        )
    if unverified is None:
        trail.append("no barcode digit line could be read; nothing to surface for review")
    return None, "", None, trail, unverified


# ---------------------------------------------------------------- lookup


def _cache() -> sqlite3.Connection:
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS lookups (code TEXT PRIMARY KEY, payload TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS upcitemdb_lookups (code TEXT PRIMARY KEY, payload TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS bars_decode_v2 "
              "(key TEXT PRIMARY KEY, code TEXT, points TEXT, source TEXT)")
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
    code, source, points, trail, unverified = find_barcode(rgb, ocr_text)

    if code is None:
        return BarcodeResult(trail=trail, unverified_candidate=unverified)

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
