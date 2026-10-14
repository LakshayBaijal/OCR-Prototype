"""Read one in-memory image into a structured ProductExtraction, twice over.

The existing `stage2.pipeline.collect_lines_and_barcode` reads a PRODUCT - a group of
file paths, front/back/panel, unioned - because that's the only honest way to score a
catalog record when the FSSAI number is on one face and the MRP on another. This
module deliberately does the opposite: ONE image, held in memory, never touching disk.

That difference is the whole point of this package. We are not asking "what is this
product" (many faces, union everything). We are asking "did THIS ONE image survive
being redrawn" - and the only fair way to answer that is to run identical extraction
over the before and the after of that single frame. Unioning in other faces would
import text that the enhanced image never claimed to contain.

ONE reading path, deliberately: `ocr_read` runs RapidOCR and stops at lines, boxes
and confidences. It does not build named fields. The VLM paths this module used to
carry (a local MLX-served Gemma, and Gemini) are gone - they fabricated fields
constantly, and every control added to suppress that was another reason not to trust
the output. `verify.signals` compares what was read without naming it, and
`verify.labeled` names only the four fields a pack labels explicitly.

FAIRNESS RULE, and the reason this module exists rather than calling the Stage-0
pipeline: whatever preprocessing runs, runs IDENTICALLY on both images. The enhanced
image arrives upright, sharp and evenly lit because that is what the enhancer does;
if the original were handed to OCR raw while the enhanced one got a clean run, every
field OCR failed to find in the blurry original would be reported as "the enhancer
invented this" - the verifier would be measuring its own unfairness. So both sides
get the same deskew/normalise/resize, and neither gets the background-crop or
product-grouping steps, which change FRAMING and could legitimately cut different
content out of two differently-framed images.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps

import barcode as barcode_mod
from . import ocr_engine
from preprocessing import step_deskew, step_normalise, step_resolution


@dataclass
class ImageReading:
    """Everything RapidOCR read out of one image - lines, text, barcode. Nothing here
    is interpreted into named fields; that is `verify.labeled`'s narrow job."""

    ocr_text: str                                   # newline-joined lines
    barcode: barcode_mod.BarcodeResult | None
    engine: str
    lines: int = 0
    lines_data: list = field(default_factory=list)  # OCRLine objects: text, bbox, confidence
    mean_confidence: float | None = None
    lang: str = ""
    elapsed: float = 0.0
    error: str | None = None


# ---------------------------------------------------------------- loading


def load_rgb_bytes(data: bytes) -> np.ndarray:
    """Decode uploaded bytes to an RGB array, honouring EXIF orientation.

    PIL applies the EXIF rotation tag; OpenCV ignores it entirely. A phone photo that
    is portrait on screen but landscape in its pixel buffer would otherwise reach OCR
    sideways - and sideways text reads as nothing at all, which the verifier would
    then blame on the enhancer.
    """
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)).convert("RGB"))
    return np.asarray(img)


def to_png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def to_jpeg_bytes(rgb: np.ndarray, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def crop_to_product(
    rgb: np.ndarray, pad_frac: float = 0.02
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Crop to just the product, dropping the scene around it.

    Returns (image, foreground_mask, note) - the mask cropped to match the image, or
    None when no usable matte was produced. The mask is returned as well as used
    because a BOUNDING BOX is not enough for every caller: a bottle held in front of a
    shelf has a tall narrow silhouette, so its box still contains a lot of shelf, and
    `marks.detect_mark` was finding "veg marks" in that leftover clutter. Callers that
    need "is this ON the product" rather than "is this near the product" need the
    silhouette itself. OCR still only needs the crop.

    This is the single highest-value correction in the whole package, and it exists
    because of a measured failure, not a theory. The originals are supermarket SHELF
    photos: a product held in one hand, with price tags, promo signs and a dozen
    neighbouring packs behind it. RapidOCR reads all of that, indiscriminately - and
    removing exactly that background is the ENHANCER'S JOB. So before this step, every
    field that came from background clutter showed up as LOST or CHANGED, and the
    enhancer was being penalised for doing precisely what it was asked to do. Measured
    on the audit set: img1's "Net Quantity 330 ml" came off a shelf tag rather than the
    bottle; img20's Brand read "BiO" off neighbouring packs, so the enhanced image's
    correct "Agrino" was reported as the corruption; the visual check's "MRP 1.69 ->
    1.49" was two unrelated shelf labels.

    The mask comes from `bgremove` (u2netp salient-object segmentation, ~0.15s and
    cached on the pixels) rather than the classical `preprocessing.segment` colour
    heuristic. That choice matters: `ondc-intelligence/image_enhancement.py`'s own
    docstring records that its Otsu+contour crop was REMOVED because it "too often
    locked onto background clutter instead of the product" - which is the exact failure
    mode a shelf photo provokes. A learned salient-object model does not key on
    backdrop colour at all, so it does not share that failure. Verified before wiring
    it in: cropping removed the phantom price tag from img12 and the neighbouring-pack
    brand from img20, while the product's own text survived intact (img12 actually
    gained the nutrition line RapidOCR had been missing).

    The CROP itself uses only the mask's bounding box, never its per-pixel silhouette,
    so a slightly wrong matte costs a looser crop and never an erased label - the same
    rule `bgremove`'s own docstring sets for the Stage-0 pipeline. The silhouette is
    handed back separately for callers that need it (see above); no pixel is ever
    whited out here.

    Degrades to the uncropped image (never raises) when rembg isn't installed, the
    matte comes back empty, or the mask is implausibly small - a suspiciously tiny mask
    means the model latched onto a speck, and cropping to that would destroy the read
    rather than clean it.
    """
    try:
        import bgremove
    except ImportError:
        return rgb, None, "rembg not installed - read the whole frame, background included"
    if not bgremove.available():
        return rgb, None, "rembg unavailable - read the whole frame, background included"

    try:
        mask = bgremove.mask(rgb)
    except Exception as exc:
        return rgb, None, f"foreground matte failed ({type(exc).__name__}) - read the whole frame"
    if mask is None:
        return rgb, None, "no foreground matte - read the whole frame, background included"

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return rgb, None, "empty foreground matte - read the whole frame"

    h, w = rgb.shape[:2]
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    area_frac = ((x1 - x0) * (y1 - y0)) / float(w * h)
    if area_frac < 0.02:
        return rgb, None, (f"foreground matte covered only {area_frac:.1%} of the frame - too "
                     f"small to trust, so nothing was cropped")

    pad_x, pad_y = int(w * pad_frac), int(h * pad_frac)
    x0, x1 = max(0, x0 - pad_x), min(w, x1 + pad_x + 1)
    y0, y1 = max(0, y0 - pad_y), min(h, y1 + pad_y + 1)
    cropped = rgb[y0:y1, x0:x1]
    cropped_mask = mask[y0:y1, x0:x1]
    return cropped, cropped_mask, f"cropped to the product ({area_frac:.0%} of the original frame)"


def prepare_for_ocr(rgb: np.ndarray) -> np.ndarray:
    """The identical treatment both images get before OCR.

    Deliberately almost nothing, and that is a measured decision rather than laziness.
    An earlier version ran deskew + CLAHE + a clamp to 2200px; comparing recovered
    characters across three packs showed CLAHE LOSING text every time (img12 495->487,
    img15 1992->1846, img40 112->76) and the downscale costing more on the dense back
    panel. PP-OCR normalises contrast and resizes internally, so a second pass on top
    fights its own preprocessing. See verify/ocr_engine.py for the full table.

    What remains: deskew (a no-op on straight crops, real help on a rotated one) and an
    UPSCALE-ONLY resize, so a genuinely small crop is enlarged for the detector while a
    large one is never thrown away. The FAIRNESS RULE still holds - one function, called
    on both images, so neither side gets a treatment the other did not.
    """
    rgb = step_deskew(rgb, max_angle=15.0).image
    # min_side 1200, no upper bound: enlarge small crops, never shrink large ones.
    return step_resolution(rgb, min_side=1200, max_side=100_000).image


# ---------------------------------------------------------------- barcode


def decode_barcode(rgb: np.ndarray, ocr_text: str = "", lookup: bool = False):
    """Decode the barcode from one image.

    `lookup=False` by default, and that default matters for correctness here, not just
    latency. `read_and_validate` finishes by querying Open Food Facts and writing the
    canonical brand/name it gets back onto the result - and `stage2.heuristic`
    trusts that canonical brand at confidence 0.9, above anything it reads from the
    pixels. In a before/after comparison that is poison: both images decode the same
    barcode, so both inherit the SAME externally-fetched brand, and a genuinely
    corrupted brand on the pack compares as a perfect match. The lookup is a fine
    thing to show a reviewer; it must not be allowed to answer the question we are
    asking. So field extraction always runs on a pixel-only decode, and the lookup is
    opt-in for display alone.
    """
    if lookup:
        return barcode_mod.read_and_validate(rgb, ocr_text)
    code, source, points, trail, unverified = barcode_mod.find_barcode(rgb, ocr_text)
    if code is None:
        return barcode_mod.BarcodeResult(trail=trail, unverified_candidate=unverified)
    return barcode_mod.BarcodeResult(
        code=code, valid_checksum=barcode_mod.checksum_valid(code),
        source=source, points=points, trail=trail,
    )


# ---------------------------------------------------------------- the OCR read


# Minimum fraction of a detected line's box that must fall on the product silhouette
# to be kept. Measured on a hand-held bottle against a packed shelf (bbox crop was 41%
# of the frame; the actual bottle only 24%): every shelf-tag/neighbouring-product line
# scored 0.00-0.18 overlap and every line actually printed on the bottle scored
# 0.55-1.00 - a clean gap, not a borderline call.
_PRODUCT_OVERLAP_THRESHOLD = 0.3


def _filter_to_product(lines: list, mask: np.ndarray | None) -> tuple[list, int]:
    """Drop detected lines that sit mostly OFF the product silhouette.

    This exists because a bounding-box crop is not the same as background removal, and
    trying to make it one made things WORSE. Whiting out every pixel outside the mask
    before running OCR was tried first and measured: on a glass bottle (thin, curved,
    reflective - exactly what salient-object segmentation struggles with) it cut
    recovered text from 250 to 34 characters, because the imperfect matte clipped into
    the label along with the shelf. Even a heavily dilated mask only recovered to 48
    characters. RapidOCR's detector evidently needs the FULL image's contrast and
    context to find text reliably; pre-masking pixels starves it.

    So the mask is applied AFTER detection instead: OCR sees the whole crop and reads
    it at full quality, exactly as before, and only then is each detected line's own
    box checked against the mask - keeping the pixels intact and discarding only the
    LOCATIONS that turned out to be background. Verified on the same bottle: every
    shelf-tag and neighbouring-product line (four-plus-two price promos, "Henninger",
    "3,72 E") scored 0.00-0.18 overlap and was dropped; every line actually printed on
    the bottle scored 0.55-1.00 and was kept - a clean separation, not a borderline call.

    Returns (kept_lines, n_dropped). Falls through unchanged when no mask is available
    (rembg missing, or the crop step degraded to uncropped) - never a reason to lose a
    read.
    """
    if mask is None or not lines:
        return lines, 0

    kept, dropped = [], 0
    for ln in lines:
        xs = [p[0] for p in ln.bbox]
        ys = [p[1] for p in ln.bbox]
        x0, x1 = max(0, min(xs)), min(mask.shape[1], max(xs))
        y0, y1 = max(0, min(ys)), min(mask.shape[0], max(ys))
        region = mask[y0:y1, x0:x1]
        overlap = float((region > 0).mean()) if region.size else 0.0
        if overlap >= _PRODUCT_OVERLAP_THRESHOLD:
            kept.append(ln)
        else:
            dropped += 1
    return kept, dropped


def ocr_read_paddle_vl(rgb: np.ndarray) -> ImageReading:
    """PaddleOCR-VL-1.5 (local, via MLX - see stage1_ocr.py) on one prepared image.

    Kept as a SEPARATE function from `ocr_read`, not a branch inside it, because the
    two engines return genuinely different shapes of information and pretending
    otherwise would be the bug: RapidOCR detects individual text lines with a real
    box and a real per-line confidence; this model's plain "OCR:" task returns one
    flowing transcription of the whole crop, in reading order, with neither.

    That gap has a real, specific consequence rather than being a vague caveat.
    Two things downstream key on genuine per-line geometry, and both degrade for a
    reading built this way:
      - `_filter_to_product` drops lines whose box falls mostly off the product
        silhouette (shelf tags, neighbouring packs). Every line here shares the
        same whole-crop box, so nothing is ever dropped as background - this
        reading has no way to tell "printed on the product" from "in the crop".
      - `signals.prominent_lines` sizes lines by box height as a brand/name proxy.
        With every line the same height, it falls back to reading order, so the
        "Prominent text" check is weaker for this engine, not equivalent.
      Everything else - barcode decode, digit/quantity/price/word-set comparison,
      the four labelled fields (which need `.text` and `.confidence`, never
      `.bbox`) - works from the flat text and is unaffected either way.

    So this exists for the comparison the user actually asked for - is the raw
    text itself more accurate - not as a silent drop-in replacement for `ocr_read`.
    `report.run_verification`'s caller is expected to say which engine produced a
    reading, not just report a verdict as if the two were interchangeable.
    """
    import time

    started = time.perf_counter()
    try:
        import stage1_ocr
        text = stage1_ocr.read_array(rgb)
    except Exception as exc:
        return ImageReading(ocr_text="", barcode=None, engine="paddleocr-vl-1.5",
                            error=f"{type(exc).__name__}: {exc}")

    h, w = rgb.shape[:2]
    whole_image_bbox = [(0, 0), (w, 0), (w, h), (0, h)]
    # One line per newline PaddleOCR-VL emits, not one line for the whole block -
    # `_prominent_text_check`'s containment test and the word/digit/quantity sets
    # all want individual strings to compare, and collapsing to a single giant
    # line would make "which lines survived" unanswerable.
    lines = [ocr_engine.Line(text=t, bbox=whole_image_bbox, confidence=None)
            for t in text.splitlines() if t.strip()]

    bc = decode_barcode(rgb, text, lookup=False)
    return ImageReading(
        ocr_text=text, barcode=bc, engine="paddleocr-vl-1.5",
        lines=len(lines), lines_data=lines,
        mean_confidence=None,  # the model reports no per-line confidence - see
                               # ocr.types.OCRLine's own convention for this
        lang="",
        elapsed=time.perf_counter() - started,
    )


def ocr_read(
    rgb: np.ndarray, lang: str = "auto", product_mask: np.ndarray | None = None,
) -> ImageReading:
    """RapidOCR on one prepared image, with the recognition model matched to the script.

    This is the whole reading layer. It stops at "here are the lines, their boxes and
    their confidences" and does not call `stage2.heuristic` to turn them into named
    fields - that parser guesses which line is the Brand (confidence 0.4) and nearly
    every false finding in the audit came from that guess rather than from OCR
    misreading characters.

    `lang="auto"` picks the recogniser from the script actually on the pack; see
    verify.ocr_engine.read. Callers comparing a PAIR should resolve the language once
    and pass it explicitly to both sides.

    `product_mask`, when given (from `crop_to_product`, same pixel dimensions as `rgb`),
    drops detected lines that sit mostly off the product - see `_filter_to_product` for
    why this runs after detection rather than by pre-masking pixels.
    """
    import time

    started = time.perf_counter()
    try:
        result = ocr_engine.read(rgb, lang=lang)
    except Exception as exc:
        return ImageReading(ocr_text="", barcode=None, engine="rapidocr",
                            error=f"{type(exc).__name__}: {exc}")

    lines, n_dropped = _filter_to_product(result.lines, product_mask)
    union_text = "\n".join(l.text for l in lines if l.text)
    bc = decode_barcode(rgb, union_text, lookup=False)
    engine_name = f"rapidocr:{result.lang}"
    if result.detected_from:
        engine_name += f" ({result.detected_from})"
    if n_dropped:
        engine_name += f", {n_dropped} background line(s) filtered"

    confidences = [l.confidence for l in lines if l.confidence is not None]
    return ImageReading(
        ocr_text=union_text, barcode=bc, engine=engine_name,
        lines=len(lines), lines_data=lines,
        mean_confidence=(sum(confidences) / len(confidences)) if confidences else None,
        lang=result.lang,
        elapsed=time.perf_counter() - started,
    )
