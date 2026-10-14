"""The veg / non-veg mark - the one field OCR structurally cannot check.

Indian food labelling law (FSS Packaging & Labelling Regulations) requires a symbol,
not a word: a filled circle inside a square outline, green for vegetarian, brown for
non-vegetarian. Plenty of packs print that mark and never write "veg" or "non-veg"
anywhere, which is why `stage2/heuristic.py`'s `_extract_diet_type` - a regex over OCR
text - finds nothing on them, and why `ocr/local_vlm.py` and `stage2/vlm_direct.py`
both single this field out in their own comments as the one their text-shaped view
misses.

That gap matters more here than anywhere else in the pipeline. A generative enhancer
treats a small coloured square near the edge of a pack as visual clutter - exactly the
kind of thing its prompt tells it to clean up - so this is a mark it can plausibly
erase, move, or (worst case) recolour. And a green mark redrawn brown is not a
cosmetic defect: it is a vegetarian product labelled as non-vegetarian.

Two independent detectors, because neither is trustworthy alone:

  * `detect_mark` here - classical colour+shape CV. Free, deterministic, and honest
    about its own limits: a green logo, a leaf graphic, or a green-on-green package
    will all produce candidates, and a mark printed small on a busy background will
    be missed. Treat its answer as a HINT, never a verdict.
  * `verify.vlm_check.check_veg_mark` - Gemini looking at the image and answering the
    question directly. Slower and costs money, but it understands what the symbol IS
    rather than matching a colour range.

The app shows both and lets them disagree visibly. What actually matters for
verification is not the absolute answer on either image - it is whether the two
images AGREE with each other, which is a much easier question than "what is the
correct diet type", and the only one this package needs answered.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# HSV ranges, OpenCV convention (H 0-179, S/V 0-255).
#
# Green is the easy one - the regulation's green is saturated and sits in a part of
# the hue circle that packaging rarely uses for large flat areas.
_GREEN_LO, _GREEN_HI = (35, 70, 40), (85, 255, 255)
# Brown/maroon is genuinely hard: it is dark, desaturated red, and red straddles the
# hue wrap-around, so it needs two ranges. It also overlaps with shadow, with printed
# red text, and with a lot of Indian packaging's base palette - which is why the
# non-veg side of this detector is the weaker one and says so in its confidence.
_BROWN_LO_A, _BROWN_HI_A = (0, 60, 25), (18, 255, 190)
_BROWN_LO_B, _BROWN_HI_B = (165, 60, 25), (179, 255, 190)

# A mark is small but not invisible: roughly 0.01%-4% of the frame. Below that it is
# a JPEG artifact or a speck of noise; above it, it is the packaging's own artwork,
# not a regulatory stamp.
_MIN_AREA_FRAC = 0.0001
_MAX_AREA_FRAC = 0.04


@dataclass
class MarkDetection:
    """One candidate veg/non-veg mark."""

    found: bool
    colour: str | None          # "green" | "brown" | None
    diet_type: str | None       # "Vegetarian" | "Non-Vegetarian" | None
    confidence: float           # 0..1, this detector's own (modest) self-assessment
    bbox: tuple[int, int, int, int] | None  # x, y, w, h
    note: str = ""


def _colour_mask(hsv: np.ndarray, colour: str) -> np.ndarray:
    if colour == "green":
        return cv2.inRange(hsv, np.array(_GREEN_LO), np.array(_GREEN_HI))
    a = cv2.inRange(hsv, np.array(_BROWN_LO_A), np.array(_BROWN_HI_A))
    b = cv2.inRange(hsv, np.array(_BROWN_LO_B), np.array(_BROWN_HI_B))
    return cv2.bitwise_or(a, b)


def _ring_score(mask: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """How much this patch looks like 'filled blob inside a hollow square outline'.

    That structure is what separates the regulatory mark from any other small
    coloured thing on a pack. A solid green dot, a green letter, a leaf - all fail
    this. The test: the border ring of the bounding box should be substantially
    covered by the colour (the square outline), the ring just inside it should be
    mostly EMPTY (the white gap the regulation leaves), and the centre should be
    covered again (the filled circle).
    """
    patch = mask[y:y + h, x:x + w]
    if patch.size == 0 or min(w, h) < 8:
        return 0.0
    patch = (patch > 0).astype(np.float32)

    band = max(1, min(w, h) // 8)
    border = patch.copy()
    border[band:-band, band:-band] = 0
    border_frac = border.sum() / max(1, (patch.size - max(0, (h - 2 * band)) * max(0, (w - 2 * band))))

    cy0, cy1 = int(h * 0.35), int(h * 0.65)
    cx0, cx1 = int(w * 0.35), int(w * 0.65)
    centre = patch[cy0:cy1, cx0:cx1]
    centre_frac = centre.mean() if centre.size else 0.0

    gap = patch[band:cy0, band:w - band] if cy0 > band else np.array([])
    gap_frac = gap.mean() if gap.size else 1.0

    # All three conditions must hold; a product of the three punishes any one failing.
    return float(min(1.0, border_frac) * min(1.0, centre_frac) * (1.0 - min(1.0, gap_frac)))


def detect_mark(rgb: np.ndarray, foreground: np.ndarray | None = None) -> MarkDetection:
    """Best veg/non-veg mark candidate in one image.

    `foreground` is the product silhouette from `extract.crop_to_product`. Pass it
    whenever it exists: a candidate that does not sit ON the product is discarded
    outright, which is the only thing that reliably separates a real mark from
    background clutter here.

    That is not a guess - it is what the measurement forced. On a Heineken shelf photo
    (a product with no veg mark at all, being beer) this detector reported a "brown
    mark" at shape score 0.17, while the genuine green mark on a Mangat Ram pulses pack
    scored 0.18. The two are indistinguishable by shape score, so no threshold can
    separate them and raising the bar would only delete the real detections along with
    the false ones. What actually differs is WHERE they are: the Heineken candidate was
    shelf clutter sitting beside the bottle inside its bounding box, and the Mangat Ram
    one was printed on the pack. A regulatory mark is by definition printed on the
    product, so requiring the candidate to lie on the silhouette is a rule that follows
    from what the mark IS, rather than a tuned constant.

    Green is preferred over brown at equal score, deliberately. Brown's HSV range
    overlaps shadow and ordinary red print (see the range comments above), so a brown
    candidate scoring the same as a green one is the less believable of the two.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    frame_area = rgb.shape[0] * rgb.shape[1]

    fg = None
    if foreground is not None and foreground.shape[:2] == rgb.shape[:2]:
        # Erode a little so a candidate hugging the silhouette's edge - where the matte
        # is least certain and background bleeds in - does not count as "on the product".
        fg = cv2.erode((foreground > 0).astype(np.uint8),
                       np.ones((5, 5), np.uint8), iterations=1)

    best: tuple[float, str, tuple[int, int, int, int]] | None = None
    for colour in ("green", "brown"):
        mask = _colour_mask(hsv, colour)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area_frac = (w * h) / frame_area
            if not (_MIN_AREA_FRAC <= area_frac <= _MAX_AREA_FRAC):
                continue
            aspect = w / h if h else 0
            if not (0.6 <= aspect <= 1.7):  # the mark is square; a logo usually is not
                continue
            if fg is not None:
                # Must sit ON the product, not merely inside its bounding box - see the
                # Heineken/Mangat Ram measurement in this function's docstring.
                on_product = fg[y:y + h, x:x + w].mean() if fg[y:y + h, x:x + w].size else 0.0
                if on_product < 0.6:
                    continue
            score = _ring_score(mask, x, y, w, h)
            if score < 0.05:
                continue
            if colour == "brown":
                score *= 0.8  # see docstring - brown is the less reliable channel
            if best is None or score > best[0]:
                best = (score, colour, (x, y, w, h))

    if best is None:
        return MarkDetection(False, None, None, 0.0, None,
                             "no square-outline-with-filled-centre mark found by colour+shape")

    score, colour, bbox = best
    return MarkDetection(
        found=True, colour=colour,
        diet_type="Vegetarian" if colour == "green" else "Non-Vegetarian",
        # Capped well below 1.0 on purpose: this detector matches a colour and a
        # shape, it does not understand the symbol, and overstating it would invite a
        # reviewer to trust it over the VLM's answer.
        confidence=round(min(0.75, score), 3),
        bbox=bbox,
        note=f"{colour} square-outline mark, shape score {score:.2f}",
    )


def compare_marks(original: MarkDetection, enhanced: MarkDetection) -> tuple[str, str]:
    """(status, explanation) for the mark, as a before/after question.

    Status is one of MATCH / CHANGED / LOST / ADDED / ABSENT, matching
    `verify.compare.Status` so the UI renders this row exactly like a field row - it
    IS a field, just one that needed a different instrument to read.
    """
    if not original.found and not enhanced.found:
        return "ABSENT", "no veg/non-veg mark detected in either image"
    if original.found and not enhanced.found:
        return ("LOST", f"a {original.colour} mark was detected in the original but none "
                        f"in the enhanced image - the enhancer may have erased it")
    if enhanced.found and not original.found:
        return ("ADDED", f"a {enhanced.colour} mark appears in the enhanced image but was "
                         f"not detected in the original - could be a genuine mark the "
                         f"original was too blurry to resolve, or an invented one")
    if original.colour != enhanced.colour:
        return ("CHANGED", f"the mark changed colour: {original.colour} -> "
                           f"{enhanced.colour}. This flips the product's declared diet "
                           f"type and is never an acceptable enhancement artifact")
    return "MATCH", f"{original.colour} mark present in both images"
