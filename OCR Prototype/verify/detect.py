"""Bounding boxes for the product(s) in a photo. Category-agnostic, CPU-only.

WHY NOT YOLO - measured, not assumed. torchvision's COCO-pretrained Faster R-CNN was
tried first on this dataset and it does not work for packaged grocery, because COCO has
no class for a box, pouch, sachet or carton. Results on three real inputs:

    Heineken on a shelf   -> 'person' (the hand) + SIX 'bottle's (the held one AND the
                             shelf behind it - no way to tell which is the subject)
    Coco Pops carton      -> 'bowl' (the cereal-bowl artwork) and 'teddy bear' (the
                             monkey mascot). The box itself: undetected.
    Rice pack on a table  -> 'bed'

Any COCO-class detector inherits this - the failure is the label set, not the
architecture, so swapping Faster R-CNN for YOLOv8 would reproduce it. Detecting "a
retail product" as a class would need a model fine-tuned on retail data, which is a
labelling project rather than a library swap.

WHAT IS USED INSTEAD: salient-object segmentation (rembg / u2netp / BiRefNet). It asks
"which pixels are the subject of this photograph" with no notion of category at all, so
a pouch, a carton and a bottle are all equally findable - and on a photo of a held
product, the thing being held IS the salient object, which is exactly the "only the
product I need" selection the shelf behind it fails. Boxes come from connected
components of that mask, so several products are returned when several are genuinely
prominent.

Model tradeoff, measured on this machine (CPU, 3x-downscaled input):

    model                   img1 box    fill    time
    u2netp                  41% frame    59%    0.2s
    birefnet-general-lite   33% frame    74%    7.0s

BiRefNet finds a tighter, cleaner silhouette; u2netp is ~30x faster. Interactive use
wants BiRefNet, a full-directory run wants u2netp - hence `model` is a parameter with no
default opinion baked into the callers.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Downscale before segmentation. These models are trained around ~1024px inputs and get
# no benefit from a 4032px phone photo, while the cost scales with pixels - BiRefNet on
# a full-size frame takes over a minute. Boxes are scaled back to full resolution
# afterwards, so nothing is lost but time.
_SEG_MAX_DIM = 1024

FAST_MODEL = "u2netp"
ACCURATE_MODEL = "birefnet-general-lite"

# A component smaller than this fraction of the frame is noise - a speck of matte, a
# reflection - not a product worth boxing.
_MIN_AREA_FRAC = 0.01


@dataclass
class Box:
    """One detected product region, in FULL-resolution image coordinates."""

    x: int
    y: int
    w: int
    h: int
    area_frac: float     # of the whole frame
    fill: float          # how much of the box is actually mask, not background
    is_primary: bool     # the largest component - the "product you need"

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h

    def crop(self, rgb: np.ndarray, pad_frac: float = 0.02) -> np.ndarray:
        h, w = rgb.shape[:2]
        pad = int(max(h, w) * pad_frac)
        x0, y0 = max(0, self.x - pad), max(0, self.y - pad)
        x1, y1 = min(w, self.x + self.w + pad), min(h, self.y + self.h + pad)
        return rgb[y0:y1, x0:x1]


def product_mask(rgb: np.ndarray, model: str = FAST_MODEL) -> np.ndarray | None:
    """Full-resolution binary mask of the salient product(s). None if unavailable.

    Segmentation runs on a downscaled copy (see `_SEG_MAX_DIM`) and the mask is scaled
    back up - the mask is used for boxes and coarse overlap tests, never for per-pixel
    compositing, so upscaling costs nothing that matters.
    """
    try:
        from PIL import Image
        from rembg import new_session, remove
    except ImportError:
        return None

    h, w = rgb.shape[:2]
    scale = min(1.0, _SEG_MAX_DIM / max(h, w))
    small = rgb if scale == 1.0 else cv2.resize(
        rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    try:
        session = new_session(model)
        out = remove(Image.fromarray(small), session=session, only_mask=True)
    except Exception:
        return None

    mask = np.asarray(out)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = (mask > 128).astype(np.uint8) * 255
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


# Mask R-CNN can name any of COCO's 91 categories, but only a NARROW set of them is
# safe to treat as "definitely not the product" here - this is the direct answer to
# "does this work for a table or an animal too, or only humans":
#
#   SAFE to exclude - a live animal is categorically never a packaged retail product,
#   so seeing "dog" or "cat" is trustworthy evidence of a real dog or cat in frame:
_ANIMAL_CATEGORIES = frozenset({
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
})
#
#   UNSAFE to exclude - COCO's own furniture/container classes overlap the exact
#   objects this pipeline is trying to detect, and it has ALREADY been caught
#   confusing them: this module's own header records a rice pack sitting on a table
#   being classified as `bed`, and a Coco Pops carton as `bowl` + `teddy bear`. Adding
#   "dining table" or "bed" to the exclusion set would mean a misclassified PRODUCT
#   gets treated as background and erased - recreating the original failure in
#   reverse. `bottle`, `cup`, `bowl`, `vase`, `book`, `teddy bear`, `handbag` are the
#   same story: each one is also a shape a real product regularly takes.
#
# So the excluded set is `person` (what this was built for) plus the animal classes
# above, and stops there. Untested against a real animal-in-frame photo - none exist in
# this dataset - but the reasoning is the same either way: Mask R-CNN reading "dog" off
# an actual dog is a different confidence class of claim than it reading "bed" off a
# rice pack, because nothing else in a product photo looks like a dog, while plenty of
# products look like furniture or a container from one COCO-trained model's point of view.
_DISTRACTOR_CATEGORIES = _ANIMAL_CATEGORIES | {"person"}

# Loaded lazily and cached: ~170MB of weights and ~15s on first use, then ~0.7s per
# image on CPU regardless of how many categories are being matched against - the model
# already scores all 91 classes in one forward pass, so widening the set from just
# "person" costs nothing extra.
_distractor_model = None
_distractor_categories: list[str] = []

# Dilate the distractor mask before subtracting. The matte's edge sits a few pixels
# inside the actual hand/animal, and without this a rim of it survives and keeps
# dragging the box outward - the very thing the subtraction is meant to fix.
_DISTRACTOR_DILATE = 21

# Close the mask after subtracting. Fingers (or fur) crossing the front of a pack cut
# it into separate pieces; without a close, `boxes_from_mask` returns the largest
# FRAGMENT rather than the product.
_POST_SUBTRACT_CLOSE = 41


def distractor_mask(rgb: np.ndarray, score_threshold: float = 0.7) -> np.ndarray | None:
    """Full-resolution mask of anything in `_DISTRACTOR_CATEGORIES` - in practice, on
    this dataset, the hand holding the product. Would also catch a real animal in frame;
    deliberately does NOT catch furniture/container classes - see that constant's
    comment for the measured reason why.

    Returns None if torchvision is unavailable or nothing scored above the threshold -
    never an exception, since this is a refinement and its absence should cost box
    quality, not the whole detection.
    """
    global _distractor_model, _distractor_categories
    try:
        import torch
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn)
    except ImportError:
        return None

    if _distractor_model is None:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        _distractor_model = maskrcnn_resnet50_fpn(weights=weights).eval()
        _distractor_categories = weights.meta["categories"]

    h, w = rgb.shape[:2]
    # Quarter resolution: a hand (or an animal) is a large, coarse shape and detecting
    # it does not need the detail OCR does. Keeps this at ~0.7s instead of many seconds.
    small = cv2.resize(rgb, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)

    try:
        import torch

        x = torch.from_numpy(np.ascontiguousarray(small)).permute(2, 0, 1).float() / 255
        with torch.no_grad():
            out = _distractor_model([x])[0]
    except Exception:
        return None

    acc = np.zeros(small.shape[:2], np.uint8)
    found = False
    for label, score, m in zip(out["labels"].tolist(), out["scores"].tolist(), out["masks"]):
        if _distractor_categories[label] in _DISTRACTOR_CATEGORIES and score >= score_threshold:
            acc |= (m[0].numpy() > 0.5).astype(np.uint8)
            found = True
    if not found:
        return None
    return cv2.resize(acc * 255, (w, h), interpolation=cv2.INTER_NEAREST)


def subtract_distractors(mask: np.ndarray, distractors: np.ndarray | None) -> np.ndarray:
    """Remove the holding hand (or a real animal in frame) from a product mask, then
    re-close the result.

    Worth stating why this is a separate step rather than something the segmenter does:
    salient-object models treat a hand as part of the subject, because in most photos it
    IS. On a held-product photo that is wrong for our purposes twice over - the hand
    inflates the box, and the ARM runs off the frame edge and stretches it further
    (measured: a box starting at x=0 that should have started at x=446).
    """
    if distractors is None:
        return mask
    grown = cv2.dilate(distractors, np.ones((_DISTRACTOR_DILATE,) * 2, np.uint8))
    out = cv2.bitwise_and(mask, cv2.bitwise_not(grown))
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE,
                            np.ones((_POST_SUBTRACT_CLOSE,) * 2, np.uint8))


# Grow every box outward by this fraction of its OWN width/height per side. The matte
# reliably under-covers a pack at the edges - a promo panel in a different colour plane,
# a shrink-wrap highlight, a corner in shadow - and on the tuna carton that clipped the
# "2+1 δώρο" panel off the right-hand side. Covering the product completely matters more
# here than a tight box, so the default errs outward. Set to 0.0 for the tightest box
# the mask supports.
_BOX_PAD_FRAC = 0.04


def boxes_from_mask(mask: np.ndarray, min_area_frac: float = _MIN_AREA_FRAC,
                    pad_frac: float = _BOX_PAD_FRAC) -> list[Box]:
    """Connected components of the mask -> product boxes, largest first.

    Components are closed with a small kernel first: a printed label often splits the
    matte into pieces (a dark band across a bottle, a window in a carton), and without
    the close a single product returns as three boxes.

    Each box is then grown outward by `pad_frac` of its own size and clipped to the
    frame - see `_BOX_PAD_FRAC`. `fill` is reported against the PADDED box, so the
    number stays honest about how much of what you see is actually product.
    """
    h, w = mask.shape[:2]
    frame_area = h * w
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                              np.ones((max(3, min(h, w) // 100),) * 2, np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats((closed > 0).astype(np.uint8), 8)
    found: list[Box] = []
    for i in range(1, n):  # 0 is the background label
        x, y, bw, bh, area = stats[i]
        if (bw * bh) / frame_area < min_area_frac:
            continue

        px, py = int(bw * pad_frac), int(bh * pad_frac)
        x0, y0 = max(0, int(x) - px), max(0, int(y) - py)
        x1, y1 = min(w, int(x) + int(bw) + px), min(h, int(y) + int(bh) + py)
        pw, ph = x1 - x0, y1 - y0
        box_area = pw * ph
        found.append(Box(x0, y0, pw, ph,
                         area_frac=box_area / frame_area,
                         fill=float(area) / box_area if box_area else 0.0,
                         is_primary=False))

    found.sort(key=lambda b: -(b.w * b.h))
    if found:
        found[0].is_primary = True
    return found


# If subtracting the hand leaves less than this fraction of the original mask, the
# segmenter had latched onto the HAND rather than the product and there is no product
# mask to refine - so the subtraction is abandoned rather than trusted.
#
# This is not hypothetical. On a flat tuna carton held against a shelf of near-identical
# tuna cartons, BiRefNet's "salient object" came back 87% hand: subtracting it left
# scattered fragments and the box collapsed to a 4%-of-frame sliver at 15% fill, far
# worse than the un-subtracted box. Salient-object models struggle when the subject is
# visually the same as its background, and a held hand is then the most distinctive
# thing in frame. Falling back keeps that case no worse than before.
_MIN_SURVIVING_MASK = 0.25


def detect(
    rgb: np.ndarray, model: str = FAST_MODEL, exclude_distractors: bool = True,
) -> tuple[list[Box], np.ndarray | None]:
    """(boxes, mask) for one image. Empty list when nothing salient was found.

    `exclude_distractors` removes a holding hand, and any real animal in frame, from
    the mask before boxing - see `subtract_distractors` and `_DISTRACTOR_CATEGORIES`
    for exactly what is (and, deliberately, is not) covered. Measured on a held tuna
    carton: box area 33% -> 13% of frame and fill 53% -> 69%, with the box no longer
    starting at the frame edge where the arm ran out of shot. Photos with nothing to
    exclude are unaffected (verified: a pack lying on a table produced an identical box
    either way).
    """
    mask = product_mask(rgb, model=model)
    if mask is None:
        return [], None

    if exclude_distractors:
        distractors = distractor_mask(rgb)
        if distractors is not None:
            refined = subtract_distractors(mask, distractors)
            before, after = (mask > 0).sum(), (refined > 0).sum()
            if before and after / before >= _MIN_SURVIVING_MASK:
                mask = refined
            # else: the mask was mostly distractor - see _MIN_SURVIVING_MASK. Keep the original.

    return boxes_from_mask(mask), mask


def draw(rgb: np.ndarray, boxes: list[Box], show_secondary: bool = True) -> np.ndarray:
    """Boxes drawn on a copy of the image. Primary in green, others in amber.

    Line thickness and font scale off the image's own size - a fixed 2px box is
    invisible on a 4032px phone photo.
    """
    vis = rgb.copy()
    h, w = vis.shape[:2]
    thick = max(2, int(max(h, w) * 0.004))
    font_scale = max(0.5, max(h, w) * 0.0009)

    for i, b in enumerate(boxes):
        if not b.is_primary and not show_secondary:
            continue
        colour = (0, 200, 0) if b.is_primary else (255, 170, 0)
        x0, y0, x1, y1 = b.xyxy
        cv2.rectangle(vis, (x0, y0), (x1, y1), colour, thick)
        label = ("PRODUCT" if b.is_primary else f"#{i + 1}") + f"  {b.area_frac:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        cv2.rectangle(vis, (x0, max(0, y0 - th - thick * 3)), (x0 + tw + 8, y0), colour, -1)
        cv2.putText(vis, label, (x0 + 4, max(th, y0 - thick)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick)
    return vis


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """The silhouette tinted over the image - shows what the box was derived FROM,
    which is what you need to see when a box looks wrong."""
    vis = rgb.copy()
    tint = np.zeros_like(vis)
    tint[:, :, 1] = 255
    m = (mask > 0)[:, :, None]
    return np.where(m, (vis * (1 - alpha) + tint * alpha).astype(np.uint8), vis)
