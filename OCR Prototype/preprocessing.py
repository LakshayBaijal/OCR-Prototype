"""
Stage 0 - Preprocessing for the OCR pipeline.

Each architecture step is a standalone function returning a StepResult so the
Streamlit app can render the output (and a debug overlay) of every step:

    Format Check -> Orientation Fix -> Deskew -> Normalisation
                 -> Resolution Band -> Crop to Product -> Multi-Image Grouping

All images are RGB uint8 numpy arrays.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps

import bgremove

# Background in this dataset is studio white; treat near-white as background.
WHITE = (255, 255, 255)


@dataclass
class StepResult:
    name: str
    image: np.ndarray
    debug: np.ndarray | None = None
    info: dict = field(default_factory=dict)
    changed: bool = False
    note: str = ""


# ---------------------------------------------------------------- helpers


def load_rgb(path: str) -> np.ndarray:
    """Open as RGB, compositing any transparency onto WHITE.

    Many catalog PNGs carry an alpha channel; a plain .convert("RGB") maps those
    transparent pixels to black, which would hand the background detector a black
    frame and make it believe the product fills the whole image.
    """
    pil = Image.open(path)
    if pil.mode in ("RGBA", "LA") or (pil.mode == "P" and "transparency" in pil.info):
        rgba = pil.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        pil = Image.alpha_composite(canvas, rgba)
    return np.array(pil.convert("RGB"))


def corner_pixels(rgb: np.ndarray, patch: float = 0.05) -> np.ndarray:
    """Pixels from the four corner patches. Kept around for `border_colour` (a single
    representative fill colour for rotation padding, where a little contamination is
    cosmetic, not a correctness risk). For the background *model* itself, see
    `border_ring_pixels` - corners alone are not a safe sample for that.
    """
    h, w = rgb.shape[:2]
    ph, pw = max(8, int(h * patch)), max(8, int(w * patch))
    return np.concatenate([
        rgb[:ph, :pw].reshape(-1, 3), rgb[:ph, -pw:].reshape(-1, 3),
        rgb[-ph:, :pw].reshape(-1, 3), rgb[-ph:, -pw:].reshape(-1, 3),
    ])


def border_ring_pixels(rgb: np.ndarray, frac: float = 0.015) -> np.ndarray:
    """Pixels from a thin strip around the FULL image perimeter.

    Four small corner squares break down the moment a product spans the whole frame
    edge-to-edge - e.g. a tray of many small identical items (pens, sachets) shot close
    together, which routinely touches all four corners at once. That is not a rare edge
    case here: it is the standard layout for a bulk/multipack product photo. A full
    perimeter ring is far harder to swamp - the product would need to hug nearly the
    ENTIRE border, not just a corner, to outweigh the real backdrop in the sample.
    """
    h, w = rgb.shape[:2]
    t = max(6, int(min(h, w) * frac))
    return np.concatenate([
        rgb[:t, :].reshape(-1, 3), rgb[-t:, :].reshape(-1, 3),
        rgb[:, :t].reshape(-1, 3), rgb[:, -t:].reshape(-1, 3),
    ])


def background_model(rgb: np.ndarray, k: int = 3) -> tuple[np.ndarray, float]:
    """Learn the backdrop's colours from a ring around the image border.

    The dataset is not uniformly white studio: plenty of shots stage the product on a
    coloured, gradient backdrop. So rather than hardcoding white, we cluster border
    pixels and let the image tell us what its own background looks like.

    A cluster that only claims a sliver of the ring is not a second backdrop tone - it is
    product bleeding across *part* of the border (a calculator's edge, a tray of pens
    touching one side). Blending its colour into the tolerance would make the model
    swallow the product itself: on one bulk-pens photo this alone inflated the tolerance
    from ~20 to 124, misclassifying most of each pen's body as background and shredding
    the whole tray into noise that later got welded back into a single giant blob. So
    only clusters holding a real share of the ring count toward the backdrop colour or
    its tolerance; a small cluster is dropped, not blended in.
    """
    b = border_ring_pixels(rgb).astype(np.float32)

    if len(b) > 20000:  # subsample: k-means does not need every ring pixel
        b = b[np.random.default_rng(0).choice(len(b), 20000, replace=False)]

    k = min(k, len(np.unique(b, axis=0)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(b, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()

    counts = np.bincount(labels, minlength=k)
    keep = [i for i in range(k) if counts[i] / len(labels) >= 0.08]
    if not keep:  # pathological: every cluster is a sliver - fall back to all of them
        keep = list(range(k))
    centers = centers[keep]

    mask = np.isin(labels, keep)
    nearest = np.min([np.linalg.norm(b[mask] - c, axis=1) for c in centers], axis=0)
    tol = max(15.0, float(np.percentile(nearest, 95)) * 1.3 + 8.0)
    return centers, tol


# KNOWN RESIDUAL LIMITATION (found via regression testing, not fixed - see commands.txt /
# conversation history for the reasoning): a backdrop with a second, strongly-coloured
# decorative element (e.g. a printed leaf/graphic pattern behind the product) can occasionally
# produce an extra "instance" in step_group that is just that decoration, not a real product -
# because the ring sample above only lightly touches such elements and correctly treats them
# as too small a share of the ring to count as backdrop, but they're not product either.
# Deliberately NOT auto-filtered: tested both a size-ratio filter and an edge-density
# ("does this crop have any print on it") filter against real examples, and both would have
# silently dropped genuine product content (a low-print tagline banner, a matte accessory)
# in other images - a worse failure than an occasional harmless empty extra crop, since this
# pipeline scores OCR text as a union across crops, so a blank crop costs a free local OCR
# call and nothing else.


def solidify(fg: np.ndarray, bridge: int = 3) -> np.ndarray:
    """Fill each product's outline solid so it can never be hollowed out from the inside.

    Light packaging (a silver calculator body, a white key) can match the backdrop colour
    closely enough to be deleted. Filling the outline means we only ever remove pixels
    OUTSIDE a product's silhouette - whatever sits within it is kept, whatever its colour.

    `bridge` closes small breaks in an outline before it is filled, and it must stay SMALL.
    At 9px it welded four separately-photographed cake packs, stacked a few pixels apart,
    into a single blob - four products became one crop. The repair work is done by the
    contour FILL, not by the close (the calculator's coverage is identical at every bridge
    size), so a wide bridge buys nothing and silently merges neighbouring products.
    """
    k = np.ones((bridge, bridge), np.uint8)
    closed = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(fg)
    cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    return filled


def edge_density(rgb: np.ndarray, region: np.ndarray) -> float:
    """Fraction of a region that is edges. A backdrop is smooth; product detail is not."""
    edges = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 80, 200)
    return float(edges[region].mean() / 255) if region.sum() > 100 else 0.0


def segment(
    rgb: np.ndarray,
    min_area_frac: float = 0.002,
    max_erase_structure: float = 0.015,
) -> tuple[np.ndarray, str]:
    """Mask of the product(s), plus the reason for the verdict.

    Prefers a learned matte (bgremove / BiRefNet) when rembg is installed - it gives a clean
    silhouette on the glossy / white-on-white / gradient packs the colour heuristic mangles.
    Falls back to the classical `_heuristic_segment` when the model is unavailable, so the
    pipeline never hard-depends on it. Either way the mask is only ever used to CROP.
    """
    m = bgremove.mask(rgb)
    if m is not None:
        return _refine_model_mask(rgb, m, min_area_frac)
    return _heuristic_segment(rgb, min_area_frac, max_erase_structure)


def _refine_model_mask(
    rgb: np.ndarray, m: np.ndarray, min_area_frac: float
) -> tuple[np.ndarray, str]:
    """Tidy a raw model matte into the pipeline's mask contract: fill each product solid so
    its interior can never hollow out, drop specks, and bail to "keep whole" if the matte is
    degenerate (near-empty or near-full)."""
    h, w = rgb.shape[:2]
    keep_all = np.full((h, w), 255, np.uint8)

    fg = solidify(m)  # fill outlines solid; interior can never be punched through

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    cleaned = np.zeros_like(fg)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_frac * h * w:
            cleaned[labels == i] = 255

    coverage = float(cleaned.mean() / 255)
    if coverage < 0.02:
        return keep_all, f"model matte nearly empty (product {coverage:.1%}); kept whole"
    if coverage > 0.98:
        return cleaned, f"model matte: product fills the frame ({coverage:.1%})"
    return cleaned, f"foreground matte from {bgremove.MODEL} (product {coverage:.1%})"


def _heuristic_segment(
    rgb: np.ndarray,
    min_area_frac: float = 0.002,
    max_erase_structure: float = 0.015,
) -> tuple[np.ndarray, str]:
    """Classical colour + border-connectivity fallback (used only when rembg is absent).

    Mask of the product(s), plus the reason for the verdict.

    Background = pixels close to the learned backdrop colours AND reachable from the image
    border. Border-connectivity keeps a product intact: a white patch enclosed inside a
    pack cannot be reached from the edge, so it stays foreground rather than punching a
    hole through the product.

    Everything here is biased toward NOT destroying the image. If any check looks wrong,
    we return "all foreground" - crop nothing, erase nothing - and let OCR read the
    original. An uncropped image is a minor loss; an image with its label erased is fatal.
    """
    h, w = rgb.shape[:2]
    keep_all = np.full((h, w), 255, np.uint8)

    centers, tol = background_model(rgb)

    flat = rgb.reshape(-1, 3).astype(np.float32)
    dist = np.full(len(flat), np.inf, np.float32)
    for c in centers:  # nearest-centre distance, one centre at a time to stay light on memory
        dist = np.minimum(dist, np.linalg.norm(flat - c, axis=1))

    candidate = ((dist <= tol).reshape(h, w) * 255).astype(np.uint8)

    n, labels, _, _ = cv2.connectedComponentsWithStats(candidate, 8)
    if n <= 1:
        return keep_all, "no backdrop-coloured pixels found; nothing to discard"

    border = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    bg_labels = [l for l in np.unique(border).tolist() if l != 0]
    if not bg_labels:
        return keep_all, "backdrop does not reach the border; product fills the frame"

    fg = ((~np.isin(labels, bg_labels)) * 255).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg = solidify(fg)

    # Drop specks (shadows, stray pixels).
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    cleaned = np.zeros_like(fg)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_frac * h * w:
            cleaned[labels == i] = 255

    coverage = float(cleaned.mean() / 255)
    if coverage < 0.02:
        return keep_all, f"segmentation collapsed (product would be {coverage:.1%}); kept whole"

    # The decisive safety check: a real backdrop is smooth. If the region we are about to
    # delete is dense with edges, those edges are product detail and text - not backdrop.
    structure = edge_density(rgb, cleaned == 0)
    if structure > max_erase_structure:
        return keep_all, (
            f"would have erased detailed content (edge density {structure:.3f} > "
            f"{max_erase_structure}); kept whole rather than risk destroying the label"
        )

    return cleaned, f"backdrop removed (product {coverage:.1%}, erased region smooth)"


def foreground_mask(rgb: np.ndarray, min_area_frac: float = 0.002) -> np.ndarray:
    # Deskew only needs a rough silhouette to fit a rotated rectangle to, so run the CHEAP
    # heuristic here, never the heavy model - a deskew must not cost a model inference.
    return _heuristic_segment(rgb, min_area_frac=min_area_frac)[0]


def border_colour(rgb: np.ndarray) -> tuple[int, int, int]:
    """Median corner colour - i.e. whatever this shot uses as its backdrop."""
    return tuple(int(v) for v in np.median(corner_pixels(rgb), axis=0))


def merge_overlapping(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Fold overlapping (x, y, w, h) boxes into one.

    Glare or a graphic can slice a single pack's mask into pieces that then look like
    separate products. Those pieces overlap each other, whereas the packs of a genuine
    multipack sit side by side with a gap. So: if two boxes intersect, they are one
    product.
    """
    remaining = [list(b) for b in boxes]
    merged: list[list[int]] = []

    while remaining:
        x, y, w, h = remaining.pop()
        grew = True
        while grew:
            grew = False
            for other in list(remaining):
                ox, oy, ow, oh = other
                if x < ox + ow and ox < x + w and y < oy + oh and oy < y + h:
                    x2, y2 = max(x + w, ox + ow), max(y + h, oy + oh)
                    x, y = min(x, ox), min(y, oy)
                    w, h = x2 - x, y2 - y
                    remaining.remove(other)
                    grew = True
        merged.append([x, y, w, h])

    return [tuple(b) for b in merged]


def _rotate(rgb: np.ndarray, angle: float) -> np.ndarray:
    """Rotate about the centre, expanding the canvas and padding with the backdrop colour.

    Padding with white would be wrong on a coloured backdrop: the fresh white corners
    become the image border, and the background step downstream would then learn "white"
    as the backdrop and refuse to discard the real one.
    """
    h, w = rgb.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(
        rgb, m, (nw, nh),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_colour(rgb),
    )


# ---------------------------------------------------------------- steps


def step_load(path: str) -> StepResult:
    """Format check: is this a readable image, and what are we holding?"""
    pil = Image.open(path)
    fmt, mode, (w, h) = pil.format, pil.mode, pil.size
    transparent = pil.mode in ("RGBA", "LA") or (pil.mode == "P" and "transparency" in pil.info)
    rgb = load_rgb(path)

    return StepResult(
        name="Format Check",
        image=rgb,
        info={
            "file": os.path.basename(path),
            "format": fmt,
            "mode": mode,
            "size": f"{w} x {h}",
            "megapixels": round(w * h / 1e6, 2),
            "file_size_kb": round(os.path.getsize(path) / 1024, 1),
            "transparency": transparent,
        },
        note=(
            f"Loaded {fmt} {w}x{h} ({mode})."
            + (" Alpha channel composited onto white." if transparent else "")
        ),
    )


def step_orientation(path: str, rgb: np.ndarray) -> StepResult:
    """Apply the EXIF orientation tag so a sideways-shot photo stands upright."""
    pil = Image.open(path)
    exif = pil.getexif()
    tag = exif.get(274, 1) if exif else 1  # 274 = Orientation
    changed = tag not in (0, 1)

    if not changed:
        return StepResult(
            name="Orientation Fix",
            image=rgb,
            info={"exif_orientation_tag": tag, "applied": False},
            note="EXIF orientation 1 (or absent): already upright, nothing to rotate.",
        )

    fixed = np.array(ImageOps.exif_transpose(pil).convert("RGB"))
    return StepResult(
        name="Orientation Fix",
        image=fixed,
        info={"exif_orientation_tag": tag, "applied": True},
        changed=True,
        note=f"EXIF orientation {tag} -> image transposed upright.",
    )


def step_deskew(rgb: np.ndarray, max_angle: float = 15.0) -> StepResult:
    """Straighten small tilts by fitting a min-area rectangle to the product silhouette."""
    mask = foreground_mask(rgb)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return StepResult("Deskew", rgb, info={"angle": 0.0}, note="No product found; skipped.")

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)

    # A rectangle's angle is ambiguous modulo 90 (a 90deg turn is the same box), so
    # fold it into [-45, 45]: that is the true tilt away from level.
    angle = rect[2] % 90
    if angle > 45:
        angle -= 90

    debug = rgb.copy()
    cv2.drawContours(debug, [cv2.boxPoints(rect).astype(int)], 0, (255, 0, 0), 3)

    apply = abs(angle) > 0.5 and abs(angle) <= max_angle
    out = _rotate(rgb, angle) if apply else rgb

    return StepResult(
        name="Deskew",
        image=out,
        debug=debug,
        info={"detected_angle": round(float(angle), 2), "applied": apply, "max_angle": max_angle},
        changed=apply,
        note=(
            f"Product tilted {angle:.2f}deg -> rotated back to level."
            if apply
            else f"Detected {angle:.2f}deg: within 0.5deg deadzone (or beyond {max_angle}deg guard); left as-is."
        ),
    )


def step_normalise(rgb: np.ndarray, clip: float = 2.0) -> StepResult:
    """CLAHE on the L channel: lifts dim/uneven lighting so faint print survives OCR."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)

    return StepResult(
        name="Normalisation",
        image=out,
        info={
            "clip_limit": clip,
            "contrast_before": round(float(l.std()), 1),
            "contrast_after": round(float(l2.std()), 1),
        },
        changed=True,
        note=f"CLAHE raised local contrast (L std {l.std():.1f} -> {l2.std():.1f}).",
    )


def step_resolution(rgb: np.ndarray, min_side: int = 900, max_side: int = 2200) -> StepResult:
    """Pull resolution into a band: upscale tiny images so small print is legible,
    downscale huge ones so OCR stays fast."""
    h, w = rgb.shape[:2]
    short, long = min(h, w), max(h, w)

    scale = 1.0
    if short < min_side:
        scale = min_side / short
    elif long > max_side:
        scale = max_side / long

    if abs(scale - 1.0) < 0.01:
        return StepResult(
            "Resolution Band", rgb,
            info={"size": f"{w} x {h}", "scale": 1.0, "band": f"{min_side}-{max_side}px"},
            note=f"{w}x{h} already inside the {min_side}-{max_side}px band.",
        )

    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    out = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=interp)

    return StepResult(
        name="Resolution Band",
        image=out,
        info={
            "from": f"{w} x {h}",
            "to": f"{out.shape[1]} x {out.shape[0]}",
            "scale": round(scale, 3),
            "band": f"{min_side}-{max_side}px",
        },
        changed=True,
        note=f"{'Upscaled' if scale > 1 else 'Downscaled'} {scale:.2f}x -> {out.shape[1]}x{out.shape[0]}.",
    )


def step_background(
    rgb: np.ndarray, mask: np.ndarray, reason: str, pad: int = 12
) -> tuple[StepResult, np.ndarray, np.ndarray]:
    """Crop to the product (crop-only): trim to the mask's bounding box with a little pad.

    Deliberately does NOT white-out interior pixels. Earlier versions flattened the
    background to white, which damaged glossy / white-on-white packs whose mask was slightly
    off - it overwrote real print - and the Stage-1 benchmark showed white-out did not improve
    field recall anyway. Cropping is strictly safe: a loose crop keeps a little backdrop, which
    costs OCR nothing, whereas an erased label is fatal.

    Takes the mask rather than deriving its own, and returns the mask cropped the same way so
    the grouping step can reuse it.
    """
    h, w = rgb.shape[:2]
    coverage = float(mask.mean() / 255)

    debug = rgb.copy()
    debug[mask == 0] = (debug[mask == 0] * 0.25).astype(np.uint8)  # dim what we'd crop away

    if coverage <= 0 or coverage > 0.98:
        return (
            StepResult(
                "Crop to Product", rgb, debug=debug,
                info={
                    "verdict": reason,
                    "foreground_coverage": round(coverage, 3),
                    "applied": False,
                },
                note=f"Left untouched — {reason}.",
            ),
            rgb,
            mask,
        )

    ys, xs = np.where(mask > 0)
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad)
    out = rgb[y0:y1, x0:x1]

    return (
        StepResult(
            name="Crop to Product",
            image=out,
            debug=debug,
            info={
                "verdict": reason,
                "foreground_coverage": round(coverage, 3),
                "crop_box": [int(x0), int(y0), int(x1), int(y1)],
                "from": f"{w} x {h}",
                "to": f"{out.shape[1]} x {out.shape[0]}",
            },
            changed=True,
            note=(
                f"Cropped to the product ({out.shape[1]}x{out.shape[0]}) — background kept, "
                "not whited out."
            ),
        ),
        out,
        mask[y0:y1, x0:x1],
    )


def step_group(
    rgb: np.ndarray, mask: np.ndarray, min_area_frac: float = 0.005, pad: int = 8
) -> tuple[StepResult, list[np.ndarray]]:
    """Multi-image grouping: catalog shots often show the same pack 2-3 times.
    Split them so OCR reads one product face at a time instead of a collage.

    `min_area_frac` used to default to 0.04 (4% of the image). That is fine for telling
    a repeated product copy apart from noise, but a single back-panel photo routinely lays
    out several DIFFERENT-sized info blocks on its own printed background (nutrition table,
    ingredients, barcode, FSSAI licence, flavour callouts) - and at 4%, a barcode (~3.3% on
    a real example) and an FSSAI licence number (~2.2%) both fell under the bar and were
    silently dropped from OCR entirely, never recovered by any engine downstream. Lowering
    this close to `segment()`'s own despeckle floor (0.002) means real content blocks - even
    small ones - survive; the only images affected are the (rare) ones with an isolated
    scrap this small, which becomes a harmless extra crop rather than lost data.
    """
    h, w = rgb.shape[:2]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    boxes = [
        (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        for i in range(1, n)
        if stats[i, cv2.CC_STAT_AREA] >= min_area_frac * h * w
    ]
    fragments = len(boxes)
    boxes = merge_overlapping(boxes)
    boxes.sort(key=lambda b: b[0])  # left to right

    debug = rgb.copy()
    for idx, (x, y, bw, bh) in enumerate(boxes, 1):
        cv2.rectangle(debug, (x, y), (x + bw, y + bh), (255, 0, 0), 3)
        cv2.putText(debug, str(idx), (x + 8, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    crops = [
        rgb[max(0, y - pad):min(h, y + bh + pad), max(0, x - pad):min(w, x + bw + pad)]
        for (x, y, bw, bh) in boxes
    ]

    multi = len(crops) > 1
    return StepResult(
        name="Multi-Image Grouping",
        image=rgb,
        debug=debug,
        info={
            "instances_found": len(crops),
            "mask_fragments": fragments,
            "merged_into": len(boxes),
            "boxes": [list(map(int, b)) for b in boxes],
        },
        changed=multi,
        note=(
            f"{len(crops)} product instances detected -> split into {len(crops)} crops for OCR."
            if multi
            else f"{len(crops) or 'No'} single instance; no split needed."
        )
        + (
            f" ({fragments} mask fragments merged down to {len(boxes)} by overlap.)"
            if fragments > len(boxes)
            else ""
        ),
    ), crops


# ---------------------------------------------------------------- pipeline


def run_pipeline(path: str, cfg: dict | None = None) -> tuple[list[StepResult], list[np.ndarray]]:
    """Run every enabled step in architecture order.

    Returns the per-step results (for display) and the final crop(s) to hand to OCR.
    """
    cfg = cfg or {}
    steps: list[StepResult] = []

    s = step_load(path)
    steps.append(s)
    img = s.image

    if cfg.get("orientation", True):
        s = step_orientation(path, img)
        steps.append(s)
        img = s.image

    if cfg.get("deskew", True):
        s = step_deskew(img, max_angle=cfg.get("max_angle", 15.0))
        steps.append(s)
        img = s.image

    if cfg.get("normalise", True):
        s = step_normalise(img, clip=cfg.get("clip_limit", 2.0))
        steps.append(s)
        img = s.image

    if cfg.get("resolution", True):
        s = step_resolution(img, cfg.get("min_side", 900), cfg.get("max_side", 2200))
        steps.append(s)
        img = s.image

    # Segment ONCE here, then hand the mask to both background and grouping. Letting
    # grouping re-derive it after the crop would be wrong - the cropped image's corners
    # are product, so the backdrop model would learn the product and erase it.
    mask, reason = segment(img)

    if cfg.get("background", True):
        s, img, mask = step_background(img, mask, reason)
        steps.append(s)

    crops = [img]
    if cfg.get("group", True):
        s, found = step_group(img, mask, min_area_frac=cfg.get("min_area_frac", 0.005))
        steps.append(s)
        if found:
            crops = found

    return steps, crops
