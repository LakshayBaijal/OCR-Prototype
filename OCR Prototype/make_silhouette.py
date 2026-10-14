"""Cut a transparent-background silhouette PNG from each image in a folder.

Own model choice - birefnet-general, NOT bgremove.py's u2netp. Measured
directly on real shelf photos (a hand grabbing one item off a rack of several
near-identical duplicates - three Whiskas boxes, a wall of Twinkies boxes):
u2netp either grabbed the wrong duplicate, merged several boxes into one mask,
or kept only a fragment of the product. birefnet-general got 5 of 6 such cases
right outright. bgremove.py stays on u2netp on purpose - it feeds
preprocessing.step_background, which only ever reads a mask's BOUNDING BOX
(never its per-pixel shape), where the two models measured within 1-3px of
each other - so switching that shared module would cost real speed for zero
benefit there. This script needs the actual per-pixel shape (it writes the
mask as alpha), so it earns the ~10s/image the stronger model costs.

CENTER-COMPONENT SELECTION, not "keep everything": on a shelf full of
near-duplicate products the mask often contains several similar-sized blobs
(one per product/price-tag/decoration), and "biggest" is not "the one being
held". The one closest to the frame's own center reliably is - that's
compositionally where a product gets held up to camera. Components under 1% of
the image area are dropped as noise before this comparison.

WHAT THIS DOES NOT FIX: two products touching with no gap between them mask as
ONE connected blob no matter which model reads it - there is no boundary for
component analysis to find. Measured on one such case (three Whiskas boxes
flush against each other): the only fix was a rough manual crop to isolate the
one product before segmentation, done ad hoc outside this script - it is not
something this script can detect or do automatically.

Unlike `preprocessing.step_background` (crop-only, keeps the backdrop pixels
on purpose), this WRITES the mask as the alpha channel - the output is a real
transparent PNG. That's the point here: a downstream OCR step gets a plain
white product shot with `_flatten()` (see stage1_ocr_app.py / batch_ocr.py),
not a busy background to fight through.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

MODEL = "birefnet-general"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
PAD = 12

_session = None


def _get_session():
    global _session
    if _session is None:
        from rembg import new_session
        _session = new_session(MODEL, providers=["CPUExecutionProvider"])
    return _session


def _sort_key(p: Path):
    digits = "".join(c for c in p.stem if c.isdigit())
    return (int(digits) if digits else 0, p.stem)


def _raw_mask(rgb: np.ndarray) -> np.ndarray:
    from rembg import remove

    out = remove(rgb, session=_get_session(), only_mask=True, post_process_mask=True)
    alpha = np.asarray(out)
    if alpha.ndim == 3:
        alpha = alpha[..., 0]
    return alpha.astype(np.uint8)


def _pick_center_component(mask: np.ndarray, thresh: int = 128) -> np.ndarray:
    """Keep only the connected component nearest the image's own center, among
    components with a non-trivial area (>1% of the image) - see module
    docstring for why "nearest center" beats "biggest" here."""
    binary = mask >= thresh
    labeled, n = ndimage.label(binary)
    if n <= 1:
        return (binary * 255).astype(np.uint8)

    h, w = mask.shape
    cy, cx = h / 2, w / 2
    min_area = 0.01 * h * w

    best_label, best_dist = None, None
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < min_area:
            continue
        dist = ((ys.mean() - cy) ** 2 + (xs.mean() - cx) ** 2) ** 0.5
        if best_dist is None or dist < best_dist:
            best_label, best_dist = i, dist

    if best_label is None:
        return (binary * 255).astype(np.uint8)
    return ((labeled == best_label) * 255).astype(np.uint8)


def cut_one(path: Path) -> Image.Image | None:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    fg_mask = _pick_center_component(_raw_mask(rgb))

    ys, xs = np.where(fg_mask > 0)
    if len(ys) == 0:
        return None  # nothing detected as foreground - skip rather than guess

    h, w = rgb.shape[:2]
    y0, y1 = max(0, ys.min() - PAD), min(h, ys.max() + PAD)
    x0, x1 = max(0, xs.min() - PAD), min(w, xs.max() + PAD)

    rgba = np.dstack([rgb, fg_mask])[y0:y1, x0:x1]
    return Image.fromarray(rgba, mode="RGBA")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: make_silhouette.py <in_dir> <out_dir>")
    in_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        (p for p in in_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
        key=_sort_key,
    )
    print(f"{len(images)} image(s) in {in_dir}  (model: {MODEL})")
    for i, path in enumerate(images, 1):
        out_path = out_dir / (path.stem + ".png")
        if out_path.exists():
            print(f"  [{i}/{len(images)}] {path.name} -> skipped (already done)")
            continue
        try:
            cut = cut_one(path)
            if cut is None:
                print(f"  [{i}/{len(images)}] {path.name} -> SKIP (nothing detected)")
                continue
            cut.save(out_path)
            print(f"  [{i}/{len(images)}] {path.name} -> {out_path.name} ({cut.size[0]}x{cut.size[1]})")
        except Exception as exc:
            print(f"  [{i}/{len(images)}] {path.name} -> ERROR {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
