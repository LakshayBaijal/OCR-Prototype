"""Box every product in image_enhancement_input/, save the annotated result.

Pure CV, same pipeline `detect_app.py` lets you drive interactively: salient-object
segmentation (u2netp) minus the holding hand (Mask R-CNN's `person` class), boxes grown
4% outward per side for full coverage. No Gemini, no VLM, no network call - see
verify/detect.py's module docstring for why a COCO-class detector (YOLO included)
doesn't work on packaged grocery in the first place.

Usage:
    venv/bin/python detect_batch.py
    venv/bin/python detect_batch.py --input-dir /path/to/image_enhancement_input
    venv/bin/python detect_batch.py --accurate         # BiRefNet instead of u2netp
    venv/bin/python detect_batch.py --overwrite

For each input image, writes to image_enhancement_processed/ (a sibling of
image_enhancement_input/ unless --output-dir overrides it):
    <stem>.<ext>    the product CROPPED OUT - just those pixels, nothing drawn on top.
                    This is the file meant to go to Gemini or anywhere else downstream:
                    it should see the product alone, not a green box and a "PRODUCT 45%"
                    label burned into a shelf photo.
    <stem>.json     the crop's source coordinates/metrics against the ORIGINAL frame,
                    for whatever reads this next - same "write a sidecar next to the
                    image" convention image_enhancement.py already uses for its own
                    outputs

AND to image_enhancement_processed_silhouette/ (a sibling of image_enhancement_input/
unless --silhouette-dir overrides it):
    <stem>.png      a CUTOUT, not a flat shape - the real product pixels (actual
                    colours, labels, every detail) cropped to the identical region as
                    the file above, with the background made fully TRANSPARENT rather
                    than blacked out or left in. Opens as "the bottle floating on
                    nothing" in anything that reads PNG alpha; composite it onto white
                    or black yourself if a viewer flattens transparency. An earlier
                    version of this file was a flat white-on-black SHAPE with no image
                    detail at all - if that silhouette style is what's wanted instead,
                    see `_shape_mask` below, no longer called by default.
                    PNG, not JPEG: JPEG has no alpha channel at all.
                    Written from the SAME detection pass as the crop above - no second
                    model run, no extra cost beyond one more image write.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from verify import detect, extract

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _cutout(cropped_rgb: np.ndarray, full_mask: np.ndarray, box_xyxy: tuple) -> np.ndarray:
    """The real product pixels with the background made transparent - a BGRA array
    ready for `cv2.imwrite(..., png)`. Takes RGB in (what `Box.crop` returns, and what
    every other array in this loop is - the BGR conversion elsewhere in this file
    happens inline, at the imwrite call, never by reassigning the array) and converts
    to BGR itself, so a future caller passing the same `cropped` variable used for the
    JPEG write does not have to remember which colour order it is secretly in.

    Takes the box's own xyxy rather than re-deriving one, so this crops the mask by
    the EXACT same coordinates `Box.crop` used for the colour pixels - if the two ever
    disagreed by even a pixel, the alpha channel would not line up with the image it is
    supposed to be masking.
    """
    x0, y0, x1, y1 = box_xyxy
    alpha = (full_mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
    b, g, r = cv2.split(cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR))
    return cv2.merge([b, g, r, alpha])


def _shape_mask(full_mask: np.ndarray, box_xyxy: tuple) -> np.ndarray:
    """The flat white-shape-on-black rendering this script used before `_cutout`
    replaced it as the default - kept for anyone who specifically wants the pure
    silhouette shape rather than a cutout with real image detail. Not called anywhere
    in this file currently; wire it back in if that style is needed again."""
    x0, y0, x1, y1 = box_xyxy
    return (full_mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255


def _find_input_dir() -> Path:
    """Same sibling-checkout search as verify_app.py's `_find_image_enhancement_root` -
    the ondc-intelligence checkout's directory name isn't stable across branches/
    re-clones (it has already changed once in this project), so this finds it by
    content rather than hardcoding one name."""
    root = Path(__file__).resolve().parent.parent
    exact = root / "ondc-intelligence" / "image_enhancement" / "image_enhancement_input"
    if exact.is_dir():
        return exact
    for sibling in sorted(p for p in root.iterdir() if p.is_dir()):
        if sibling.name.startswith("ondc-intelligence"):
            candidate = sibling / "image_enhancement" / "image_enhancement_input"
            if candidate.is_dir():
                return candidate
    return exact


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=None,
                    help="default: auto-found image_enhancement_input/ (see _find_input_dir)")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="default: image_enhancement_processed/, sibling of --input-dir")
    ap.add_argument("--silhouette-dir", type=Path, default=None,
                    help="default: image_enhancement_processed_silhouette/, sibling of "
                         "--input-dir")
    ap.add_argument("--no-silhouette", action="store_true",
                    help="skip writing the silhouette (binary mask crop) output")
    ap.add_argument("--accurate", action="store_true",
                    help="BiRefNet instead of u2netp - tighter boxes, ~30x slower on CPU, "
                         "and the model measured to occasionally latch onto the hand "
                         "instead of the product (see detect.py's _MIN_SURVIVING_MASK "
                         "docstring) - u2netp is the validated default for this dataset")
    ap.add_argument("--no-exclude-distractors", action="store_true",
                    help="skip hand/animal removal (faster, boxes will include them). "
                         "Only ever excludes a holding hand or a real live animal - "
                         "COCO's furniture/container classes are deliberately never "
                         "trusted here, see detect.py's _DISTRACTOR_CATEGORIES for why")
    ap.add_argument("--pad", type=float, default=detect._BOX_PAD_FRAC,
                    help=f"outward padding per side, fraction of box size "
                         f"(default {detect._BOX_PAD_FRAC})")
    ap.add_argument("--overwrite", action="store_true",
                    help="reprocess images that already have output")
    ap.add_argument("--n", type=int, default=0, help="cap the number of images (0 = all)")
    args = ap.parse_args()

    input_dir = args.input_dir or _find_input_dir()
    if not input_dir.is_dir():
        raise SystemExit(f"input dir not found: {input_dir}")
    output_dir = args.output_dir or (input_dir.parent / "image_enhancement_processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    silhouette_dir = None
    if not args.no_silhouette:
        silhouette_dir = args.silhouette_dir or (input_dir.parent / "image_enhancement_processed_silhouette")
        silhouette_dir.mkdir(parents=True, exist_ok=True)

    model = detect.ACCURATE_MODEL if args.accurate else detect.FAST_MODEL
    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if args.n:
        images = images[: args.n]

    print(f"\n{'=' * 78}")
    print(f"PRODUCT BOX DETECTION - {len(images)} images  ·  model={model}  ·  "
          f"exclude_distractors={not args.no_exclude_distractors}  ·  pad={args.pad:.0%}")
    print(f"input:  {input_dir}")
    print(f"output: {output_dir}")
    print(f"{'=' * 78}\n")

    ok = no_box = failed = 0
    started_all = time.perf_counter()

    for path in images:
        out_img_path = output_dir / path.name
        out_json_path = output_dir / f"{path.stem}.json"
        out_sil_path = silhouette_dir / f"{path.stem}.png" if silhouette_dir else None
        already_done = out_img_path.exists() and (out_sil_path is None or out_sil_path.exists())
        if already_done and not args.overwrite:
            print(f"{path.name:24} cached")
            ok += 1
            continue

        t0 = time.perf_counter()
        try:
            rgb = extract.load_rgb_bytes(path.read_bytes())
            boxes, mask = detect.detect(rgb, model=model,
                                        exclude_distractors=not args.no_exclude_distractors)
            if mask is not None and args.pad != detect._BOX_PAD_FRAC:
                boxes = detect.boxes_from_mask(mask, pad_frac=args.pad)
        except Exception as exc:
            failed += 1
            print(f"{path.name:24} ERROR  {type(exc).__name__}: {exc}")
            continue

        elapsed = time.perf_counter() - t0
        if not boxes:
            no_box += 1
            print(f"{path.name:24} no product region found ({elapsed:.1f}s)")
            continue

        # The CROP itself, not the full frame with a box drawn on it - this is what's
        # meant to be handed to something downstream (Gemini or otherwise) that should
        # see only the product, not a green rectangle and "PRODUCT 45%" burned into a
        # shelf photo. `pad_frac=0.0` here is deliberate, not a missing padding: the box
        # already grew outward by `--pad` in boxes_from_mask, so padding again would
        # double it.
        primary = next(b for b in boxes if b.is_primary)
        cropped = primary.crop(rgb, pad_frac=0.0)
        cv2.imwrite(str(out_img_path), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR),
                   [cv2.IMWRITE_JPEG_QUALITY, 95])

        if out_sil_path is not None and mask is not None:
            cv2.imwrite(str(out_sil_path), _cutout(cropped, mask, primary.xyxy))

        out_json_path.write_text(json.dumps({
            "source_image": path.name,
            "image_size": {"w": int(rgb.shape[1]), "h": int(rgb.shape[0])},
            "model": model,
            "exclude_distractors": not args.no_exclude_distractors,
            "pad_frac": args.pad,
            "product_box": {
                "x": primary.x, "y": primary.y, "w": primary.w, "h": primary.h,
                "area_frac": round(primary.area_frac, 4),
                "fill": round(primary.fill, 4),
            },
            "secondary_boxes": [
                {"x": b.x, "y": b.y, "w": b.w, "h": b.h,
                 "area_frac": round(b.area_frac, 4), "fill": round(b.fill, 4)}
                for b in boxes if not b.is_primary
            ],
            "elapsed_s": round(elapsed, 3),
        }, indent=2), encoding="utf-8")

        ok += 1
        print(f"{path.name:24} OK  box=({primary.x},{primary.y},{primary.w},{primary.h})  "
              f"area={primary.area_frac:.0%}  fill={primary.fill:.0%}  ({elapsed:.1f}s)")

    total = time.perf_counter() - started_all
    print(f"\n{'-' * 78}")
    print(f"  done: {ok} written, {no_box} with no region found, {failed} errored  "
          f"({total:.1f}s total)")
    print(f"  crops:       {output_dir}")
    if silhouette_dir:
        print(f"  silhouettes: {silhouette_dir}")
    print()


if __name__ == "__main__":
    main()
