"""
Product Box Finder - point it at a photo, get a box around the product.

Run:  venv/bin/streamlit run detect_app.py

Deliberately does one thing: find the product in a photo and show you its bounding box.
No OCR, no verification, no comparison - see verify_app.py for that.

NOT built on YOLO, and that was a measured decision rather than a preference. A
COCO-pretrained detector (torchvision Faster R-CNN) was tried on this dataset first and
fails on packaged grocery, because COCO has no class for a box, pouch or carton: a Coco
Pops carton came back as 'bowl' + 'teddy bear' (it detected the artwork), a rice pack
came back as 'bed', and a held Heineken returned SIX 'bottle' detections with no way to
tell the held one from the five on the shelf behind it. Any COCO-class model inherits
that - it is the label set, not the architecture.

What works instead is salient-object segmentation: "which pixels are the SUBJECT of
this photograph", with no notion of category. On a photo of a held product that is
exactly the product you want, and the shelf behind it is exactly what it rejects.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import warnings

warnings.filterwarnings("ignore")

import streamlit as st

from verify import detect, extract

st.set_page_config(page_title="Product Box Finder", layout="wide")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _find_default_dir() -> Path:
    """Same sibling-checkout search as verify_app.py's `_find_image_enhancement_root` -
    the ondc-intelligence checkout's directory name isn't stable across branches/
    re-clones, so this finds it by content rather than hardcoding one name."""
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


DEFAULT_DIR = _find_default_dir()

st.title("Product Box Finder")
st.caption(
    "Finds the product in a photo and boxes it. Salient-object segmentation, not YOLO — "
    "COCO has no class for a carton or pouch, so a COCO detector reads a cereal box as "
    "'bowl' and a rice pack as 'bed'. This asks *what is the subject of the photo* "
    "instead, which is category-agnostic."
)

# ---------------------------------------------------------------- sidebar

model_label = st.sidebar.radio(
    "Segmentation model",
    ["Fast (u2netp, ~0.2s)", "Accurate (BiRefNet, ~7s)"],
    help="Measured on a held-bottle shelf photo: u2netp boxed 41% of the frame at 59% "
         "fill, BiRefNet 33% at 74% — tighter and cleaner, ~30x slower on CPU.",
)
model = detect.FAST_MODEL if model_label.startswith("Fast") else detect.ACCURATE_MODEL

exclude_distractors = st.sidebar.checkbox(
    "Exclude hands / animals", value=True,
    help="Mask R-CNN subtracts a holding hand or a real animal in frame (COCO's "
         "`person` + live-animal classes, 0.96-0.99 confidence on this data). It does "
         "NOT trust COCO's furniture/container classes (bed, dining table, bowl, "
         "bottle...) — this pipeline already caught COCO calling an actual rice pack "
         "'bed' and a cereal box 'bowl', so treating those as safe-to-erase background "
         "would risk deleting the real product instead of a distractor. Measured on a "
         "held tuna carton: box 33% -> 13% of frame, no longer running off the frame "
         "edge where the arm left shot. Adds ~0.7s; first use downloads ~170MB.",
)
pad_pct = st.sidebar.slider(
    "Grow box outward (% per side)", 0, 15, 4, 1,
    help="The matte under-covers pack edges — a promo panel in a different colour plane, "
         "a corner in shadow. 4% covers those; 0 gives the tightest box the mask supports.",
) / 100.0

show_secondary = st.sidebar.checkbox(
    "Show secondary boxes", value=True,
    help="Every salient region found, not just the biggest. Off = only the main product.",
)
show_mask = st.sidebar.checkbox(
    "Show the silhouette", value=False,
    help="Tints the pixels the box was derived FROM — the thing to look at when a box "
         "looks wrong.",
)
min_area = st.sidebar.slider(
    "Ignore regions smaller than (% of frame)", 0.1, 10.0, 1.0, 0.1,
) / 100.0

# ---------------------------------------------------------------- input

source = st.radio("Image source", ["Browse the dataset", "Upload"], horizontal=True)

rgb = None
if source == "Upload":
    up = st.file_uploader("Product photo", type=[e[1:] for e in IMAGE_EXTS])
    if up:
        rgb = extract.load_rgb_bytes(up.getvalue())
else:
    if not DEFAULT_DIR.is_dir():
        st.error(f"Dataset folder not found: `{DEFAULT_DIR}`")
        st.stop()
    files = sorted(p for p in DEFAULT_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        st.error(f"No images in `{DEFAULT_DIR}`")
        st.stop()
    pick = st.selectbox(f"Image ({len(files)} available)", [p.name for p in files])
    rgb = extract.load_rgb_bytes((DEFAULT_DIR / pick).read_bytes())

if rgb is None:
    st.info("Pick or upload a photo to find the product in it.")
    st.stop()

# ---------------------------------------------------------------- detect

with st.spinner(f"Finding the product ({model})…"):
    boxes, mask = detect.detect(rgb, model=model, exclude_distractors=exclude_distractors)
    if mask is not None:
        boxes = detect.boxes_from_mask(mask, min_area_frac=min_area, pad_frac=pad_pct)

if mask is None:
    st.error("Segmentation unavailable — is `rembg` installed?")
    st.stop()
if not boxes:
    st.warning("No region large enough to be a product. Try lowering the minimum size.")
    st.stop()

canvas = detect.overlay_mask(rgb, mask) if show_mask else rgb
st.image(detect.draw(canvas, boxes, show_secondary=show_secondary),
         caption=f"{len(boxes)} region(s) · {rgb.shape[1]}×{rgb.shape[0]} · {model}",
         use_container_width=True)

# ---------------------------------------------------------------- the boxes

st.subheader("Boxes")
st.caption(
    "**fill** is how much of the box is actually product rather than background caught "
    "inside it — a tall narrow subject (a held bottle) boxes loosely because the box is "
    "a rectangle and the bottle is not."
)

rows = []
for i, b in enumerate(boxes):
    if not b.is_primary and not show_secondary:
        continue
    rows.append(
        f"| {'**PRODUCT**' if b.is_primary else f'#{i + 1}'} | "
        f"`{b.x}, {b.y}` | `{b.w} × {b.h}` | {b.area_frac:.0%} | {b.fill:.0%} |"
    )
st.markdown(
    "| Region | x, y | w × h | % of frame | fill |\n|---|---|---|---|---|\n" + "\n".join(rows)
)

st.subheader("Cropped out")
cols = st.columns(min(3, len([b for b in boxes if b.is_primary or show_secondary])))
shown = 0
for i, b in enumerate(boxes):
    if not b.is_primary and not show_secondary:
        continue
    with cols[shown % len(cols)]:
        st.image(b.crop(rgb, pad_frac=0.0),
                 caption=("PRODUCT" if b.is_primary else f"#{i + 1}") + f" — {b.w}×{b.h}",
                 use_container_width=True)
    shown += 1

with st.expander("Why not YOLO?"):
    st.markdown(
        "torchvision's COCO Faster R-CNN was tested on these exact images first:\n\n"
        "| image | what COCO detected |\n|---|---|\n"
        "| Heineken held at a shelf | `person` (the hand) + **six** `bottle` — no way to "
        "tell the held one from the shelf |\n"
        "| Coco Pops carton | `bowl` and `teddy bear` — the artwork, not the box |\n"
        "| Rice pack on a table | `bed` |\n\n"
        "COCO has no class for a retail package, and swapping in YOLOv8 would inherit the "
        "same label set. Detecting *a product* as a class needs a model fine-tuned on "
        "retail imagery — a labelling project, not a library swap. Salient-object "
        "segmentation sidesteps the whole issue by not classifying at all."
    )
