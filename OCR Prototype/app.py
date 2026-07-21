"""
Stage 0 - Preprocessing visualiser (product-level).

Run:  .venv/bin/streamlit run app.py

Pick any image from the Dataset and the app automatically pulls in every OTHER image of
the same product - front, back, side panels - because they belong together: the barcode
token in the filename (`Image_<view>_<barcode>.0.jpg`) is shared across a product's shots,
and that is exactly what the ground-truth CSV groups on. You never have to add the
associated images by hand.

Everything downstream is then scored per PRODUCT, not per image, which is the only honest
way: the FSSAI number lives on one face, the MRP on another, so a single shot can never
hold them all. Stage 1 OCR text, the barcode, and field recall are all unioned across the
whole product.
"""

from __future__ import annotations

import os

# Must be set before pyarrow's C++ library loads (Arrow reads this exactly once, at first
# load) - so this has to come before `import pandas`, which pulls pyarrow in transitively.
# Without it, this machine segfaults (SIGSEGV in libarrow's mi_thread_init/mi_heap_main -
# confirmed via 3 identical macOS crash reports, same address, same thread, same stack)
# on Streamlit's own pandas -> pyarrow DataFrame serialization, on Python 3.14 + pyarrow
# 25.0.0. This forces Arrow to use the plain system allocator instead of its bundled
# mimalloc, sidestepping the crashing code path entirely. Known, documented Arrow env var:
# https://arrow.apache.org/docs/cpp/env_vars.html
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import barcode
import memguard
import products
from evaluate_ocr import found
from ocr import router
from ocr.types import OCRResult
from preprocessing import run_pipeline

ROOT = Path(__file__).parent
DATASET = ROOT / "Dataset"
GROUND_TRUTH = ROOT / "Ground Truth.csv"


def draw_boxes(rgb: np.ndarray, result: OCRResult) -> np.ndarray:
    """Overlay each detected line, coloured by confidence: green = trusted, red = shaky.
    This is how you SEE what the router is reacting to when it escalates."""
    vis = rgb.copy()
    for line in result.lines:
        c = line.confidence
        colour = (0, 170, 0) if c is None or c >= 0.9 else (255, 165, 0) if c >= 0.7 else (220, 0, 0)
        cv2.polylines(vis, [np.array(line.bbox, np.int32)], True, colour, 2)
    return vis


def barcode_source_image(steps: list) -> np.ndarray:
    """The image the barcode reader should see: oriented/deskewed/normalised but BEFORE
    background removal, so the bar detector keeps the surrounding contrast it localises on.
    Mirrors the reasoning in the module docstring of barcode.py."""
    names = [s.name for s in steps]
    bg_idx = names.index("Crop to Product") if "Crop to Product" in names else None
    return steps[bg_idx - 1].image if bg_idx else steps[0].image


def render_steps(steps: list) -> None:
    """Render the full step-by-step Stage 0 breakdown for ONE image (before / after / debug
    overlay per step). Called once per image in the product so every shot's preprocessing
    is visible, not just the selected one."""
    for i, step in enumerate(steps):
        badge = "✅ applied" if step.changed else "➖ no change"
        st.divider()
        st.subheader(f"{i}. {step.name}  ·  {badge}")
        st.write(step.note)

        before = steps[i - 1].image if i else None
        cols = st.columns(3 if step.debug is not None else (2 if before is not None else 1))

        c = 0
        if before is not None:
            cols[c].image(before, caption=f"Before — {before.shape[1]}×{before.shape[0]}", width="stretch")
            c += 1
        cols[c].image(step.image, caption=f"After — {step.image.shape[1]}×{step.image.shape[0]}", width="stretch")
        c += 1
        if step.debug is not None:
            cols[c].image(step.debug, caption="What the step detected", width="stretch")

        if step.info:
            st.json(step.info, expanded=False)


GT_FIELDS = [
    "Product Name", "Brand", "Mrp", "Net Quantity", "Product Quantity",
    "Upc / Ean", "Fssai No", "Manufacturer", "Country Of Origin", "Product Type Name",
]
RECALL_FIELDS = ["Brand", "Mrp", "Net Quantity", "Fssai No", "Upc / Ean", "Manufacturer"]

st.set_page_config(page_title="Stage 0 - Preprocessing", layout="wide")


@st.cache_data(show_spinner=False)
def list_images() -> list[str]:
    return sorted(p.name for p in DATASET.glob("*.jpg"))


@st.cache_data(show_spinner=False)
def group_index() -> dict[str, list[str]]:
    """product key -> its image filenames, view-ordered. Built once over the listing."""
    return products.group_index(list_images())


@st.cache_data(show_spinner="Preprocessing (background matte ~10s/image, first time only)…")
def run_pipeline_cached(name: str, cfg_items: tuple):
    """Stage 0 for one image, cached on (filename, config) so switching the inspected image
    or toggling a Stage 1 checkbox does not re-run preprocessing for the whole product. The
    background matte (BiRefNet) is the slow part on a cold cache; it is cached to disk too."""
    return run_pipeline(str(DATASET / name), dict(cfg_items))


@st.cache_data(show_spinner=False)
def ground_truth_index() -> dict[str, dict]:
    """Map every image filename -> its product row, so we can show what OCR should find."""
    if not GROUND_TRUTH.exists():
        return {}
    df = pd.read_csv(GROUND_TRUTH, low_memory=False)
    cols = [c for c in df.columns if "Filename" in c]
    index: dict[str, dict] = {}
    for _, row in df.iterrows():
        record = {f: row.get(f) for f in GT_FIELDS if f in df.columns}
        for c in cols:
            name = row.get(c)
            if isinstance(name, str) and name.strip():
                index.setdefault(name.strip(), record)
    return index


images = list_images()
gt = ground_truth_index()

# ---------------------------------------------------------------- sidebar

st.sidebar.title("Stage 0 - Preprocessing")
st.sidebar.caption(f"{len(images):,} images · {len(group_index()):,} products in Dataset/")

query = st.sidebar.text_input("Filter by filename / barcode", placeholder="e.g. 8901030834691")
matches = [n for n in images if query in n] if query else images

if not matches:
    st.sidebar.error("No image matches that filter.")
    st.stop()

if st.sidebar.button("🎲 Pick a random image", width='stretch'):
    st.session_state.choice = random.choice(matches)

options = matches[:500]
if st.session_state.get("choice") in matches and st.session_state.choice not in options:
    options = [st.session_state.choice] + options

choice = st.sidebar.selectbox(
    f"Image ({len(matches):,} match{'es' if len(matches) != 1 else ''})",
    options,
    index=options.index(st.session_state.choice) if st.session_state.get("choice") in options else 0,
)

# Every image of the same product, auto-included. This is the whole point: you pick one
# shot, the app reads the entire product.
group = products.associated(choice, images)
st.sidebar.divider()
st.sidebar.subheader(f"Product group · {len(group)} images")
st.sidebar.caption("Auto-included from the same barcode — you don't add these by hand.")
for name in group:
    marker = "👉 " if name == choice else "     "
    st.sidebar.write(f"{marker}`{name}`")

st.sidebar.divider()
st.sidebar.subheader("Steps")
cfg = {
    "orientation": st.sidebar.checkbox("Orientation Fix", True),
    "deskew": st.sidebar.checkbox("Deskew", True),
    "normalise": st.sidebar.checkbox("Normalisation", True),
    "resolution": st.sidebar.checkbox("Resolution Band", True),
    "background": st.sidebar.checkbox("Crop to Product", True),
    "group": st.sidebar.checkbox("Multi-Image Grouping", True),
}

st.sidebar.subheader("Parameters")
cfg["max_angle"] = st.sidebar.slider("Deskew max angle (deg)", 1.0, 30.0, 15.0, 0.5)
cfg["clip_limit"] = st.sidebar.slider("CLAHE clip limit", 0.5, 6.0, 2.0, 0.5)
cfg["min_side"], cfg["max_side"] = st.sidebar.slider(
    "Resolution band (px)", 300, 3000, (900, 2200), 50
)
cfg["min_area_frac"] = st.sidebar.slider(
    "Grouping: min instance area (frac of image)", 0.002, 0.30, 0.005, 0.001,
    help="How small a foreground block can be and still get its own OCR crop. Too high "
         "and small-but-real content (a barcode, an FSSAI number) gets silently dropped.",
)

# ---------------------------------------------------------------- run Stage 0 for the group

cfg_key = tuple(sorted(cfg.items()))
group_steps: dict[str, list] = {}
group_crops: dict[str, list[np.ndarray]] = {}
for name in group:
    steps_i, crops_i = run_pipeline_cached(name, cfg_key)
    group_steps[name] = steps_i
    group_crops[name] = crops_i

# The product row is shared by every image in the group; take it from whichever has one.
record = next((gt.get(n) for n in group if gt.get(n)), None)

stage0, stage1 = st.tabs(["Stage 0 · Preprocessing", "Stage 1 · OCR"])

# ================================================================ Stage 0

with stage0:
    st.caption(
        f"Product **`{products.product_key(choice) or choice}`** — "
        f"{len(group)} associated images, auto-grouped and read together."
    )

    if record:
        with st.expander("📋 Ground truth for this product", expanded=False):
            # Every value is stringified: the column mixes text (Product Name) with numbers
            # (Mrp), and Arrow requires a single type per column to serialise the table.
            st.table(
                pd.DataFrame(
                    [(k, "—" if pd.isna(v) else str(v)) for k, v in record.items()],
                    columns=["Field", "Expected value"],
                ).set_index("Field")
            )
    else:
        st.info("No ground-truth row found for this product.")

    st.markdown("**Preprocessing steps — every image in the product**")
    st.caption(
        "One tab per image; each runs the full Stage 0 pipeline. 👉 marks the image you "
        "picked. All of them feed Stage 1."
    )
    img_tabs = st.tabs([
        ("👉 " if name == choice else "") + f"Image {products.view_number(name)}"
        for name in group
    ])
    for tab, name in zip(img_tabs, group):
        with tab:
            st.caption(f"`{name}`")
            render_steps(group_steps[name])

    st.divider()
    total_crops = sum(len(c) for c in group_crops.values())
    st.header(f"➡️ Output to OCR — {total_crops} image{'s' if total_crops != 1 else ''} across the product")
    st.caption("These are the exact pixels Stage 1 (OCR) will receive, grouped by source image.")
    for name in group:
        crops = group_crops[name]
        st.markdown(f"**`{name}`** — {len(crops)} crop{'s' if len(crops) != 1 else ''}")
        for col, crop in zip(st.columns(min(len(crops), 4)), crops):
            col.image(crop, caption=f"{crop.shape[1]}×{crop.shape[0]}", width="stretch")

# ================================================================ Stage 1

with stage1:
    st.caption("Cheapest-first routing, unioned across the whole product. Paid engines stay off unless switched on.")

    allow_paid = st.checkbox(
        "💸 Allow paid engines (Google Vision / Claude)", value=False,
        help="Off by default. Nothing can bill while this is unchecked.",
    )
    use_vlm = st.checkbox(
        "Use local VLM tier (free; needs the MLX server running - see commands.txt)",
        value=False,
        help="Merges rapidocr + local VLM text (union), rather than only calling the VLM "
             "when rapidocr's read is weak. They catch different things on the same image.",
    )

    if not st.button("Run OCR", type="primary", width="stretch"):
        st.stop()

    # This machine has crashed before - silently, no traceback, process just gone - from
    # the MLX server (3-4GB resident) plus Streamlit/RapidOCR running at once and the OS
    # OOM-killing something. Check actual free memory before doing more work rather than
    # find out the hard way again.
    free_gb = memguard.free_memory_gb()
    if free_gb is not None and free_gb < 4.0:
        vlm_note = " The local VLM tier is on, which talks to the MLX server - the exact " \
            "combination that's crashed before." if use_vlm else ""
        st.warning(
            f"⚠️ Only ~{free_gb:.1f} GB free system memory.{vlm_note} "
            "Close some apps (or stop the MLX server if it's running) before continuing, "
            "or proceed at your own risk."
        )
        if free_gb < 2.0 and not st.checkbox(
            "I understand the risk - run anyway", value=False, key="mem_override"
        ):
            st.stop()

    # union_free_tiers mirrors use_vlm: when the VLM tier is on, merge its text with
    # rapidocr's rather than treating it as an escalate-only fallback. This is the policy
    # decided after the Hindi test - rapidocr and the VLM fail on different things on the
    # same image, so merging strictly beats picking one.
    policy = router.Policy(allow_paid=allow_paid, use_local_vlm=use_vlm, union_free_tiers=use_vlm)

    # Every crop of every image in the product, tagged with which image it came from.
    indexed = [
        (name, i, crop)
        for name in group
        for i, crop in enumerate(group_crops[name], 1)
    ]
    results = [router.read(crop, policy) for (_, _, crop) in indexed]
    union_text = "\n".join(r.text for r in results if r.text)

    # ---- Barcode: a whole-product signal. It lives on ONE face, so decode each image and
    # keep the strongest hit (an actual bar decode beats an OCR-digit fallback). The digit
    # cross-check uses the product-wide union text, which is the fullest evidence available.
    best_bc = best_bc_img = None
    last_bc = last_bc_img = None
    best_unverified = None
    for name in group:
        bc_img_i = barcode_source_image(group_steps[name])
        r = barcode.read_and_validate(bc_img_i, union_text)
        last_bc, last_bc_img = r, bc_img_i
        if r.code and (best_bc is None or r.source == "bars"):
            best_bc, best_bc_img = r, bc_img_i
        if r.unverified_candidate and not best_unverified:
            best_unverified = r.unverified_candidate
        if r.source == "bars":
            break
    bc = best_bc or last_bc
    bc_image = best_bc_img if best_bc is not None else last_bc_img
    # Nothing verified anywhere in the group: surface the best raw candidate seen on ANY of
    # the product's images, not just whichever image happened to be processed last.
    if bc.code is None and bc.unverified_candidate is None:
        bc.unverified_candidate = best_unverified

    st.subheader("Barcode (whole product)")
    bcol1, bcol2 = st.columns([1, 2])
    bc_crop = barcode.crop_region(bc_image, bc.points)
    if bc_crop is not None:
        bcol1.image(bc_crop, caption="Detected bars", width="stretch")
    elif bc.code:
        bcol1.caption("(no visual crop — code came from OCR digit text, not a bar detection)")
    else:
        bcol1.caption("No barcode symbol located in any image of this product.")
    if bc.code:
        bcol2.code(bc.code, language=None)
        if not bc.valid_checksum:
            bcol2.error("Checksum INVALID — not looked up.")
        else:
            bcol2.success("Checksum valid.")
            if bc.text_match is not None:
                label = "✅ matches OCR text" if bc.text_match_ok else "⚠️ weak match with OCR text"
                bcol2.metric("Barcode ↔ OCR text match", f"{bc.text_match:.0%}", label)
            if bc.looked_up:
                bcol2.write(f"**{bc.canonical_brand or '—'}** · {bc.canonical_name or '—'}")
                bcol2.caption(f"Source: {bc.lookup_source}")
                if bc.brand_match is not None:
                    bcol2.caption(f"Brand cross-check vs OCR: {bc.brand_match:.0%}")
            else:
                bcol2.caption("Not found in Open Food Facts or UPCitemdb.")
    elif bc.unverified_candidate:
        bcol2.warning(
            "No barcode could be verified automatically — the bar pattern couldn't be "
            "decoded and OCR's reading of the printed digits couldn't be confirmed, so this "
            "was NOT auto-trusted or looked up. It is OCR's best reading of the barcode line "
            "(it may include a reconstructed leading digit) — a starting point only. "
            "**Please verify it against the physical barcode.**"
        )
        bcol2.code(bc.unverified_candidate, language=None)
        bcol2.caption("See the 'Barcode trail' below for exactly how this candidate was formed.")
    else:
        bcol2.caption("No barcode digits found anywhere in this product's images either.")
    with st.expander("Barcode trail", expanded=False):
        for t in bc.trail:
            st.write(f"- {t}")

    # ---- Product-level field recall: the honest score. Union the OCR text across ALL the
    # product's images, then look for each ground-truth field. Scoring a single front-of-pack
    # shot against an FSSAI number that is only on the back would be nonsense.
    if record:
        st.divider()
        st.subheader("Field recall — product-level (union of all images vs ground truth)")
        hits = []
        for f in RECALL_FIELDS:
            got = found(f, record.get(f), union_text)
            if got is not None:
                hits.append({"field": f, "expected": str(record.get(f)),
                             "recovered": "✅" if got else "❌"})
        if hits:
            n_ok = sum(1 for h in hits if h["recovered"] == "✅")
            st.metric("Fields recovered", f"{n_ok} / {len(hits)}")
            st.dataframe(pd.DataFrame(hits), width="stretch", hide_index=True)
        else:
            st.info("No comparable ground-truth fields for this product.")

    # ---- Per-crop detail, grouped by source image.
    st.divider()
    st.subheader("Per-crop detail")
    for (name, i, crop), result in zip(indexed, results):
        st.divider()
        st.markdown(f"**`{name}`** · crop {i} — engine: `{result.engine}`")

        cost = f"${result.cost_usd:.4f}" if result.cost_usd else "free"
        conf = result.mean_confidence
        m = st.columns(4)
        m[0].metric("Lines", len(result.lines))
        m[1].metric("Characters", result.char_count)
        m[2].metric("Mean confidence", f"{conf:.2f}" if conf is not None else "—")
        m[3].metric("Cost", cost + (" (cached)" if result.cached else ""))

        left, right = st.columns(2)
        left.image(draw_boxes(crop, result), caption="Green = confident · red = weak", width="stretch")
        right.text_area("Extracted text", result.text or "(nothing found)", height=340,
                        key=f"t_{name}_{i}")

        with st.expander("Why this engine? (routing trail)", expanded=False):
            for t in result.trail:
                st.write(f"- {t}")

        if result.lines:
            st.dataframe(
                pd.DataFrame([
                    {"text": l.text,
                     "confidence": round(l.confidence, 3) if l.confidence is not None else None}
                    for l in result.lines
                ]),
                width="stretch", hide_index=True,
            )
