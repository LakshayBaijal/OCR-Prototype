"""
Stage 0 - Preprocessing visualiser.

Run:  .venv/bin/streamlit run app.py

Pick any image from the Dataset and inspect what every preprocessing step does to
it, before a single pixel reaches the OCR engine.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

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

GT_FIELDS = [
    "Product Name", "Brand", "Mrp", "Net Quantity", "Product Quantity",
    "Upc / Ean", "Fssai No", "Manufacturer", "Country Of Origin", "Product Type Name",
]

st.set_page_config(page_title="Stage 0 - Preprocessing", layout="wide")


@st.cache_data(show_spinner=False)
def list_images() -> list[str]:
    return sorted(p.name for p in DATASET.glob("*.jpg"))


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
st.sidebar.caption(f"{len(images):,} images in Dataset/")

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

st.sidebar.divider()
st.sidebar.subheader("Steps")
cfg = {
    "orientation": st.sidebar.checkbox("Orientation Fix", True),
    "deskew": st.sidebar.checkbox("Deskew", True),
    "normalise": st.sidebar.checkbox("Normalisation", True),
    "resolution": st.sidebar.checkbox("Resolution Band", True),
    "background": st.sidebar.checkbox("Discard Background", True),
    "group": st.sidebar.checkbox("Multi-Image Grouping", True),
}

st.sidebar.subheader("Parameters")
cfg["max_angle"] = st.sidebar.slider("Deskew max angle (deg)", 1.0, 30.0, 15.0, 0.5)
cfg["clip_limit"] = st.sidebar.slider("CLAHE clip limit", 0.5, 6.0, 2.0, 0.5)
cfg["min_side"], cfg["max_side"] = st.sidebar.slider(
    "Resolution band (px)", 300, 3000, (900, 2200), 50
)
cfg["min_area_frac"] = st.sidebar.slider(
    "Grouping: min instance area (frac of image)", 0.01, 0.30, 0.04, 0.01
)

# ---------------------------------------------------------------- run

path = DATASET / choice
steps, crops = run_pipeline(str(path), cfg)

record = gt.get(choice)

stage0, stage1 = st.tabs(["Stage 0 · Preprocessing", "Stage 1 · OCR"])

# ================================================================ Stage 0

with stage0:
    st.caption(f"`{choice}` — every step below is shown before it is handed to the OCR engine.")

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
        st.info("No ground-truth row found for this filename.")

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

    st.divider()
    st.header(f"➡️ Output to OCR — {len(crops)} image{'s' if len(crops) != 1 else ''}")
    st.caption("These are the exact pixels Stage 1 (OCR) will receive.")
    for col, crop in zip(st.columns(min(len(crops), 4)), crops):
        col.image(crop, caption=f"{crop.shape[1]}×{crop.shape[0]}", width="stretch")

# ================================================================ Stage 1

with stage1:
    st.caption("Cheapest-first routing. Paid engines stay off unless you switch them on.")

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

    # union_free_tiers mirrors use_vlm: when the VLM tier is on, merge its text with
    # rapidocr's rather than treating it as an escalate-only fallback. This is the policy
    # decided after the Hindi test - rapidocr and the VLM fail on different things on the
    # same image, so merging strictly beats picking one.
    policy = router.Policy(allow_paid=allow_paid, use_local_vlm=use_vlm, union_free_tiers=use_vlm)

    for idx, crop in enumerate(crops, 1):
        result = router.read(crop, policy)

        st.divider()
        st.subheader(f"Crop {idx} — engine: `{result.engine}`")

        cost = f"${result.cost_usd:.4f}" if result.cost_usd else "free"
        conf = result.mean_confidence
        m = st.columns(4)
        m[0].metric("Lines", len(result.lines))
        m[1].metric("Characters", result.char_count)
        m[2].metric("Mean confidence", f"{conf:.2f}" if conf is not None else "—")
        m[3].metric("Cost", cost + (" (cached)" if result.cached else ""))

        left, right = st.columns(2)
        left.image(draw_boxes(crop, result), caption="Green = confident · red = weak", width="stretch")
        right.text_area("Extracted text", result.text or "(nothing found)", height=340, key=f"t{idx}")

        with st.expander("Why this engine? (routing trail)", expanded=True):
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

        # Did OCR actually recover what the ground truth says is printed here?
        if record:
            hits = []
            for f in ["Brand", "Mrp", "Net Quantity", "Fssai No", "Upc / Ean", "Manufacturer"]:
                got = found(f, record.get(f), result.text)
                if got is not None:
                    hits.append({"field": f, "expected": str(record.get(f)),
                                 "recovered": "✅" if got else "❌"})
            if hits:
                st.markdown("**Field recall on this crop** (vs ground truth)")
                st.dataframe(pd.DataFrame(hits), width="stretch", hide_index=True)
