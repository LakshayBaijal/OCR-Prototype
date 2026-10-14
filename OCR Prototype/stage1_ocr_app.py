"""PaddleOCR-VL-1.5 text check - raw silhouette vs Gemini-enhanced silhouette.

Run:  venv/bin/streamlit run stage1_ocr_app.py

Deliberately separate from verify_app.py / app.py, and deliberately narrow: this does
ONE thing - run PaddleOCR-VL-1.5 on a matched pair of silhouette images and show the
raw extracted text side by side. No verdicts, no field comparison, no diffing - that is
verify_app.py's job, once the text itself is trusted. This exists to answer "is the OCR
any good", not "did the product change".

Compares two fixed folders by filename stem:
    RAW_DIR      image_enhancement_processed_silhouette/       (cut from the original photo)
    ENHANCED_DIR image_enhancement_output_nb_silhouette/       (cut from Gemini's output)

THE ALPHA GOTCHA, again, because it bites every silhouette file in this project: these
PNGs have a transparent background, but only the ALPHA channel was zeroed when they were
made - the pixels *underneath* it still hold the original shelf colours. A naive
`.convert("RGB")` (which is what PIL/mlx_vlm would do if handed the file directly) brings
that hidden background back in full, so PaddleOCR-VL would silently read shelf clutter
as if it were on the product. Every image here is flattened onto white BEFORE it reaches
the model - see `_flatten`.
"""

from __future__ import annotations

import os

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

import stage1_ocr

RAW_DIR = Path(
    "/Users/lakshay.baijal/Desktop/Projects/Image:Text Enhancement/"
    "ondc-intelligence-fea-enhancement-evals/image_enhancement/"
    "image_enhancement_processed_silhouette"
)
ENHANCED_DIR = Path(
    "/Users/lakshay.baijal/Desktop/Projects/Image:Text Enhancement/"
    "ondc-intelligence-fea-enhancement-evals/image_enhancement/"
    "image_enhancement_output_nb_silhouette"
)
# Both folders also hold embeddings.json/.npz/embeddings_index.json from earlier work -
# real files, not images, and must not show up as a "product".
NOT_A_PRODUCT = {"embeddings", "embeddings_index"}

st.set_page_config(page_title="PaddleOCR-VL Text Check", layout="wide")


def _flatten(path: Path) -> np.ndarray:
    """PNG (possibly with a transparent background hiding real shelf pixels) -> a
    flat RGB array, background composited to white. See the module docstring."""
    with Image.open(path) as im:
        has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
        if not has_alpha:
            return np.asarray(im.convert("RGB"))
        rgba = im.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return np.asarray(Image.alpha_composite(white, rgba).convert("RGB"))


def matched_stems() -> list[str]:
    if not RAW_DIR.is_dir() or not ENHANCED_DIR.is_dir():
        return []
    raw = {p.stem for p in RAW_DIR.glob("*.png")} - NOT_A_PRODUCT
    enhanced = {p.stem for p in ENHANCED_DIR.glob("*.png")} - NOT_A_PRODUCT
    return sorted(raw & enhanced, key=lambda s: (int("".join(c for c in s if c.isdigit()) or 0), s))


def run_ocr(stem: str) -> dict:
    """OCR both sides of one pair. Cached in session_state - MLX inference is local
    and free, but it is not instant, and nothing here should re-run just because a
    widget elsewhere on the page changed."""
    cache = st.session_state.setdefault("ocr_cache", {})
    if stem in cache:
        return cache[stem]
    raw_path, enh_path = RAW_DIR / f"{stem}.png", ENHANCED_DIR / f"{stem}.png"
    result = {}
    for side, path in (("raw", raw_path), ("enhanced", enh_path)):
        try:
            text = stage1_ocr.read_array(_flatten(path))
            result[side] = {"text": text, "error": None}
        except Exception as exc:
            result[side] = {"text": "", "error": f"{type(exc).__name__}: {exc}"}
    cache[stem] = result
    return result


st.title("PaddleOCR-VL Text Check")
st.caption(
    "Raw silhouette vs Gemini-enhanced silhouette, same product, side by side. Just the "
    "extracted text for now — no verdicts, no field comparison. Model: "
    f"`{stage1_ocr.MODEL}`, running locally through MLX. First OCR call downloads and "
    "caches the model (a few hundred MB, one time); every call after that is local."
)

stems = matched_stems()
if not stems:
    st.error(
        f"No matching filenames found between:\n\n`{RAW_DIR}`\n\n`{ENHANCED_DIR}`\n\n"
        "Check both folders exist and contain `imgN.png` files."
    )
    st.stop()

with st.sidebar:
    st.caption(f"{len(stems)} matched pair(s)")
    stem = st.selectbox("Product", stems)
    run_all = st.button(f"Run OCR on all {len(stems)}", width="stretch")
    if st.button("Clear cached results", width="stretch"):
        st.session_state.pop("ocr_cache", None)
        st.rerun()

a, b = st.columns(2)
with a:
    st.subheader("Raw")
    st.image(str(RAW_DIR / f"{stem}.png"), width="stretch")
with b:
    st.subheader("Gemini-enhanced")
    st.image(str(ENHANCED_DIR / f"{stem}.png"), width="stretch")

if st.button(f"Run OCR on `{stem}`", type="primary"):
    with st.spinner("Reading both images with PaddleOCR-VL-1.5…"):
        run_ocr(stem)

result = st.session_state.get("ocr_cache", {}).get(stem)
if result:
    ta, tb = st.columns(2)
    for col, side in ((ta, "raw"), (tb, "enhanced")):
        with col:
            r = result[side]
            if r["error"]:
                st.error(r["error"])
            else:
                st.caption(f"{len(r['text'])} characters, {r['text'].count(chr(10)) + 1} lines")
                st.code(r["text"] or "(nothing read)", language=None)
else:
    st.info("Click **Run OCR** to extract text from this pair.")

if run_all:
    progress = st.progress(0.0, text="Starting…")
    for i, s in enumerate(stems):
        progress.progress((i + 1) / len(stems), text=f"{s} ({i + 1}/{len(stems)})")
        run_ocr(s)
    progress.empty()
    st.success(f"Done — {len(stems)} pairs processed. Pick any product above to view its text.")

# ---------------------------------------------------------------- all results so far
cache = st.session_state.get("ocr_cache", {})
if cache:
    st.divider()
    st.subheader(f"Processed so far ({len(cache)}/{len(stems)})")
    for s in stems:
        if s not in cache:
            continue
        r = cache[s]
        with st.expander(
            f"{s} — raw {len(r['raw']['text'])} chars · enhanced {len(r['enhanced']['text'])} chars"
        ):
            c1, c2 = st.columns(2)
            c1.code(r["raw"]["text"] or r["raw"]["error"] or "(nothing)", language=None)
            c2.code(r["enhanced"]["text"] or r["enhanced"]["error"] or "(nothing)", language=None)
