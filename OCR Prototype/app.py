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
from ocr import local_vlm, router
from ocr.types import OCRResult
from preprocessing import run_pipeline
from stage2 import GT_FIELD_ORDER, heuristic
from stage2.schema import PositionedLine, ProductExtraction

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

stage0, stage1, stage2_tab = st.tabs(
    ["Stage 0 · Preprocessing", "Stage 1 · OCR", "Stage 2 · Field Parsing"]
)

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
    vlm_backend_choice = st.radio(
        "VLM text tier",
        ["Off", "Local Gemma-3 (MLX, free, unlimited)", "OpenRouter Gemma-4 (free tier, 50 requests/DAY)"],
        index=0, horizontal=True,
        help="Off: RapidOCR only. Local Gemma-3: needs the MLX server running (see "
             "commands.txt) - unlimited, but Gemma-3, not Gemma-4. OpenRouter Gemma-4: the "
             "model actually asked for, hosted free by openrouter.ai - BUT this tier is "
             "called PER CROP, not per product, and the free quota is only 50 requests/day - "
             "a single product with 10+ crops can burn most of a day's quota. Prefer union "
             "OFF (below) with this backend unless you specifically mean to spend the quota "
             "on a small test batch.",
    )
    use_vlm = vlm_backend_choice.startswith("Local")
    use_openrouter_vlm = vlm_backend_choice.startswith("OpenRouter")
    union_vlm = False
    if use_vlm or use_openrouter_vlm:
        union_vlm = st.checkbox(
            "Merge with RapidOCR on every crop (union) instead of escalation-only",
            value=False,
            help="Union: run on EVERY crop and merge the text - catches things RapidOCR "
                 "misses even when RapidOCR's own read already looked fine, but for "
                 "OpenRouter this can burn the daily quota fast. Off (escalation-only): only "
                 "call the VLM tier when RapidOCR's own read is already weak - far more "
                 "quota-friendly for OpenRouter, and the recommended default for it.",
        )

    run_clicked = st.button("Run OCR", type="primary", width="stretch")

    # This machine has crashed before - silently, no traceback, process just gone - from
    # the MLX server (3-4GB resident) plus Streamlit/RapidOCR running at once and the OS
    # OOM-killing something. Check actual free memory before doing more work rather than
    # find out the hard way again. Only checked at the moment "Run OCR" is actually
    # clicked - never gates anything on a rerun triggered by some OTHER widget, since that
    # used to st.stop() the whole script (see below).
    memory_blocked = False
    if run_clicked:
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
                memory_blocked = True

    # Keyed per-product so switching products in the sidebar doesn't show a stale OCR result
    # from whatever product was last run (same convention Stage 2's modes use below).
    stage1_cache_key = f"stage1_result::{products.product_key(choice) or choice}"

    if run_clicked and not memory_blocked:
        # union_vlm mirrors the "merge with RapidOCR" checkbox above: when on, rapidocr and
        # the selected VLM backend's text are merged rather than the VLM only escalating in
        # when rapidocr's own read is weak. Merging strictly beats picking one for local_vlm
        # (decided after the Hindi test - they fail on different things on the same image);
        # for openrouter_vlm, escalation-only is the quota-friendlier default instead (see
        # the radio/checkbox help text above).
        policy = router.Policy(
            allow_paid=allow_paid, use_local_vlm=use_vlm, use_openrouter_vlm=use_openrouter_vlm,
            union_free_tiers=union_vlm,
        )

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

        st.session_state[stage1_cache_key] = (indexed, results, bc, bc_image, union_text)

    cached_stage1 = st.session_state.get(stage1_cache_key)
    # IMPORTANT: no st.stop() below. Streamlit reruns this whole script on ANY interaction
    # anywhere in the app (a Stage 2 button, an unrelated checkbox) - stopping here on every
    # one of those reruns is exactly what made Stage 2 render blank on every click before this
    # fix, since Stage 2's own code (further down the script) never got a chance to run.
    if cached_stage1 is None:
        st.info("Click **Run OCR** above to read this product's images.")
    else:
        indexed, results, bc, bc_image, union_text = cached_stage1

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

        # Stash this run's OCR output where the Stage 2 tab (below) can reach it. Re-stashed
        # from the cache on every rerun (not only right after clicking "Run OCR") so it's
        # always current even on reruns triggered by some other widget entirely.
        st.session_state["stage1_indexed"] = indexed
        st.session_state["stage1_results"] = results
        st.session_state["stage1_bc"] = bc

# ================================================================ Stage 2

with stage2_tab:
    st.caption(
        "Turns OCR lines + the decoded barcode into the same JSON shape as "
        "`ground_truth_100/rowN.json` — view it here directly, nothing to download."
    )

    if "stage1_indexed" not in st.session_state:
        st.info("Run OCR in the **Stage 1 · OCR** tab first — Stage 2 needs its barcode result.")
        st.stop()

    indexed_s1 = st.session_state["stage1_indexed"]
    bc_s1 = st.session_state["stage1_bc"]

    def _render_vlm_mode(
        mode_label: str, extract_fn, cache_prefix: str, widget_key: str,
        is_available, unavailable_msg: str, spinner_verb: str = "Calling the local VLM",
    ):
        """Shared UI for all model-based modes (local-VLM direct, local-VLM OCR-then-map,
        real Gemini) - they differ only in which extract_fn does the work and how
        availability is checked (MLX server reachable vs. GEMINI_API_KEY set), not in how
        the button/cache/status/JSON flow works. Returns (extraction, source_caption)."""
        use_fallback = st.checkbox(
            f"Fall back to heuristic + LayoutLM for fields {mode_label} leaves null", value=True,
            key=f"{widget_key}_fallback",
            help="The model's answer is always the PRIMARY source when present; this only "
                 "fills fields it left null (or supplies the whole record if the call isn't "
                 "available at all). Barcode-decoded UPC still wins over the model's guess "
                 "when both are available.",
        )
        if not is_available():
            st.warning(
                unavailable_msg
                + (" Falling back to heuristic + LayoutLM only." if use_fallback
                   else " Nothing to show without the fallback enabled — check the box above.")
            )

        image_paths = [str(DATASET / name) for name in group]
        # Keyed per-product so switching images in the sidebar doesn't show a stale result
        # from whatever product was last extracted.
        cache_key = f"{cache_prefix}::{products.product_key(choice) or choice}"

        if st.button(f"Run {mode_label} extraction", type="primary", key=f"{widget_key}_button"):
            with st.spinner(f"{spinner_verb} ({mode_label})…"):
                extraction, vlm_dict = extract_fn(image_paths, use_fallback=use_fallback)
            st.session_state[cache_key] = (extraction, vlm_dict)

        cached = st.session_state.get(cache_key)
        if cached is None:
            # IMPORTANT: no st.stop() here. Streamlit reruns this whole script on ANY
            # interaction anywhere in the app (switching images, toggling an unrelated
            # checkbox, clicking "Run OCR" in Stage 1) - stopping here on every one of those
            # reruns (which is what an earlier version of this did) meant the JSON view below
            # almost never rendered at all, which is exactly the "page looks broken, nothing
            # renders" bug. Instead, fall through with an empty placeholder so the rest of
            # the tab (including this section's own layout) still renders normally.
            st.info(f"Click 'Run {mode_label} extraction' above to see a result for this product.")
            return ProductExtraction(), "No result yet for this product — click the button above."

        extraction, vlm_dict = cached
        # This distinction matters: "the model ran and left some fields null" and "the call
        # failed outright" look identical in the merged JSON below unless surfaced separately.
        if vlm_dict is None:
            st.error(
                "The call did not return usable JSON (unavailable, timed out, or every retry "
                "produced unparseable output) — "
                + ("showing heuristic + LayoutLM fallback only below." if use_fallback else
                   "result below is EMPTY (fallback is off). Check the box above or verify "
                   "the model/server is reachable.")
            )
            source_caption = f"{mode_label} call FAILED — heuristic/LayoutLM fallback only" if use_fallback else f"{mode_label} call FAILED — no fallback"
        else:
            vlm_filled = sum(1 for v in vlm_dict.values() if v not in (None, [], ""))
            st.success(f"{mode_label} returned {vlm_filled} non-null field(s).")
            source_caption = mode_label + (" + heuristic/LayoutLM fallback for the rest" if use_fallback else "")

        with st.expander("Raw VLM response (before merge)", expanded=False):
            st.json(vlm_dict if vlm_dict is not None else {"error": "no usable response"})

        return extraction, source_caption

    extraction_mode = st.radio(
        "Extraction mode",
        [
            "Gemini API (real, proven baseline — costs money)",
            "OpenRouter (free hosted models — Gemma-4, Nemotron)",
            "Direct VLM → JSON (local Gemma-3, free)",
            "OCR then Map (local, two calls: read text first, map separately)",
            "Heuristic + LayoutLM (regex-based)",
        ],
        index=0,
        help="Gemini API: the real production-grade model ondc-intelligence already uses — "
             "not free, but the honest quality reference everything else here is measured "
             "against. OpenRouter: free, ALREADY-HOSTED models (no local MLX/GPU needed) — "
             "default is Gemma-4 26B-A4B, which sidesteps the local mlx-vlm quantization bug "
             "entirely; 50 requests/day free. Direct VLM: one local call, image in, "
             "structured JSON out, for free. OCR then Map: two local calls — OCR each image "
             "first, then a text-only call maps the combined text into the schema (can't see "
             "purely visual cues like the veg/non-veg dot, but decomposes the task into two "
             "easier steps). Heuristic + LayoutLM: the older regex-based path.",
    )

    if extraction_mode.startswith("OpenRouter"):
        from stage2.openrouter_direct import MODEL_CHOICES, api_key_available, extract_product_openrouter

        or_model_key = st.selectbox(
            "OpenRouter model", list(MODEL_CHOICES),
            help="All three are confirmed free (:free, $0 cost) and vision-capable as of "
                 "2026-07. gemma-4-26b is the model asked about specifically; the two "
                 "nemotron options are alternatives worth comparing.",
        )

        def _extract_openrouter(image_paths, use_fallback):
            return extract_product_openrouter(
                image_paths, use_fallback=use_fallback, model=MODEL_CHOICES[or_model_key],
            )

        extraction, source_caption = _render_vlm_mode(
            "OpenRouter", _extract_openrouter, f"openrouter_result_{or_model_key}", f"openrouter_{or_model_key}",
            is_available=api_key_available,
            unavailable_msg="OPENROUTER_API_KEY is not set (see stage2/openrouter_direct.py's "
                             "module docstring — sign up free at openrouter.ai, no card/phone "
                             "needed, put the key in OCR Prototype/.env, never paste it into a "
                             "terminal/chat that gets logged).",
            spinner_verb="Calling OpenRouter",
        )

    elif extraction_mode.startswith("Gemini API"):
        from stage2.gemini_direct import api_key_available, extract_product_gemini

        extraction, source_caption = _render_vlm_mode(
            "Gemini API", extract_product_gemini, "gemini_result", "gemini",
            is_available=api_key_available,
            unavailable_msg="GEMINI_API_KEY is not set (see stage2/gemini_direct.py's module "
                             "docstring — put it in OCR Prototype/.env, never paste it into a "
                             "terminal/chat that gets logged).",
            spinner_verb="Calling the real Gemini API",
        )

    elif extraction_mode.startswith("Direct VLM"):
        from stage2.vlm_direct import extract_product_direct

        extraction, source_caption = _render_vlm_mode(
            "Direct VLM → JSON", extract_product_direct, "vlm_direct_result", "direct",
            is_available=local_vlm.available,
            unavailable_msg="MLX server not reachable at 127.0.0.1:8080 (see commands.txt to start it).",
        )

    elif extraction_mode.startswith("OCR then Map"):
        from stage2.vlm_direct import extract_product_two_call

        extraction, source_caption = _render_vlm_mode(
            "OCR then Map", extract_product_two_call, "vlm_two_call_result", "twocall",
            is_available=local_vlm.available,
            unavailable_msg="MLX server not reachable at 127.0.0.1:8080 (see commands.txt to start it).",
        )

    else:
        use_vlm_text = st.checkbox(
            "Use local VLM (Gemma-3) for TEXT, RapidOCR for boxes only — recommended",
            value=True,
            help="RapidOCR's transcription has frequent spelling errors on this dataset "
                 "(\"Puried ater\" instead of \"Purified Water\"). The local VLM reads far "
                 "more accurately but reports no real per-line geometry, so it REPLACES "
                 "RapidOCR's text for every regex/keyword/block field extractor here, while "
                 "RapidOCR's boxes still power the brand/name layout heuristic and "
                 "LayoutLMv3 (both need real word positions, which the VLM can't provide). "
                 "Needs the MLX server running (see commands.txt) — silently falls back to "
                 "RapidOCR's own text if the server isn't reachable.",
        )
        use_layoutlm_ui = st.checkbox(
            "Use LayoutLMv3 tier (local inference, no per-call cost — on by default)",
            value=True,
            help="Only ever fills a field the heuristic parser left blank, or overrides one "
                 "when markedly more confident. Automatically skipped if no fine-tuned "
                 "checkpoint exists at layoutlmv3_finetuned/ (heuristic-only result either "
                 "way). Always reasons over RapidOCR's real word positions, regardless of "
                 "the VLM toggle above.",
        )

        dual_policy = router.Policy(use_local_vlm=use_vlm_text)
        if use_vlm_text and not local_vlm.available():
            st.warning(
                "MLX server not reachable at 127.0.0.1:8080 — using RapidOCR's own text as "
                "a fallback. Start the server (see commands.txt) and rerun for the VLM's reading."
            )

        # Re-read each crop with BOTH sources. This is cheap even for RapidOCR: it hits the
        # same checksum cache (ocr_cache.sqlite) Stage 1 already populated for this exact
        # crop, so only a genuinely new VLM call (if enabled and not yet cached) does real work.
        layout_lines: list[PositionedLine] = []
        text_lines: list[PositionedLine] = []
        for name, i, crop in indexed_s1:
            h, w = crop.shape[:2]
            geometry_result, text_result = router.read_dual(crop, dual_policy)
            for ln in geometry_result.lines:
                layout_lines.append(
                    PositionedLine(text=ln.text, bbox=ln.bbox, confidence=ln.confidence, img_w=w, img_h=h)
                )
            for ln in text_result.lines:
                text_lines.append(
                    PositionedLine(text=ln.text, bbox=ln.bbox, confidence=ln.confidence, img_w=w, img_h=h)
                )

        extraction = heuristic.parse(text_lines, layout_lines, bc_s1)

        layoutlm_ran = False
        if use_layoutlm_ui:
            from stage2 import layoutlm

            if layoutlm.available():
                layoutlm.merge_into(extraction, layout_lines)  # real geometry only, never text_lines
                layoutlm_ran = True
            else:
                st.caption("No fine-tuned checkpoint at `layoutlmv3_finetuned/` — heuristic-only result below.")

        source_caption = (
            ("Heuristic parser + LayoutLMv3 tier" if layoutlm_ran else "Heuristic parser only")
            + (", text from local VLM" if (use_vlm_text and local_vlm.available()) else ", text from RapidOCR")
        )

    st.subheader("Ground-truth-format JSON")
    st.caption(
        source_caption + ". `Category` is always \"Grocery\"; `Sub_Category`/`Images`/"
        "`Variants` are catalog metadata this pipeline doesn't fill from a photo "
        "(see schema.to_ground_truth_dict)."
    )
    gt_dict = extraction.to_ground_truth_dict()
    st.json({k: gt_dict.get(k) for k in GT_FIELD_ORDER})

    with st.expander("Field provenance (confidence + which rule/engine fired)", expanded=False):
        prov_rows = [
            {"field": name, "value": f.value, "confidence": round(f.confidence, 2), "source": f.source}
            for name, f in sorted(extraction.fields.items())
        ]
        if prov_rows:
            st.dataframe(pd.DataFrame(prov_rows), width="stretch", hide_index=True)
        if extraction.bullets:
            st.write("**Bullet_Point candidates:**")
            for b in extraction.bullets:
                st.write(f"- {b}")
        if not prov_rows and not extraction.bullets:
            st.info("Nothing resolved for this product.")
