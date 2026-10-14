"""
Image Enhancement Verifier - upload a product photo, watch it get cleaned up, and see
field by field whether the clean version still tells the truth.

Run:  venv/bin/streamlit run verify_app.py

The flow is deliberately linear, because the whole point is that step 3 is not
optional:

    1. upload a rough product photo
    2. Gemini redraws it as a studio shot          (verify.enhance)
    3. both images are read and compared          (verify.report)
    4. every field gets its own verdict row, and the pair gets one overall verdict

Verification is OCR-ONLY: RapidOCR plus deterministic comparison (barcode decode,
digit/quantity set arithmetic, ORB feature matching). No VLM, no LLM, nothing
generative, no GPU - it runs on CPU and returns the same answer every time, which is
what makes the output usable as evidence.

Gemini is involved ONLY in step 2, the enhancement itself, and only in the "Enhance it
now" mode. The other two modes never call it at all.

Two other modes exist for when step 2 shouldn't run for real:

  * browse the pairs the sibling ondc-intelligence checkout already produced on disk
    (matched by filename stem, found by content - see _find_image_enhancement_root -
    no upload, no generation, no cost)
  * upload both images yourself, for a pair enhanced anywhere else

and for working on the verifier itself without paying for an image generation on
every reload.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Must be set before pyarrow's C++ library loads - same reason as app.py, which
# documents the macOS/Python 3.14 segfault this avoids. Kept here too because this app
# is launched independently and never imports that one.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import warnings

warnings.filterwarnings("ignore")

import streamlit as st

from verify import enhance, extract, report

# Model ids confirmed against `client.models.list()` rather than typed from memory -
# display names ("Nano Banana 2") do not map onto API ids in any guessable way, and a
# wrong id fails as a 404 at call time, halfway through a run. Only image GENERATION
# still needs Gemini - MLX has no image-output model, so the enhancement step itself
# is unaffected by the field/visual VLM switch to local below.
ENHANCE_MODELS = ["gemini-3.1-flash-image", "gemini-3.1-flash-lite-image",
                  "gemini-2.5-flash-image"]

MODE_FOLDER = "Browse image_enhancement_input/output"
MODE_ENHANCE = "Enhance it now with Gemini"
MODE_UPLOAD = "I'll upload both images"

def _find_image_enhancement_root() -> Path:
    """The sibling ondc-intelligence checkout's image_enhancement/ dir - found by
    CONTENT, not by a hardcoded name.

    OCR Prototype/ and the ondc-intelligence checkout are siblings under the same
    project root (`.resolve()` matters here: some runners, e.g. Streamlit's own AppTest
    harness, hand `__file__` back as a bare relative "verify_app.py", and `.parent.parent`
    on that lands one directory short of the real project root).

    The checkout's directory name is NOT stable, though - it has already changed once
    in this project from plain "ondc-intelligence" to "ondc-intelligence-fea-enhancement-
    evals" (a feature-branch checkout), and will change again on the next branch/re-clone.
    So this looks for any sibling directory whose name starts with "ondc-intelligence"
    and actually contains an image_enhancement/ folder, rather than hardcoding one name
    that the next checkout would silently break.
    """
    root = Path(__file__).resolve().parent.parent
    exact = root / "ondc-intelligence" / "image_enhancement"
    if exact.is_dir():
        return exact
    for sibling in sorted(p for p in root.iterdir() if p.is_dir()):
        if sibling.name.startswith("ondc-intelligence"):
            candidate = sibling / "image_enhancement"
            if candidate.is_dir():
                return candidate
    return exact  # nothing found; callers already handle a missing directory


IMAGE_ENHANCEMENT_ROOT = _find_image_enhancement_root()
FOLDER_INPUT_DIR = IMAGE_ENHANCEMENT_ROOT / "image_enhancement_input"
FOLDER_OUTPUT_DIRS = {
    "Nano Banana 2 (gemini-3.1-flash-image)": IMAGE_ENHANCEMENT_ROOT / "image_enhancement_output_nb",
    "Nano Banana 2 Lite (gemini-3.1-flash-lite-image)":
        IMAGE_ENHANCEMENT_ROOT / "image_enhancement_output_nb_lite",
}
FOLDER_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

st.set_page_config(page_title="Image Enhancement Verifier", layout="wide")


def find_output(output_dir: Path, stem: str) -> Path | None:
    """The enhanced file for one input, matched by STEM not extension - the enhancer
    writes whatever format the model returned (usually PNG), not the input's own."""
    matches = sorted(p for p in output_dir.glob(f"{stem}.*")
                     if p.suffix.lower() in FOLDER_IMAGE_EXTENSIONS)
    return matches[0] if matches else None


def folder_pairs(output_dir: Path) -> list[tuple[str, Path, Path]]:
    """(stem, input_path, output_path) for every input image that has a match in
    output_dir. Skipped silently if either directory is missing, so a stale path
    degrades to an empty picker rather than a crash."""
    if not FOLDER_INPUT_DIR.is_dir() or not output_dir.is_dir():
        return []
    out = []
    for p in sorted(FOLDER_INPUT_DIR.iterdir()):
        if p.suffix.lower() not in FOLDER_IMAGE_EXTENSIONS:
            continue
        match = find_output(output_dir, p.stem)
        if match:
            out.append((p.stem, p, match))
    return out

# ---------------------------------------------------------------- styling

STATUS_STYLE = {
    "MATCH":   ("#16a34a", "#dcfce7", "kept"),
    "CHANGED": ("#dc2626", "#fee2e2", "changed"),
    "LOST":    ("#ea580c", "#ffedd5", "lost"),
    "ADDED":   ("#ca8a04", "#fef9c3", "appeared"),
    "ABSENT":  ("#64748b", "#f1f5f9", "not read"),
}
VERDICT_STYLE = {
    "PASS":   ("#16a34a", "#dcfce7", "PASS",
               "Every field that could be checked survived the enhancement."),
    "REVIEW": ("#ca8a04", "#fef9c3", "NEEDS REVIEW",
               "Something could not be confirmed automatically. A human should look "
               "before this image is used."),
    "REJECT": ("#dc2626", "#fee2e2", "REJECT",
               "A critical, legally or commercially meaningful fact changed. Do not "
               "use this enhanced image."),
}

st.markdown("""
<style>
  .chip { display:inline-block; padding:2px 9px; border-radius:11px; font-size:11px;
          font-weight:700; letter-spacing:.03em; text-transform:uppercase; }
  .tier { font-size:10px; text-transform:uppercase; letter-spacing:.05em; font-weight:700; }
  .vrow { border:1px solid #e2e8f0; border-left-width:5px; border-radius:8px;
          padding:11px 14px; margin-bottom:7px; }
  .vrow .fname { font-weight:650; font-size:14px; }
  .vals { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px;
          margin-top:5px; }
  .vals .lbl { color:#64748b; }
  .note { color:#475569; font-size:12px; margin-top:5px; line-height:1.5; }
  .verdict { border-radius:12px; padding:18px 22px; margin:6px 0 18px; }
  .verdict h2 { margin:0 0 4px; font-size:26px; letter-spacing:.01em; }
  .verdict p { margin:0; font-size:13.5px; }
  .ftable { width:100%; border-collapse:collapse; font-size:13px; margin-top:4px; }
  .ftable th { text-align:left; padding:8px 10px; border-bottom:2px solid #cbd5e1;
               font-size:11px; text-transform:uppercase; letter-spacing:.04em;
               color:#64748b; }
  .ftable td { padding:8px 10px; border-bottom:1px solid #eef2f6; vertical-align:top;
               word-break:break-word; }
  .ftable td:nth-child(2), .ftable td:nth-child(3) {
               font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px;
               width:31%; }
  .ftable td:nth-child(4) { width:96px; white-space:nowrap; }
</style>
""", unsafe_allow_html=True)


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if s is not None else "—")


# ---------------------------------------------------------------- sidebar

st.sidebar.title("Settings")

mode = st.sidebar.radio(
    "Where does the enhanced image come from?",
    [MODE_FOLDER, MODE_ENHANCE, MODE_UPLOAD],
    help="Browse reads a pair already sitting on disk (no cost, no upload). Enhance "
         "calls Gemini's image model for real. Upload verifies a pair from anywhere "
         "else.",
)

enhance_model = st.sidebar.selectbox(
    "Enhancement model", ENHANCE_MODELS,
    disabled=(mode != MODE_ENHANCE),
    help="Image generation is not on the Gemini free tier — these all return a 429 "
         "with an unbilled key. Use Browse or Upload to verify pairs enhanced "
         "elsewhere.",
)
use_cv_preprocess = st.sidebar.checkbox(
    "Classical CV pass before enhancing", value=True,
    help="Deterministic sharpen / CLAHE / upscale. Hands the generator a cleaner image "
         "to read, so it has less to invent.",
    disabled=(mode != MODE_ENHANCE),
)

st.sidebar.divider()
st.sidebar.caption("**Verification**")
st.sidebar.info(
    "Verification is OCR-only — a recogniser plus deterministic comparison. No cloud "
    "VLM, no LLM, no network call for this step. Same pair in, same answer out, "
    "every time."
)

OCR_ENGINES = {
    "RapidOCR (PP-OCR, ONNX, CPU)": "rapidocr",
    "PaddleOCR-VL-1.5 (local, MLX)": "paddleocr-vl-1.5",
}
ocr_engine_label = st.sidebar.radio(
    "OCR engine", list(OCR_ENGINES),
    help="RapidOCR is PaddleOCR's own PP-OCR detection+recognition models, re-exported "
         "to ONNX Runtime — script-specific, returns real per-line boxes and "
         "confidence. PaddleOCR-VL-1.5 is Baidu's newer 0.9B document-parsing model, "
         "run locally through MLX (no server, first use downloads the model) — one "
         "model reads every script, but its plain OCR task returns flowing text with "
         "no per-line geometry, which weakens (not breaks) the background-line filter "
         "and the Prominent Text check below. Everything else — barcode, digit/"
         "quantity/price sets, the four labelled fields — is unaffected either way.",
)
ocr_engine = OCR_ENGINES[ocr_engine_label]

OCR_LANGS = {
    "Auto-detect from the pack": "auto",
    "Latin / digits (default)": "ch",
    "Greek": "el",
    "Devanagari (Hindi/Marathi)": "devanagari",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "ka",
    "Cyrillic": "cyrillic",
    "Arabic": "arabic",
}
ocr_lang_label = st.sidebar.selectbox(
    "Text script", list(OCR_LANGS), index=0,
    disabled=(ocr_engine != "rapidocr"),
    help="RapidOCR-only: PP-OCR's default recogniser has no Greek/Devanagari in its "
         "character set — it maps those glyphs to Latin lookalikes, which is why a "
         "Greek pack came back as gibberish. Auto reads once, detects the script, and "
         "re-reads with the right model. Whichever is chosen is used for BOTH images. "
         "PaddleOCR-VL-1.5 has one model for every script, so this has no effect on it.",
)
ocr_lang = OCR_LANGS[ocr_lang_label]

check_artwork = st.sidebar.checkbox(
    "Also compare the artwork", value=True,
    help="ORB feature matching plus a hue histogram over the product. Catches a pack "
         "whose design was rebuilt rather than re-rendered, and flavour/variant colour "
         "swaps — neither of which any text check can see. Classical CV, adds ~1s.",
)

st.sidebar.divider()
if enhance.api_key_available():
    st.sidebar.success("GEMINI_API_KEY found (image generation only)")
else:
    st.sidebar.warning("No GEMINI_API_KEY — the 'Enhance it now' mode needs it; "
                       "Browse and Upload do not.")

# ---------------------------------------------------------------- header

st.title("Image Enhancement Verifier")
st.caption(
    "An AI enhancer does not filter a photo — it redraws one. Every digit and every "
    "label in its output was re-rendered from what the model *believed* it saw, so it "
    "can quietly change a price, drop an FSSAI number, erase the veg mark, or invent a "
    "badge. This page checks what survived."
)

# ---------------------------------------------------------------- inputs

folder_input_path = folder_output_path = None
up_original = up_enhanced = None

if mode == MODE_FOLDER:
    if not FOLDER_INPUT_DIR.is_dir():
        st.error(f"Input folder not found: `{FOLDER_INPUT_DIR}`")
        st.stop()

    output_choice = st.selectbox("Compare the input against", list(FOLDER_OUTPUT_DIRS.keys()))
    output_dir = FOLDER_OUTPUT_DIRS[output_choice]
    available = folder_pairs(output_dir)

    if not available:
        st.warning(f"No filename-matched pairs between `{FOLDER_INPUT_DIR}` and "
                   f"`{output_dir}`.")
        st.stop()

    stem = st.selectbox(f"Image ({len(available)} available)",
                        [s for s, _, _ in available])
    folder_input_path, folder_output_path = next(
        (ip, op) for s, ip, op in available if s == stem
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.image(str(folder_input_path), caption=f"Original — {folder_input_path.name}",
                 use_container_width=True)
    with col_b:
        st.image(str(folder_output_path), caption=f"Enhanced — {folder_output_path.name}",
                 use_container_width=True)
    ready = True
else:
    col_a, col_b = st.columns(2)
    with col_a:
        up_original = st.file_uploader("Original product photo",
                                       type=["jpg", "jpeg", "png", "webp", "bmp"])
    with col_b:
        up_enhanced = (
            st.file_uploader("Enhanced image", type=["jpg", "jpeg", "png", "webp", "bmp"])
            if mode == MODE_UPLOAD else None
        )
    ready = up_original is not None and (mode != MODE_UPLOAD or up_enhanced is not None)

run = st.button(
    "Verify this pair" if mode == MODE_FOLDER
    else "Enhance and verify" if mode == MODE_ENHANCE else "Verify the pair",
    type="primary", disabled=not ready,
)

if not ready:
    st.info("Upload a product photo to begin." if mode == MODE_ENHANCE
            else "Upload both the original and the enhanced image.")
    st.stop()
if not run:
    st.stop()

# ---------------------------------------------------------------- 1. load + enhance

if mode == MODE_FOLDER:
    original_rgb = extract.load_rgb_bytes(folder_input_path.read_bytes())
    enhanced_rgb = extract.load_rgb_bytes(folder_output_path.read_bytes())

    meta_path = folder_output_path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        timing = (f"loaded from disk · model `{meta.get('model', '?')}` · "
                  f"preprocess {meta.get('preprocess_time_s', '?')}s · "
                  f"gemini {meta.get('gemini_time_s', '?')}s · "
                  f"generated at enhancement time, not now")
    else:
        timing = f"loaded from disk — {folder_input_path.name} / {folder_output_path.name}"

elif mode == MODE_ENHANCE:
    original_rgb = extract.load_rgb_bytes(up_original.getvalue())
    with st.spinner("Redrawing the photo as a studio image…"):
        result = enhance.enhance(original_rgb, model=enhance_model,
                                  use_preprocess=use_cv_preprocess)
    if not result.ok:
        st.error(f"Enhancement failed: {result.error}")
        if "RESOURCE_EXHAUSTED" in str(result.error) or "429" in str(result.error):
            st.info(
                "Gemini's **image generation** models are not available on the free "
                "tier — the quota is 0, not merely low, so retrying will not help. "
                "Either enable billing on the key in `OCR Prototype/.env`, or switch "
                "the sidebar to **Browse** or **Upload** to verify a pair that was "
                "enhanced elsewhere (for example the batches already in "
                f"`{IMAGE_ENHANCEMENT_ROOT / 'image_enhancement_output_nb'}`). "
                "Everything downstream of the enhancement works either way."
            )
        st.stop()
    enhanced_rgb = result.image
    timing = (f"preprocess {result.preprocess_time_s:.2f}s · generation "
              f"{result.gemini_time_s:.2f}s · model `{result.model}`"
              + (f" · input blur score {result.blur_score:.0f}"
                 if result.blur_score is not None else ""))

else:
    original_rgb = extract.load_rgb_bytes(up_original.getvalue())
    enhanced_rgb = extract.load_rgb_bytes(up_enhanced.getvalue())
    timing = "enhanced image supplied by upload — nothing was generated here"

st.subheader("1 · Before and after")
img_l, img_r = st.columns(2)
with img_l:
    st.image(original_rgb, caption=f"Original — {original_rgb.shape[1]}×{original_rgb.shape[0]}",
             use_container_width=True)
with img_r:
    st.image(enhanced_rgb, caption=f"Enhanced — {enhanced_rgb.shape[1]}×{enhanced_rgb.shape[0]}",
             use_container_width=True)
st.caption(timing)

# ---------------------------------------------------------------- 2. verify

st.subheader("2 · Verification")
progress_box = st.empty()
with st.spinner("Reading both images and comparing them…"):
    rep = report.run_verification(
        original_rgb, enhanced_rgb,
        check_artwork=check_artwork,
        lang=ocr_lang,
        engine=ocr_engine,
        progress=lambda m: progress_box.caption(m),
    )
progress_box.empty()

for err in rep.reader_errors():
    st.error(f"**OCR failed — nothing in the table below was checked.**\n\n{err}")

colour, bg, title, blurb = VERDICT_STYLE[rep.verdict]
st.markdown(
    f'<div class="verdict" style="background:{bg};border:1px solid {colour}33">'
    f'<h2 style="color:{colour}">{title}</h2><p style="color:{colour}">{blurb}</p></div>',
    unsafe_allow_html=True,
)

counts = rep.counts()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Kept", counts["MATCH"])
m2.metric("Changed", counts["CHANGED"])
m3.metric("Lost", counts["LOST"])
m4.metric("Appeared", counts["ADDED"])
m5.metric("Not read", counts["ABSENT"])

o_r, e_r = rep.original_reading, rep.enhanced_reading
st.caption(
    f"`{rep.engine}` read {o_r.lines} lines from the original and {e_r.lines} from the "
    f"enhanced image · original {rep.original_crop_note} · enhanced {rep.enhanced_crop_note}"
)

# ---------------------------------------------------------------- 3. the table

st.subheader("3 · Every check")

STATUS_ORDER = {"CHANGED": 0, "LOST": 1, "ADDED": 2, "MATCH": 3, "ABSENT": 4}


def cell(value) -> str:
    if value is None or not str(value).strip() or str(value) == "—":
        return '<span style="color:#cbd5e1">—</span>'
    return esc(str(value))


rows_html = []
for c in sorted(rep.checks, key=lambda c: (STATUS_ORDER.get(c.status, 9),
                                            not c.critical)):
    scol, sbg, stext = STATUS_STYLE.get(c.status, STATUS_STYLE["ABSENT"])
    tag = ""
    if c.critical:
        tag = ' <span class="tier" style="color:#b91c1c">critical</span>'
    if c.advisory:
        tag += ' <span style="color:#0891b2;font-size:10px">advisory</span>'
    note = (f'<br><span style="color:#94a3b8;font-size:11px">{esc(c.detail)}</span>'
            if c.status != "MATCH" else "")
    rows_html.append(
        f'<tr>'
        f'<td><b>{esc(c.name)}</b>{tag}</td>'
        f'<td>{cell(c.original)}</td>'
        f'<td>{cell(c.enhanced)}{note}</td>'
        f'<td><span class="chip" style="color:{scol};background:{sbg}">{stext}</span></td>'
        f'</tr>'
    )

st.markdown(
    '<table class="ftable">'
    '<thead><tr><th>Check</th><th>Original (raw input)</th><th>Enhanced</th>'
    '<th>Result</th></tr></thead>'
    f'<tbody>{"".join(rows_html)}</tbody></table>',
    unsafe_allow_html=True,
)

st.caption(
    "**Numbers** and **Quantities** are the sharpest signals here — digits are what OCR "
    "reads most reliably, and a number that appears in the enhanced image but not the "
    "original is fabrication. **critical** checks can reject on their own; **advisory** "
    "ones only ever ask for a human look. A check reading *not read* means neither image "
    "showed it — that is not a pass for that check, just nothing to compare."
)

# ---------------------------------------------------------------- 4. detail on demand

with st.expander("Why this verdict"):
    for r in rep.reasons:
        st.markdown(f"- {r}")

with st.expander("Full before/after sets (every number, quantity, price and word)"):
    for key in ("digits", "quantities", "prices", "words"):
        d = rep.diffs.get(key)
        if d is None:
            continue
        st.markdown(f"**{d.label}** — {d.summary()}")
        c1, c2, c3 = st.columns(3)
        c1.markdown("kept\n\n" + (", ".join(sorted(str(x) for x in d.retained)) or "—"))
        c2.markdown("lost\n\n" + (", ".join(sorted(str(x) for x in d.lost)) or "—"))
        c3.markdown("new\n\n" + (", ".join(sorted(str(x) for x in d.invented)) or "—"))
        st.divider()

with st.expander("Raw text read from each image"):
    t1, t2 = st.columns(2)
    for col, reading, name in ((t1, o_r, "Original"), (t2, e_r, "Enhanced")):
        with col:
            conf = (f", mean confidence {reading.mean_confidence:.2f}"
                    if reading.mean_confidence is not None else "")
            st.markdown(f"**{name}** — {reading.lines} lines, "
                        f"{len(reading.ocr_text)} chars{conf}, {reading.elapsed:.1f}s")
            if reading.error:
                st.error(reading.error)
            st.code(reading.ocr_text or "(nothing)", language=None)

if rep.original_reading.barcode and rep.original_reading.barcode.trail:
    with st.expander("Barcode decoder trail (original)"):
        for line in rep.original_reading.barcode.trail:
            st.text(line)
