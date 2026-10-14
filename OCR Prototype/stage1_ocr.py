"""Stage 1 OCR via PaddleOCR-VL-1.5, running locally through MLX - no server, no
network call, no cloud API. Deliberately a separate module from `verify/ocr_engine.py`
(RapidOCR/PP-OCR through ONNX Runtime) rather than a change to it: this is a second,
independent OCR path meant to be compared against the first, not a replacement grafted
into the same file.

WHY THIS MODEL. RapidOCR is not a different engine from PaddleOCR - it is PaddleOCR's
own PP-OCR detection+recognition models re-exported to ONNX Runtime for a lighter,
GPU-free install. So "RapidOCR reads badly" was really a claim about WHICH PP-OCR model
version got loaded, not about the ONNX approach. PaddleOCR-VL-1.5 is Baidu's newer,
much stronger successor in the same family - 0.9B parameters (small enough to run
comfortably on a Mac's unified memory, no discrete GPU needed) and measured at 94.5% on
OmniDocBench v1.5. It is a vision-language model architecturally, but a narrow,
task-prompted document-parsing one rather than an open-ended chat model - closer in
spirit to a trained task-specific OCR model than to Gemini/GPT, and it runs entirely on
this machine.

TWO GOTCHAS THAT WILL SILENTLY PRODUCE A BAD RESULT IF MISSED:

  1. THE PROMPT MUST BE THE LITERAL STRING "OCR:". This model is task-prompted rather
     than instruction-following: it answers "OCR:", "Table Recognition:", "Formula
     Recognition:" and a handful of other exact task strings, not a natural-language
     request. A generic prompt like "read the text in this image" or "describe this
     image" has been reported to return nothing at all - not a bad answer, no answer.

  2. THE DEFAULT max_tokens IS 100. mlx_vlm's own default is tuned for short chat
     replies, not a page of ingredients and a nutrition table - a real product label
     routinely runs several hundred tokens, and left at the default the output would
     truncate mid-word with no error raised. MAX_TOKENS below is set well above what
     a busy label needs, not left at the library default.

No server, by design: this is called in-process, matching how `verify/ocr_engine.py`
loads RapidOCR - a plain Python function call, nothing to start, bind, or leave running
between calls. The first call downloads and caches the model from Hugging Face (a few
hundred MB for the default quantisation); every call after that is local.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# bf16 (full precision), not a quantised build. Measured: the 6-bit build fails to
# load outright on this machine's installed mlx_vlm -
# "ValueError: Received 421 parameters not in model: ...layers.N.mlp.down_proj.biases,
# ...scales, ..." - and every one of those unmatched keys is a quantised-linear-layer
# parameter (.biases/.scales only exist on quantised weights). That means the bug is
# in the 6-bit checkpoint's key remapping specifically, not in loading this model
# generally - bf16 carries no such keys, so it cannot hit the same bug. Trade-off:
# a materially larger download (bf16 vs 6-bit is roughly 2.7x the bytes for a 0.9B
# model). If this line is ever changed back to a quantised build, re-verify it
# actually loads before trusting it - "smaller/faster" is not "confirmed working".
MODEL = "mlx-community/PaddleOCR-VL-1.5-bf16"

# The exact, literal task prompt this model requires - see the module docstring.
# Not an instruction to the model, a task SELECTOR.
OCR_PROMPT = "OCR:"

# mlx_vlm's own default is 100 - tuned for a short chat reply, not a product label's
# ingredients paragraph and nutrition table. Set high enough that a busy label
# finishes rather than truncates; a short label just stops early and costs nothing
# extra, since generation ends at the model's own stop token either way.
MAX_TOKENS = 2048

_loaded: dict[str, tuple] = {}  # model repo -> (model, processor, config)

# MLX's GPU stream is thread-local: it's created the first time a thread touches
# it and cannot be handed to another thread afterwards. Streamlit does not run
# script reruns on a single fixed thread, so calling generate() straight from
# request-handling code intermittently hits a thread that never created a
# stream - "RuntimeError: There is no Stream(gpu, 0) in current thread." Fix
# confirmed against upstream reports (mlx-vlm issue #1049, mlx-lm #1179): pin
# every MLX call - load AND generate - to one dedicated worker thread that
# lives for the whole process, so the stream is created once and always reused
# on the same thread it was created on. Do not call load()/generate() directly
# outside this executor.
_executor = ThreadPoolExecutor(max_workers=1)


def _get_model(model: str = MODEL):
    """Lazy-load and cache (model, processor, config) for one model repo, so a
    batch run pays the load cost once rather than per image - the same shape as
    `verify/ocr_engine.py`'s `get_engine()` cache for RapidOCR."""
    if model not in _loaded:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        # trust_remote_code=True: PaddleOCR-VL's RT-DETR backbone + reading-order
        # decoder is a custom architecture shipped as code inside the model repo,
        # not one built into transformers/mlx_vlm - without this the load fails
        # outright rather than silently degrading, so this is required, not a
        # convenience.
        mlx_model, processor = load(model, trust_remote_code=True)
        config = load_config(model)
        _loaded[model] = (mlx_model, processor, config)
    return _loaded[model]


def _pattern_key(line: str) -> str:
    """Collapse a line to its shape, so near-identical lines compare equal:
    digits -> '#', a standalone single letter -> '@'. "Vitamin C 100%" and
    "Vitamin E 100%" both become "vitamin @ #%", while genuinely different
    nutrition rows ("Protein 1.6g 7%" vs "Fat 1.7g 1%") keep distinct keys
    because their words differ - which is what makes this safe to truncate on.
    """
    import re

    key = " ".join(line.split()).lower()
    key = re.sub(r"\d+", "#", key)
    key = re.sub(r"\b[a-z]\b", "@", key)
    return key


def _dedupe_pattern_runs(text: str, max_repeats: int = 4) -> str:
    """Cut a degenerate run of lines that differ only trivially.

    `_dedupe_runaway_lines` only catches EXACT repeats, so it sails straight
    past the model walking an alphabet: measured on a Heinz label, the tail of
    the read was "Vitamin C 100%", "Vitamin E 100%", "Vitamin F 100%" ... on
    through "Vitamin T 100%" - fabricated, none of it printed on the pack.
    Same decoding trap, one character of variation per line. Threshold is
    deliberately looser than the exact-match one: a real label can legitimately
    carry a few same-shaped rows in a row, but not a dozen.
    """
    lines = text.splitlines()
    run_key, run_len = None, 0
    for i, line in enumerate(lines):
        key = _pattern_key(line)
        if key and key == run_key:
            run_len += 1
            if run_len > max_repeats:
                return "\n".join(lines[: i - max_repeats])
        else:
            run_key, run_len = key, 1
    return text


def _dedupe_runaway_lines(text: str, max_repeats: int = 3) -> str:
    """Cut off a degenerate-decoding loop: greedy decoding (temperature=0.0) has
    no randomness to escape a low-entropy trap, so once the model starts
    repeating the same line (typically a short one like a unit - "1kg" - near
    the end of a busy label) it can repeat it for the rest of the token budget
    instead of stopping. Truncate at the point a line has already repeated
    `max_repeats` times, on the assumption that anything after a real repeat
    run is the same loop, not new content."""
    lines = text.splitlines()
    run_value, run_len = None, 0
    for i, line in enumerate(lines):
        if line == run_value:
            run_len += 1
            if run_len > max_repeats:
                return "\n".join(lines[: i - max_repeats])
        else:
            run_value, run_len = line, 1
    return text


# Model's own canned fallback when a crop has no legible text - measured
# verbatim, identically, across every occurrence (never a paraphrase), so a
# plain substring check is safe: real label text never phrases anything this
# way, so this can only ever remove the refusal, never real content. Mainly
# hits `read_tiled`'s edge crops, which can land on a mostly-blank strip (a
# bottle cap, a plain background band) that the whole-image read never sees
# in isolation.
_REFUSAL_MARKERS = ("too blurry to recognize", "unable to recognize any text")


def _strip_refusals(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines()
        if not any(marker in line.lower() for marker in _REFUSAL_MARKERS)
    )


def _dedupe_intra_line_cycle(text: str, max_repeats: int = 4) -> str:
    """Cut a comma-separated PHRASE that repeats within a single line.

    Measured on img2 (Lakshay_Set): one line read as "A 100-foot-width,
    20-foot-width, 10-foot-width, 20-foot-width, 10-foot-width, ..." for
    hundreds of tokens - a real phrase, genuinely fabricated, cycling with
    period 2. None of the earlier filters catch this: `_dedupe_runaway_lines`
    and `_dedupe_pattern_runs` only compare whole LINES to each other, and
    this entire cycle sits inside ONE line, so both sail straight past it.
    `_strip_degenerate_lines` only catches a single CHARACTER running away,
    not a multi-word phrase. Checks period-1 (immediate repeat) and period-2
    (alternating) cycles - the two shapes actually observed - normalising
    digits to '#' first so "20-foot-width"/"10-foot-width" compare equal.
    """
    import re

    def norm(tok: str) -> str:
        return re.sub(r"\d+", "#", tok.strip().lower())

    out_lines = []
    for line in text.splitlines():
        tokens = line.split(", ")
        if len(tokens) > 2 * max_repeats:
            keys = [norm(t) for t in tokens]
            cut = None
            for period in (1, 2):
                run_start, i = 0, period
                while i < len(keys):
                    if keys[i] != keys[i - period]:
                        run_start = i - period + 1
                    elif (i - run_start) // period >= max_repeats:
                        cut = run_start if cut is None else min(cut, run_start)
                        break
                    i += 1
            if cut is not None:
                tokens = tokens[:cut]
                line = ", ".join(tokens)
        out_lines.append(line)
    return "\n".join(out_lines)


def _strip_degenerate_lines(text: str, min_len: int = 120, max_distinct: int = 4) -> str:
    """Drop a single line that is one long run of near-identical characters.

    Same degenerate-decoding trap as `_dedupe_runaway_lines`, but *within* one
    line instead of across several, so the line-level check never sees it.
    Measured on a Sprite bottle: the tile covering the moulded dimple pattern
    on the blank plastic (no text at all) emitted a single 2000-character run
    of "0". Real label text - even a long ingredients paragraph - always uses
    far more than a handful of distinct characters, so a long line built from
    almost none of them is never something that was printed on the pack.
    """
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) >= min_len and len(set(stripped)) <= max_distinct:
            continue
        # A long line one character RUNS AWAY with, even if the line also holds
        # a few real words. Measured on a Greek rice pack: a 2064-character
        # line that opened with plausible text and then repeated "я" 2032
        # times - 16 distinct characters overall, so the max_distinct rule
        # above sailed past it. No printed label devotes half a line to one
        # repeated letter.
        if len(stripped) >= min_len:
            letters = [c for c in stripped if c.isalpha()]
            if letters:
                top = max(set(letters), key=letters.count)
                if letters.count(top) / len(letters) > 0.5:
                    continue
        out.append(line)
    return "\n".join(out)


def _read_on_worker(image_path: str, model: str, max_tokens: int) -> str:
    """The actual mlx_vlm call. Must only ever run inside `_executor` - see the
    comment above it for why."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    mlx_model, processor, config = _get_model(model)
    formatted = apply_chat_template(processor, config, OCR_PROMPT, num_images=1)
    result = generate(
        mlx_model, processor, formatted,
        image=[image_path],
        max_tokens=max_tokens,
        temperature=0.0,  # OCR wants the model's single best reading, not variety
        # NOT repetition_penalty: measured directly (see git history / chat log)
        # that even 1.3 collapses this model to a single word on a fully legible
        # label - "Heineken" instead of the whole visible label text. Repetition
        # penalty pushes down EVERY previously-seen token, including ones that
        # legitimately recur (brand name, "kg", punctuation), and on this small
        # a model that's enough to tip the EOS token ahead of real content
        # almost immediately. The runaway-loop failure (img12: "1kg" x600) is
        # instead handled after generation by _dedupe_runaway_lines below,
        # which can only ever remove excess repeated lines, never suppress
        # real ones - confirmed it still catches the same loop with no
        # generation-time penalty at all.
        verbose=False,
    )
    return _dedupe_pattern_runs(_dedupe_runaway_lines(_dedupe_intra_line_cycle(_strip_degenerate_lines(_strip_refusals(result.text.strip())))))


def read(image_path: str | Path, model: str = MODEL, max_tokens: int = MAX_TOKENS) -> str:
    """Run PaddleOCR-VL-1.5 on one image, return the recognised text.

    `image_path` is a real file path, not bytes - mlx_vlm's `generate()` takes a
    path (or list of paths) and handles decoding itself, so there is no separate
    "load the image" step here to get subtly wrong.

    Dispatched onto `_executor` rather than run inline - see the comment on that
    executor for why this is required, not just a style choice.
    """
    return _executor.submit(_read_on_worker, str(image_path), model, max_tokens).result()


# Upper bound on tiles per image, so a very long pack shot cannot quietly turn
# one image into twenty model calls. 8 covers a 1:4 strip at 50% overlap.
MAX_TILES = 8


def _tile_boxes(size: tuple[int, int], overlap: float = 0.5) -> list[tuple[int, int, int, int]]:
    """Square-ish tiles stepped across BOTH axes with `overlap` between
    neighbours, so every region of the image is the dominant subject of at
    least one tile - including a right-hand nutrition table on a wide pack,
    which horizontal strips never isolate.

    Tile side = the image's shorter edge, so a tall pack is cut into stacked
    squares and a wide pack into side-by-side ones, with no special-casing of
    orientation - EXCEPT a roughly square photo, which is the case that rule
    quietly fails on. Measured on a Vaseline jar (2524x2493, aspect 1.01): tile
    = shorter edge = 2493 is ALREADY the whole frame, so both "tiles" produced
    were just the full image shifted 31px - no more zoom than the plain
    full-image read, and a vertical edge-printed number and a small circular
    purity seal both went unread. Halving the tile side once the image is
    close to square forces a real zoom on both axes; tall/wide photos (where
    this rule was already verified working - a nutrition table, a corner
    quantity mark) sit well above the aspect threshold and are unaffected.
    """
    w, h = size
    tile = min(w, h)
    if max(w, h) / tile < 1.35:  # roughly square - shorter-edge tile gives no zoom at all
        tile //= 2
    step = max(1, int(tile * (1 - overlap)))

    def _starts(extent: int) -> list[int]:
        if extent <= tile:
            return [0]
        out = list(range(0, extent - tile + 1, step))
        if out[-1] != extent - tile:  # always include a flush-to-the-far-edge tile
            out.append(extent - tile)
        return out

    boxes = [
        (x, y, x + tile, y + tile)
        for y in _starts(h)
        for x in _starts(w)
    ]
    if len(boxes) <= MAX_TILES:
        return boxes
    # Too many: keep an evenly spaced subset, always retaining first and last
    # so both extreme edges (where quantity/weight marks live) stay covered.
    idx = [round(i * (len(boxes) - 1) / (MAX_TILES - 1)) for i in range(MAX_TILES)]
    return [boxes[i] for i in sorted(set(idx))]


def _dedupe_fuzzy_lines(lines: list[str], threshold: float = 0.85, min_len: int = 6) -> list[str]:
    """Collapse lines that are the SAME text read differently, not just exact
    or substring duplicates.

    `_drop_contained_lines` only catches one line being a clean fragment of
    another. It does nothing when every tile garbles the same physical text
    differently - measured on a Vaseline back label, one address block
    ("Unilever Cste d'Voire Boulevard de Weil...") got re-read 3-4 times, each
    with different noise, none a substring of another, bloating one label to
    47 lines of mostly-repeated content.

    Uses `difflib.SequenceMatcher` ratio, longest-line-first (so the most
    complete garbling of a repeated block is what survives). Threshold set
    from measurement, not guessed: real duplicate pairs here scored 0.94-0.98
    similar; genuinely different sentences that merely share a template
    ("Manufactured by/FABRIQUE PAR:" vs "Manufactured for/Pour:") scored 0.67
    - 0.85 sits in the gap between them.
    """
    import difflib

    order = sorted(range(len(lines)), key=lambda i: -len(lines[i]))
    kept: list[int] = []
    for i in order:
        li = lines[i].strip()
        if len(li) < min_len:
            kept.append(i)
            continue
        is_dup = any(
            difflib.SequenceMatcher(None, li.lower(), lines[j].strip().lower()).ratio() >= threshold
            for j in kept if len(lines[j].strip()) >= min_len
        )
        if not is_dup:
            kept.append(i)
    kept.sort()
    return [lines[i] for i in kept]


def _drop_contained_lines(lines: list[str], min_len: int = 3) -> str:
    """Collapse a cluster of lines where one is just a fragment of another.

    `read_tiled` merges 8+ independent reads of overlapping crops, so the same
    real word regularly shows up whole from one tile and truncated from
    another - "BLUESEAL" and "BLUESE", "original" and "riginal". Both are
    genuine, so exact-dedup (which only strips byte-identical lines) keeps
    both, bloating the text with redundant fragments that inflate the
    precision/recall audit's character counts without adding information.

    Rule: process longest-first, keep a line unless its normalised form is
    already a substring of something already kept. Longest-first specifically
    (not shortest-first) because the failure mode measured was NOT ambiguous
    once checked: "TRIPLE PURIFICATION" and "VIL TRIPLE PURIFICATION" both
    appeared, and keeping the longer one costs a few wrong leading characters
    at worst - keeping the shorter one instead would have thrown away real
    content whenever the extra characters on the longer line turn out to be
    genuine (measured: "pure petroleum" vs "pure petroleum jelly" - the
    "jelly" is real). A short line under `min_len` is dropped outright ("R",
    "KL") - too short to be meaningful label content on its own regardless of
    containment.
    """
    ranked = sorted(lines, key=len, reverse=True)
    kept: list[str] = []
    kept_keys: list[str] = []
    for line in ranked:
        key = " ".join(line.split()).lower()
        if len(key) < min_len:
            continue
        if any(key in k for k in kept_keys):
            continue
        kept.append(line)
        kept_keys.append(key)
    # restore original relative order rather than the length-sorted one
    order = {line: i for i, line in enumerate(lines)}
    kept.sort(key=lambda l: order.get(l, 0))
    return "\n".join(kept)


def read_tiled(image_path: str | Path, model: str = MODEL, max_tokens: int = MAX_TOKENS) -> str:
    """Read the whole image, plus a tight top-edge and bottom-edge crop, and
    merge the results - recovers small/off-flow text this model's single
    holistic read skips over. Measured directly: "1 Litre" printed in a
    corner of a fruit-juice carton was completely absent from the full-image
    read, but present and correctly read from a crop of just that corner -
    the model attends to the whole image's most salient content and can drop
    peripheral text entirely, not garble it.

    THE CROP MUST BE TIGHT - measured this too, not assumed it. A 60%-height
    crop (meant to be generous) still missed "1 Litre" entirely; only a ~20%
    crop of just the bottom edge actually isolated it enough for the model to
    read it. A crop busy enough to still have a dominant subject of its own
    (the fruit photography, the rest of the label) reproduces the same
    holistic-attention drop at a smaller scale - tightness is what does the
    work here, not merely "not the full image."

    TILES ARE A GRID, NOT TOP/BOTTOM STRIPS, because product shots are not all
    portrait. A wide pack shot (a chocolate bar lying on its side) carries its
    nutrition table on the RIGHT, which horizontal strips never isolate - they
    only ever re-crop the same left-to-right span. Square-ish tiles stepped
    across BOTH axes cover the right-hand table, the corner marks, and the
    centre artwork alike, whatever the aspect ratio.

    BIASED TOWARDS COMPLETENESS, NOT TIDINESS: every tile's text is merged in,
    including near-duplicate and mangled readings of the same words. A wrong
    reading is recoverable downstream (it still scores as a near-match, and a
    human can see what it meant); text that was never emitted at all is not.

    This is NOT a fix for garbled dense text (long ingredients/nutrition
    panels read as gibberish): measured that separately - cropping tighter on
    a Heinz ingredients panel didn't produce a truer reading, it produced a
    DIFFERENT fabricated repeat loop. That failure is this model's legibility
    ceiling on small photographed print, which tiling cannot move. Use this
    for recovering dropped content, not for rescuing illegible content.

    Costs one call per tile plus one for the whole image (`MAX_TILES` caps it).
    """
    import tempfile

    from PIL import Image

    def _read_crop(crop: Image.Image) -> str:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            crop.save(tmp_path)
            return read(tmp_path, model=model, max_tokens=max_tokens)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    with Image.open(image_path) as im:
        im.load()
        boxes = _tile_boxes(im.size)
        crops = [im.crop(b) for b in boxes]

    blocks = [read(image_path, model=model, max_tokens=max_tokens)]
    blocks += [_read_crop(c) for c in crops]

    seen: set[str] = set()
    merged: list[str] = []
    for block in blocks:
        for line in block.splitlines():
            key = " ".join(line.split()).lower()
            if key and key not in seen:
                seen.add(key)
                merged.append(line)
    return _drop_contained_lines(_dedupe_fuzzy_lines(merged))


def read_tiled_array(rgb, model: str = MODEL, max_tokens: int = MAX_TOKENS) -> str:
    """`read_tiled`, for an in-memory RGB array - same reason `read_array`
    exists next to `read`."""
    import tempfile

    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        Image.fromarray(rgb).save(tmp_path)
        return read_tiled(tmp_path, model=model, max_tokens=max_tokens)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def read_array(rgb, model: str = MODEL, max_tokens: int = MAX_TOKENS) -> str:
    """Same as `read`, for an in-memory RGB array instead of a file on disk - what
    `verify/extract.py` has, since that module works on uploaded/generated images
    that were never written to a file. `generate()` only accepts a path, so this
    writes one temporary file and always cleans it up, success or exception.
    """
    import tempfile

    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        Image.fromarray(rgb).save(tmp_path)
        return read(tmp_path, model=model, max_tokens=max_tokens)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="an image file, or a directory of images")
    ap.add_argument("--model", default=MODEL, help=f"default: {MODEL}")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = ap.parse_args()

    target = Path(args.path)
    if target.is_dir():
        images = sorted(p for p in target.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise SystemExit(f"no images found in {target}")
    elif target.is_file():
        images = [target]
    else:
        raise SystemExit(f"not found: {target}")

    print(f"model: {args.model}  ({len(images)} image(s))")
    print("(first call downloads and caches the model from Hugging Face - can take a")
    print(" few minutes once; every call after that is local)\n")

    for path in images:
        try:
            text = read(path, model=args.model, max_tokens=args.max_tokens)
        except Exception as exc:
            print(f"=== {path.name} ===\nERROR  {type(exc).__name__}: {exc}\n")
            continue
        print(f"=== {path.name} ===\n{text}\n")


if __name__ == "__main__":
    main()
