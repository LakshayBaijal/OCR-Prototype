"""LayoutLMv3 tier for Stage 2 - token classification over text + LAYOUT.

Where the heuristic parser reasons about a line's words and geometry with hand-written rules,
LayoutLMv3 learns to label each token (this token is a BRAND / an MRP / fine print) from the
text, its 2-D position, AND the pixels - the thing it is built for.

The honest catch, spelled out because it drives the whole design: `microsoft/layoutlmv3-base`
is a PRETRAINED backbone with NO product-field head. It cannot label "MRP" vs "FSSAI" out of
the box, and there is no public checkpoint fine-tuned on Indian grocery packs. So this tier is
useless until it is FINE-TUNED, and we have no token-level annotations - only product-level
field VALUES in the CSV. `train_layoutlm.py` bridges that with WEAK SUPERVISION: it aligns each
GT field value back onto the OCR tokens that spell it (see `weak_labels` below) to manufacture
token labels, then fine-tunes. The labels are noisy by construction, so treat this tier as a
learned SECOND OPINION that the merge step trusts only where it is confident - never as a
replacement for the barcode/regex signals that are exact.

NOTE: an experiment training this on just the 10 hand-verified `ground_truth_100` rows (with a
much wider label set - Ingredients, Nutritional_Details, etc.) was tried and reverted - 10
products was not enough data for the model to learn coherent spans (it was producing single
disconnected words like "CLASSIC" or "Taurine" instead of real values). Back on
`Ground Truth.csv`-based training (300 products) here, which the checkpoint below reflects.
Revisit the ground_truth_100 approach once more rows are filled in.

`available()` is False until a fine-tuned checkpoint exists, so the pipeline runs fine (on the
heuristic parser alone) whether or not anyone has trained this yet.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .schema import PositionedLine, ProductExtraction

MODEL_ID = "microsoft/layoutlmv3-base"
CHECKPOINT_DIR = Path(__file__).parent.parent / "layoutlmv3_finetuned"

# Token label set. Flat field labels (no BIO) - packaging fields are short runs and BIO's
# boundary modelling buys little on this noisy weak-labelled data. "O" = not a target field.
#
# The first 7 are the original set. The next 7 (FLAVOUR .. PACK_QUANTITY) extend coverage to
# more of the ground_truth_100-shaped schema (stage2/schema.to_ground_truth_dict) - added
# because they're the fields with a `Ground Truth.csv` column AND a short/discrete-enough
# value for weak_labels() to align onto OCR tokens. Fields with NO ground-truth column at all
# (Ingredients, Nutritional_Details, Usage_Details, Storage_Instructions, Bullet_Point,
# Diet_Type, Dimension, Preservatives, Ready_to_cook, Ready_to_eat) have nothing to weakly
# align against and so cannot be added here - they stay heuristic-only. Capacity is
# deliberately NOT its own label: it's already filled by mirroring Net_Quantity+UOM (see
# schema.to_ground_truth_dict), so a separate label would just be redundant. Baby_Weight and
# Absorption_Duration are numeric RANGES ("3-6 kg") that this exact-digit weak-labeller
# handles poorly, so they're skipped rather than trained on noisy labels.
LABELS = [
    "O", "BRAND", "MRP", "NET_QUANTITY", "FSSAI", "MANUFACTURER", "COUNTRY", "PRODUCT_NAME",
    "FLAVOUR", "MINERAL_SOURCE", "RECOMMENDED_AGE", "THEME_OCCASION", "HAIR_TYPE",
    "CAFFEINE_CONTENT", "PACK_QUANTITY",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# Which field key each token label fills. Field keys here MUST match the exact strings
# `stage2.heuristic` already uses for these fields, so a LayoutLMv3 guess and the heuristic's
# guess compete/merge on the SAME key via ProductExtraction.set() - never silently creating a
# second, unused key.
LABEL_TO_FIELD = {
    "BRAND": "Brand",
    "MRP": "Mrp",
    "NET_QUANTITY": "Net Quantity",
    "FSSAI": "Fssai No",
    "MANUFACTURER": "Manufacturer",
    "COUNTRY": "Country Of Origin",
    "PRODUCT_NAME": "Product Name",
    "FLAVOUR": "Flavours_or_Spices",
    "MINERAL_SOURCE": "Mineral_Source",
    "RECOMMENDED_AGE": "Recommended_Age",
    "THEME_OCCASION": "Theme_or_Occasion_Type",
    "HAIR_TYPE": "Hair_Type",
    "CAFFEINE_CONTENT": "Caffeine_Content",
    "PACK_QUANTITY": "Pack_Quantity",
}

_model = None
_processor = None


# ---------------------------------------------------------------- geometry helpers


def line_to_words(text: str, bbox: list[tuple[int, int]]) -> tuple[list[str], list[list[int]]]:
    """Split an OCR LINE into words, each given an estimated box by slicing the line's box
    horizontally. RapidOCR reports line-level boxes; LayoutLMv3 wants word-level, and an even
    horizontal split is a good-enough approximation for reading order + rough position."""
    words = text.split()
    if not words:
        return [], []
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    span = (x1 - x0) / len(words)
    out_words, out_boxes = [], []
    for i, w in enumerate(words):
        out_words.append(w)
        out_boxes.append([int(x0 + i * span), int(y0), int(x0 + (i + 1) * span), int(y1)])
    return out_words, out_boxes


def normalise_box(box: list[int], w: int, h: int) -> list[int]:
    """Scale a pixel box to LayoutLMv3's 0-1000 space, clamped in range."""
    def clamp(v):
        return max(0, min(1000, int(v)))
    return [clamp(1000 * box[0] / w), clamp(1000 * box[1] / h),
            clamp(1000 * box[2] / w), clamp(1000 * box[3] / h)]


# ---------------------------------------------------------------- weak labelling (for training)


def _digits(s) -> str:
    """Digit signature, float-safe: 75.0 -> '75' (not '750'), so a GT value the CSV stores as
    a float (Mrp 75.0, Fssai 11515006000531.0) matches the OCR word '75' / '11515006000531'.
    Same trick as evaluate_ocr.digits()."""
    s = str(s).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return re.sub(r"\D", "", s)


def _first_number(s) -> str:
    """The FIRST standalone number in a value, for fields whose GT string carries more than
    one number alongside a unit (Caffeine Content: '99mg/330ml Can' has both 99 and 330).
    `_digits()` would concatenate them into '99330', which never matches a single OCR word's
    digits - this is why Caffeine Content uses this instead of `_digits()` below."""
    m = re.search(r"\d+", str(s))
    return m.group(0) if m else ""


def weak_labels(words: list[str], gt: dict) -> list[str]:
    """Manufacture a token label per word by aligning the ground-truth field VALUES back onto
    the OCR words that spell them. This is the weak supervision that stands in for real
    token annotations we do not have.

      * textual fields (Brand / Manufacturer / Country / Product Name): a word is labelled if
        it is one of that value's own words (lowercased, punctuation-stripped).
      * numeric fields (MRP / FSSAI): a word is labelled if its digits match the value's.

    Noisy on purpose - a word like "energy" can be both a brand token and plain text, and the
    first matching field wins. Good enough to teach the model where fields TEND to sit; not a
    substitute for hand annotation."""
    labels = ["O"] * len(words)

    def norm(s):
        return re.sub(r"[^a-z0-9]", "", str(s).lower())

    textual = {
        "BRAND": gt.get("Brand"),
        "MANUFACTURER": gt.get("Manufacturer"),
        "COUNTRY": gt.get("Country Of Origin"),
        "PRODUCT_NAME": gt.get("Product Name"),
        "FLAVOUR": gt.get("Flavours & Spices"),
        "MINERAL_SOURCE": gt.get("Mineral Source"),
        "THEME_OCCASION": gt.get("Theme/ Occasion Type"),
        "HAIR_TYPE": gt.get("Hair Type"),
    }
    # Recommended Age / Caffeine Content / Pack Quantity are printed as text ("18+",
    # "99mg/330ml Can", "Pack of 6") but their CSV values carry a distinctive digit run - the
    # same alignment trick as Mrp/Fssai, just on fields with a unit/word around the number.
    numeric = {
        "MRP": gt.get("Mrp"),
        "FSSAI": gt.get("Fssai No"),
        "RECOMMENDED_AGE": gt.get("Recommended Age"),
        "CAFFEINE_CONTENT": gt.get("Caffeine Content"),
        "PACK_QUANTITY": gt.get("Pack Quantity"),
    }

    norm_words = [norm(w) for w in words]
    digit_words = [_digits(w) for w in words]

    for label, value in textual.items():
        if not value:
            continue
        value_tokens = {norm(t) for t in str(value).split() if len(norm(t)) >= 2}
        for i, nw in enumerate(norm_words):
            if labels[i] == "O" and nw and nw in value_tokens:
                labels[i] = label

    for label, value in numeric.items():
        # Caffeine Content values carry two numbers ("99mg/330ml Can") - match on just the
        # first one; the others (single numbers) use the full digit signature as before.
        d = _first_number(value) if label == "CAFFEINE_CONTENT" else _digits(value)
        if not d or d in ("0", ""):
            continue
        for i, dw in enumerate(digit_words):
            if labels[i] == "O" and dw and dw == d:
                labels[i] = label

    return labels


# ---------------------------------------------------------------- inference


def available() -> bool:
    """True only if a fine-tuned checkpoint exists AND the deps import. No checkpoint -> this
    tier is simply skipped and Stage 2 runs on the heuristic parser alone."""
    if not (CHECKPOINT_DIR / "config.json").exists():
        return False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _load():
    """Load the fine-tuned processor + model once (CPU/MPS)."""
    global _model, _processor
    if _model is None:
        import torch
        from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

        # apply_ocr=False: WE supply words+boxes (from Stage 1), we do not want the processor's
        # bundled tesseract re-OCRing the image.
        _processor = LayoutLMv3Processor.from_pretrained(str(CHECKPOINT_DIR), apply_ocr=False)
        _model = LayoutLMv3ForTokenClassification.from_pretrained(str(CHECKPOINT_DIR))
        _model.eval()
        if torch.backends.mps.is_available():
            _model.to("mps")
    return _processor, _model


def _predict_words(rgb: np.ndarray, words: list[str], boxes_1000: list[list[int]]) -> list[tuple[str, float]]:
    """Return (label, score) per input word for one image."""
    import torch
    from PIL import Image

    processor, model = _load()
    image = Image.fromarray(rgb).convert("RGB")
    enc = processor(image, words, boxes=boxes_1000, return_tensors="pt",
                    truncation=True, padding="max_length", max_length=512)
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits[0]  # (seq_len, n_labels)
    probs = torch.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)

    # Map subword predictions back to the original words via word_ids (first subword per word).
    word_ids = enc_word_ids(processor, image, words, boxes_1000)
    out: dict[int, tuple[str, float]] = {}
    for tok_idx, wid in enumerate(word_ids):
        if wid is None or wid in out or tok_idx >= len(pred):
            continue
        out[wid] = (ID2LABEL[int(pred[tok_idx])], float(conf[tok_idx]))
    return [out.get(i, ("O", 0.0)) for i in range(len(words))]


def enc_word_ids(processor, image, words, boxes_1000):
    """The processor call above does not expose word_ids directly through the tensor encoding,
    so re-run the tokenizer once (fast) to recover the subword->word map."""
    enc = processor(image, words, boxes=boxes_1000, truncation=True,
                    padding="max_length", max_length=512)
    return enc.word_ids(0)


def merge_into(extraction: ProductExtraction, lines: list[PositionedLine]) -> None:
    """Run LayoutLMv3 over the product's lines and MERGE its field guesses into `extraction`.

    Only ever ADDS confidence: it fills a field the heuristic left blank, or overrides one only
    when the model is markedly more confident. It never touches Upc/Ean (that is the barcode's,
    exact and checksum-verified). Groups by source image because the model reads one image at a
    time (it needs the pixels)."""
    if not available():
        return

    # Bucket lines by their image size (a proxy for "same crop") to reconstruct per-image input.
    by_image: dict[tuple[int, int], list[PositionedLine]] = {}
    for ln in lines:
        by_image.setdefault((ln.img_w, ln.img_h), []).append(ln)

    # field -> best (value, confidence) across the whole product
    votes: dict[str, tuple[str, float]] = {}
    for (w, h), img_lines in by_image.items():
        words, boxes = [], []
        for ln in img_lines:
            ws, bs = line_to_words(ln.text, ln.bbox)
            for word, box in zip(ws, bs):
                words.append(word)
                boxes.append(normalise_box(box, w, h))
        if not words:
            continue
        # We do not carry the crop pixels on PositionedLine, so feed a blank canvas of the
        # right size: LayoutLMv3 still uses text+layout strongly, and this keeps the tier
        # dependency-light. (A pixel-carrying variant is a straightforward extension.)
        canvas = np.full((h, w, 3), 255, np.uint8)
        preds = _predict_words(canvas, words, boxes)

        # Collect contiguous same-label spans into candidate field values.
        buf_label, buf_words, buf_conf = None, [], []
        def flush():
            if buf_label and buf_label in LABEL_TO_FIELD and buf_words:
                field = LABEL_TO_FIELD[buf_label]
                value = " ".join(buf_words)
                conf = sum(buf_conf) / len(buf_conf)
                if field not in votes or conf > votes[field][1]:
                    votes[field] = (value, conf)
        for word, (label, conf) in zip(words, preds):
            if label == buf_label:
                buf_words.append(word); buf_conf.append(conf)
            else:
                flush()
                buf_label, buf_words, buf_conf = label, [word], [conf]
        flush()

    for field, (value, conf) in votes.items():
        # scale model confidence down a touch - weak-labelled training makes it over-sure.
        extraction.set(field, value, 0.7 * conf, "layoutlmv3")
