"""RapidOCR with the RIGHT recognition model for the script on the pack.

This module exists because of the single largest accuracy problem the verifier had, and
it was not a preprocessing problem - it was reading Greek packs with a model that has no
Greek in its character set. PP-OCR's default `ch` recogniser maps each Greek glyph to
its nearest Latin lookalike, so a pack reading

    ΠΛΟΥΣΙΑ ΣΕ ΒΙΤΑΜΙΝΗ D / ΠΡΟΩΘΗΤΙΚΕΣ ΣΥΣΚΕΥΑΣΙΕΣ / ΜΠΟΛ ΜΕ 3

came back as

    NAOYΣIA ΣE BITAMINH D / MPOOOHTIKEE / MNOAME3

Every downstream comparison was then diffing two different manglings of the same words.
Selecting `el_PP-OCRv5_rec_mobile` fixes it outright - measured on the same crop, the
three strings above come back correct. Same for Devanagari, which matters more for the
actual ONDC catalog than Greek does.

WHAT DOES NOT HELP, measured rather than assumed - because the obvious instinct here is
to add image preprocessing, and on this data it makes things worse:

    variant                     img12 chars   img15 chars   img40 chars
    raw crop                        495          1992           112
    + deskew                        495          1992           112
    + CLAHE                         487          1846            76
    + CLAHE + downscale 2200        486          1832           107

CLAHE loses text on every image tested (worst: img40, 112 -> 76 characters), and the
2200px clamp costs more on the dense back panel. PP-OCR does its own normalisation and
its own resizing internally; a second contrast pass on top fights it. So the pipeline
now hands RapidOCR the cropped image essentially as-is. Deskew is kept only because it
is free here (these crops are already straight, so it is a no-op) and genuinely helps a
rotated input, which this dataset happens not to contain.
"""

from __future__ import annotations

import logging
import re
import threading

import numpy as np

# Unicode blocks that identify a script from the text a first-pass read recovers.
# Even the wrong recogniser emits SOME correct characters of the right script (the
# mangled Greek above still contains Σ, Ω, Δ), which is exactly what makes a cheap
# first pass a usable detector.
_SCRIPT_BLOCKS = {
    "el": (0x0370, 0x03FF),           # Greek and Coptic
    "devanagari": (0x0900, 0x097F),   # Hindi, Marathi, Nepali
    "cyrillic": (0x0400, 0x04FF),
    "ta": (0x0B80, 0x0BFF),           # Tamil
    "te": (0x0C00, 0x0C7F),           # Telugu
    "ka": (0x0C80, 0x0CFF),           # Kannada
    "arabic": (0x0600, 0x06FF),
    "korean": (0xAC00, 0xD7AF),
    "japan": (0x3040, 0x30FF),
}

# Fraction of alphabetic characters that must belong to a script before we switch to
# its model. Low, because the wrong model converts most of the script's characters into
# Latin lookalikes - only the few with no lookalike survive, so the signal is faint by
# construction. Latin-script packs sit at ~0 here and are unaffected.
_SCRIPT_SWITCH_THRESHOLD = 0.03

_engines: dict[str, object] = {}
_engine_lock = threading.Lock()


def _quiet() -> None:
    logging.getLogger("RapidOCR").setLevel(logging.ERROR)


def get_engine(lang: str = "ch"):
    """A cached RapidOCR for one language. Construction costs ~1s and loads model
    weights, so engines are built once per language and reused for the whole session -
    the same reason `ocr/rapid.py` keeps its own singleton."""
    with _engine_lock:
        if lang in _engines:
            return _engines[lang]

        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

        # The non-`ch` recognisers ship as PP-OCRv5 mobile; asking for them at the
        # v6/small default the package boots with raises "Unsupported rec.lang_type".
        params = {}
        if lang != "ch":
            try:
                params = {
                    "Rec.lang_type": getattr(LangRec, lang.upper()),
                    "Rec.ocr_version": OCRVersion.PPOCRV5,
                    "Rec.model_type": ModelType.MOBILE,
                }
            except AttributeError:
                params = {}  # unknown language name - fall back to the default model

        prev = logging.root.manager.disable
        logging.disable(logging.INFO)
        try:
            engine = RapidOCR(params=params) if params else RapidOCR()
        finally:
            logging.disable(prev)
        _quiet()

        _engines[lang] = engine
        return engine


def detect_script(text: str) -> str:
    """Which recognition model this text calls for. Returns a LangRec-ish key.

    Counts characters per Unicode block in a first-pass read. Returns "ch" (the default
    model, which covers Latin and digits well) when nothing else is clearly present.
    """
    alpha = [c for c in str(text or "") if c.isalpha()]
    if len(alpha) < 12:
        return "ch"  # too little text to judge; the default handles Latin fine

    for lang, (lo, hi) in _SCRIPT_BLOCKS.items():
        hits = sum(1 for c in alpha if lo <= ord(c) <= hi)
        if hits / len(alpha) >= _SCRIPT_SWITCH_THRESHOLD:
            return lang
    return "ch"


class Line:
    """One OCR line. Mirrors `ocr.types.OCRLine` so the rest of the package is
    unchanged by this module existing."""

    __slots__ = ("text", "bbox", "confidence")

    def __init__(self, text: str, bbox, confidence):
        self.text, self.bbox, self.confidence = text, bbox, confidence


class Read:
    """The result of reading one image, plus which model produced it."""

    def __init__(self, lines: list[Line], lang: str, detected_from: str = ""):
        self.lines = lines
        self.lang = lang
        self.detected_from = detected_from

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines if l.text)

    @property
    def mean_confidence(self) -> float | None:
        scored = [l.confidence for l in self.lines if l.confidence is not None]
        return sum(scored) / len(scored) if scored else None


def _run(rgb: np.ndarray, lang: str) -> list[Line]:
    out = get_engine(lang)(rgb)
    if not out.txts:
        return []
    boxes = out.boxes if out.boxes is not None else [None] * len(out.txts)
    scores = out.scores if out.scores is not None else [None] * len(out.txts)
    lines = []
    for text, score, box in zip(out.txts, scores, boxes):
        bbox = ([(int(x), int(y)) for x, y in np.asarray(box)] if box is not None
                else [(0, 0), (0, 0), (0, 0), (0, 0)])
        lines.append(Line(str(text), bbox, float(score) if score is not None else None))
    return lines


def read(rgb: np.ndarray, lang: str = "auto") -> Read:
    """Read one image, choosing the recognition model to match the script.

    `lang="auto"` reads once with the default model, detects the script from what came
    back, and re-reads with the matching model only when it is a different one - so a
    Latin pack costs exactly one pass and a Greek pack costs two. Pass an explicit
    language to skip detection entirely (and to force BOTH images of a pair onto the
    same model, which `verify.report` does - comparing two images read by two different
    recognisers would measure the models, not the images).
    """
    if lang != "auto":
        return Read(_run(rgb, lang), lang)

    first = _run(rgb, "ch")
    detected = detect_script("\n".join(l.text for l in first))
    if detected == "ch":
        return Read(first, "ch")

    try:
        return Read(_run(rgb, detected), detected, detected_from="auto-detected")
    except Exception:
        # An unavailable or failed language model must never lose us the read we
        # already have in hand.
        return Read(first, "ch", detected_from=f"{detected} unavailable, used default")
