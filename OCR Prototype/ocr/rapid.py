"""Tier 1 - RapidOCR. Free, local, unlimited.

Runs the PP-OCR (PaddleOCR) models on ONNX Runtime. We use this rather than `paddleocr`
itself because `paddlepaddle` has no Python 3.14 build - same models, no dead dependency.

This is the only engine that gives us per-line bounding boxes AND confidences, which the
router needs to decide what to escalate and the later crop-patching stage needs to re-read.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .types import OCRLine, OCRResult

_engine = None


def _quiet_rapidocr() -> None:
    """Silence RapidOCR's own logging.

    RapidOCR resets its "RapidOCR" logger to INFO *and* attaches its own stderr handler
    during `RapidOCR()` construction, so setting the level at import time is clobbered the
    moment an engine is built. It has to be re-applied AFTER construction. Left alone it
    prints a `text detection result is empty` WARNING for every blank crop - thousands of
    lines on a full benchmark, burying the actual report. An empty detection is normal here
    (blank/backdrop crops), so ERROR is the right floor.
    """
    logging.getLogger("RapidOCR").setLevel(logging.ERROR)


def _get_engine():
    """Load once. Model init costs ~1s, so never pay it per image."""
    global _engine
    if _engine is None:
        from rapidocr import RapidOCR

        # RapidOCR logs each model it loads at INFO *from inside* its constructor - before
        # _quiet_rapidocr() below could ever run. Disable INFO globally just for the
        # construction window (restoring whatever was set before) so those startup lines
        # don't print either; WARNING+ still passes through.
        prev_disable = logging.root.manager.disable
        logging.disable(logging.INFO)
        try:
            _engine = RapidOCR()
        finally:
            logging.disable(prev_disable)
        _quiet_rapidocr()
    return _engine


def run(rgb: np.ndarray) -> OCRResult:
    t0 = time.perf_counter()
    out = _get_engine()(rgb)
    elapsed = time.perf_counter() - t0

    lines: list[OCRLine] = []
    if out.txts:
        for text, score, box in zip(out.txts, out.scores, out.boxes):
            lines.append(
                OCRLine(
                    text=str(text),
                    bbox=[(int(x), int(y)) for x, y in np.asarray(box)],
                    confidence=float(score),
                )
            )

    return OCRResult(lines=lines, engine="rapidocr", elapsed=elapsed, cost_usd=0.0)
