"""Tier 3 - Google Cloud Vision. PAID. Ships DISABLED.

DOCUMENT_TEXT_DETECTION, ~$1.50 per 1,000 images. Reached only when both free tiers have
failed on an image.

`available()` is False unless GOOGLE_APPLICATION_CREDENTIALS is set, so a misconfigured run
can never quietly start billing. The router checks it before calling.
"""

from __future__ import annotations

import io
import os
import time

import numpy as np
from PIL import Image

from .types import OCRLine, OCRResult

COST_PER_IMAGE = 0.0015


def available() -> bool:
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return False
    try:
        import google.cloud.vision  # noqa: F401
    except ImportError:
        return False
    return True


def run(rgb: np.ndarray) -> OCRResult:
    if not available():
        raise RuntimeError(
            "Google Vision is disabled. Set GOOGLE_APPLICATION_CREDENTIALS and "
            "`pip install google-cloud-vision` to enable it."
        )

    from google.cloud import vision

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")

    t0 = time.perf_counter()
    client = vision.ImageAnnotatorClient()
    resp = client.document_text_detection(image=vision.Image(content=buf.getvalue()))
    elapsed = time.perf_counter() - t0

    if resp.error.message:
        raise RuntimeError(f"Google Vision error: {resp.error.message}")

    lines: list[OCRLine] = []
    for page in resp.full_text_annotation.pages:
        for block in page.blocks:
            for para in block.paragraphs:
                text = "".join(
                    "".join(s.text for s in word.symbols) + " " for word in para.words
                ).strip()
                v = para.bounding_box.vertices
                lines.append(
                    OCRLine(
                        text=text,
                        bbox=[(p.x, p.y) for p in v],
                        confidence=float(para.confidence),
                    )
                )

    return OCRResult(
        lines=lines, engine="google_vision", elapsed=elapsed, cost_usd=COST_PER_IMAGE
    )
