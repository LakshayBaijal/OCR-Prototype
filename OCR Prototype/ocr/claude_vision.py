"""Tier 4 - Claude vision. PAID, most expensive per call. Ships DISABLED.

Deliberately the LAST resort and the most narrowly scoped: it is meant for *tight crops*
of lines the cheaper tiers could not read, not whole images. Sending a full 2000px shelf
photo here burns tokens for text the free tiers already got right.

Cheapest model first (Haiku), escalating to a stronger one only on failure - the
architecture's "Cheapest Version then Better Version if failed".

`available()` is False unless ANTHROPIC_API_KEY is set, so it cannot bill by accident.
"""

from __future__ import annotations

import base64
import io
import os
import time

import numpy as np
from PIL import Image

from .types import OCRLine, OCRResult

CHEAP_MODEL = "claude-haiku-4-5-20251001"
STRONG_MODEL = "claude-opus-4-8"

PROMPT = (
    "Transcribe every character of text visible in this product packaging image. "
    "Output ONLY the raw text, preserving line breaks. No commentary, no markdown."
)


def available() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _encode(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def run(rgb: np.ndarray, model: str = CHEAP_MODEL) -> OCRResult:
    if not available():
        raise RuntimeError(
            "Claude vision is disabled. Set ANTHROPIC_API_KEY and `pip install anthropic` "
            "to enable it."
        )

    import anthropic

    t0 = time.perf_counter()
    msg = anthropic.Anthropic().messages.create(
        model=model,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": _encode(rgb)}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    )
    elapsed = time.perf_counter() - t0

    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    h, w = rgb.shape[:2]

    # A VLM returns prose, not geometry: we know WHAT it read, not WHERE. One box for the
    # whole crop is the honest answer - and acceptable here precisely because the crop was
    # already localised by Tier 1's boxes.
    lines = [
        OCRLine(text=t, bbox=[(0, 0), (w, 0), (w, h), (0, h)], confidence=None)
        for t in text.splitlines() if t.strip()
    ]

    usd = (msg.usage.input_tokens * 1e-6) + (msg.usage.output_tokens * 5e-6)  # rough, Haiku-ish
    return OCRResult(lines=lines, engine=f"claude:{model}", elapsed=elapsed, cost_usd=usd)
