"""Shared OCR contract.

Every engine - free or paid, local or remote - returns an OCRResult, so the router can
swap them without caring which produced the text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 4 corner points, (x, y) each - the shape RapidOCR/Google both speak natively.
BBox = list[tuple[int, int]]


@dataclass
class OCRLine:
    text: str
    bbox: BBox
    confidence: float | None  # None = engine does not report one (e.g. a VLM)


@dataclass
class OCRResult:
    lines: list[OCRLine] = field(default_factory=list)
    engine: str = ""
    elapsed: float = 0.0
    cost_usd: float = 0.0
    cached: bool = False
    # Why the router settled on this engine; each escalation appends a line.
    trail: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)

    @property
    def char_count(self) -> int:
        return sum(len(l.text) for l in self.lines)

    @property
    def mean_confidence(self) -> float | None:
        scored = [l.confidence for l in self.lines if l.confidence is not None]
        return sum(scored) / len(scored) if scored else None

    @property
    def digit_lines(self) -> int:
        """Lines containing digits. MRP, FSSAI, net quantity and barcodes are all numeric,
        so a read with no digits at all has almost certainly missed the informative panel."""
        return sum(1 for l in self.lines if any(c.isdigit() for c in l.text))

    def has_non_latin(self) -> bool:
        """Devanagari or other non-Latin script present - tells us if Indic support is needed."""
        return bool(re.search(r"[^\x00-\x7F]", self.text))
