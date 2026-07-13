"""Tier 0 - permanent checksum cache.

The cheapest OCR call is the one never made. Keyed on the SHA-256 of the *exact pixels*
handed to the engine (not the filename), so a re-run, a duplicate image, or the same crop
reached by a different path all hit the cache.

This matters most for the PAID tiers: a Google Vision or Claude answer is bought once and
kept forever. The architecture calls for exactly this ("Every Resolved Image Patch is
cached permanently").
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

from .types import OCRLine, OCRResult

DB = Path(__file__).parent.parent / "ocr_cache.sqlite"


def checksum(rgb: np.ndarray) -> str:
    """Hash the pixels themselves, so identical content collides regardless of filename."""
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB)
    c.execute(
        "CREATE TABLE IF NOT EXISTS ocr ("
        "  key TEXT PRIMARY KEY, engine TEXT, payload TEXT, cost_usd REAL)"
    )
    return c


def get(rgb: np.ndarray, engine: str) -> OCRResult | None:
    key = f"{checksum(rgb)}:{engine}"
    with _conn() as c:
        row = c.execute("SELECT payload FROM ocr WHERE key = ?", (key,)).fetchone()
    if not row:
        return None

    d = json.loads(row[0])
    return OCRResult(
        lines=[
            OCRLine(text=l["text"], bbox=[tuple(p) for p in l["bbox"]], confidence=l["confidence"])
            for l in d["lines"]
        ],
        engine=d["engine"],
        elapsed=d["elapsed"],
        cost_usd=0.0,  # a cache hit costs nothing, whatever the original call cost
        cached=True,
    )


def put(rgb: np.ndarray, result: OCRResult) -> None:
    key = f"{checksum(rgb)}:{result.engine}"
    payload = json.dumps(
        {
            "engine": result.engine,
            "elapsed": result.elapsed,
            "lines": [
                {"text": l.text, "bbox": [list(p) for p in l.bbox], "confidence": l.confidence}
                for l in result.lines
            ],
        }
    )
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO ocr (key, engine, payload, cost_usd) VALUES (?,?,?,?)",
            (key, result.engine, payload, result.cost_usd),
        )


def stats() -> dict:
    with _conn() as c:
        rows = c.execute("SELECT engine, COUNT(*), SUM(cost_usd) FROM ocr GROUP BY engine").fetchall()
    return {e: {"entries": n, "spent_usd": round(s or 0, 4)} for e, n, s in rows}
