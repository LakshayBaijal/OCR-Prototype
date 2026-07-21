"""Learned foreground matte (background removal), replacing the colour heuristic.

The old classical `segment()` keys on backdrop colour + border-connectivity, which *always*
fails on glossy, white-on-white, or gradient-backdrop packs - it clips the product or punches
holes through it. A model trained for salient-object detection does not: it returns a clean
silhouette of the whole product regardless of colour.

We use rembg + BiRefNet, run locally on onnxruntime (CoreML on Apple Silicon - no API, no
tokens, same "free local tier" philosophy as RapidOCR/MLX). The pipeline uses the matte only
to CROP to the product, never to white-out pixels, so a wrong matte costs a slightly loose
crop - never an erased label.

Two deliberate choices:
  * OPTIONAL. If rembg is not installed, `available()` is False and preprocessing falls back
    to the classical heuristic. The pipeline never hard-depends on the model.
  * CACHED. BiRefNet is heavy (~10s/image here), so every matte is cached on the SHA-256 of
    the exact pixels - a re-run, a duplicate image, or the same crop reached another way all
    hit the cache. It is a one-time cost per image.
"""

from __future__ import annotations

import hashlib
import io
import os
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image

# The rembg session to request. Changing this one line switches models - the cache is keyed
# on the name, so old and new masks never collide.
#
# u2netp, not birefnet-general: measured birefnet-general at ~10s/image on CPU (see
# _get_session below for why CPU, not CoreML) - fine for a one-off, unusable for anything
# interactive or for a full-dataset run. u2netp: ~0.1-0.2s/image, a ~150-100x speedup, ~4.7MB
# model. The step this feeds is CROP-ONLY (see step_background in preprocessing.py) - it only
# ever reads the mask's bounding box, never its per-pixel shape - so u2netp's rougher edges
# and occasional internal hole (patched by solidify() anyway) cost nothing here. Measured on
# 4 sample packs: u2netp's crop bounding box landed within 1-3px of birefnet-general's on
# every one.
MODEL = "u2netp"

DB = Path(__file__).parent / "bgmask_cache.sqlite"

_session = None


def available() -> bool:
    """True only if the model tier should run: rembg importable AND not disabled.

    Set BGREMOVE=0 to force the classical heuristic instead - useful for fast benchmark
    iterations, where paying ~10s/image for the matte on a cold cache is not worth it.
    """
    if os.environ.get("BGREMOVE", "1") == "0":
        return False
    try:
        import rembg  # noqa: F401
    except ImportError:
        return False
    return True


def _get_session():
    """Build the rembg session once (weights load ~once, then stay resident).

    CPU provider on purpose. onnxruntime's CoreML EP hung for MINUTES compiling a bigger
    model's (birefnet-general) graph on this machine before u2netp replaced it - CoreML
    session-create never returned in reasonable time, near-zero CPU while it waited on the
    ANE. u2netp on plain CPU is already ~0.1-0.2s/image with a ~1.8s session build, so there
    is no upside left to chase from CoreML here, and no reason to reopen that risk.
    """
    global _session
    if _session is None:
        from rembg import new_session

        _session = new_session(MODEL, providers=["CPUExecutionProvider"])
    return _session


def _checksum(rgb: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS masks (key TEXT PRIMARY KEY, png BLOB)")
    return c


def _cache_get(key: str) -> np.ndarray | None:
    with _conn() as c:
        row = c.execute("SELECT png FROM masks WHERE key = ?", (key,)).fetchone()
    if not row:
        return None
    return np.array(Image.open(io.BytesIO(row[0])).convert("L"))


def _cache_put(key: str, alpha: np.ndarray) -> None:
    buf = io.BytesIO()
    Image.fromarray(alpha).save(buf, format="PNG")
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO masks (key, png) VALUES (?, ?)", (key, buf.getvalue()))


def mask(rgb: np.ndarray, thresh: int = 128) -> np.ndarray | None:
    """Binary foreground mask (255 = product, 0 = background), or None if rembg is
    unavailable. The raw 0-255 alpha matte is cached; the threshold is applied on read, so
    the same cached matte can be re-thresholded without re-running the model."""
    if not available():
        return None

    key = f"{_checksum(rgb)}:{MODEL}"
    alpha = _cache_get(key)
    if alpha is None:
        from rembg import remove

        out = remove(rgb, session=_get_session(), only_mask=True, post_process_mask=True)
        alpha = np.asarray(out)
        if alpha.ndim == 3:  # some models hand back HxWx1
            alpha = alpha[..., 0]
        alpha = alpha.astype(np.uint8)
        _cache_put(key, alpha)

    return ((alpha >= thresh) * 255).astype(np.uint8)
