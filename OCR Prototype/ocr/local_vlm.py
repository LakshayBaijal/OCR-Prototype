"""Tier 2 - local VLM, served by MLX. FREE (runs on this machine's GPU), no API, no tokens.

Talks to an `mlx_vlm.server` over HTTP rather than importing the model in-process. That is
deliberate, and it is the difference between a toy and something usable:

  * The weights load ONCE, into the server, and stay resident. Importing mlx_vlm directly
    meant every script paid a ~130s cold load, and every process carried 3-4GB of weights
    in its own memory - which is how the editor got starved and crashed.
  * The server keeps a vision cache and quantised KV cache across requests, so repeat
    calls are far cheaper than a fresh in-process generate().
  * Our benchmark, the Streamlit app and the router all share one loaded model instead of
    each spawning their own copy.

Start it (once, in its own terminal):

    venv/bin/python -m mlx_vlm.server \
        --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit \
        --port 8080 --kv-bits 8 --max-tokens 512

If the server is not running, `available()` is False and the router simply skips this tier -
it never blocks the free RapidOCR path.
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from .types import OCRLine, OCRResult

HOST = "http://127.0.0.1:8080"

# The model to REQUEST. mlx_vlm.server supports several cached models and lazy-swaps
# between them per-request (unload the current one, load the requested one) - so this is
# not "whatever the server happened to start with", it is a genuine choice, and changing it
# is enough to switch models with zero other code changes.
#
# The entire Qwen-VL family is BROKEN under mlx_vlm.server: both Qwen2.5-VL-3B and
# Qwen3-VL-4B crash mid-generation with "RuntimeError: There is no Stream(gpu, N) in current
# thread", from the identical get_rope_index() code path in qwen2_5_vl/language.py and
# qwen3_vl/language.py - a shared bug in mlx-vlm's threading model, not one model version.
# Confirmed on both; do not retry Qwen-VL here until mlx-vlm fixes it upstream.
# Gemma-3 is the only VLM family confirmed to serve correctly on this setup.
MODEL = "mlx-community/gemma-3-4b-it-4bit"

PROMPT = (
    "Read all text printed on this product package. "
    "Output only the text you see, one item per line. No commentary."
)


def served_model() -> str | None:
    """The model this adapter will REQUEST (not "whatever the server advertises" - /v1/models
    lists every model ever cached to disk, since mlx_vlm.server supports lazy-swapping
    between several; it is not a live "currently loaded" indicator). Returns None only if
    the server itself is unreachable.
    """
    try:
        with urllib.request.urlopen(f"{HOST}/v1/models", timeout=2) as r:
            json.loads(r.read())  # just prove the server responds; ignore its contents
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    return MODEL


def available() -> bool:
    """True only if the MLX server is actually up. No server, no tier - never an exception."""
    return served_model() is not None


def _data_uri(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def run(rgb: np.ndarray) -> OCRResult:
    if not available():
        raise RuntimeError(
            "MLX server is not running. Start it with:\n"
            "  venv/bin/python -m mlx_vlm.server "
            f"--model {MODEL} --port 8080 --kv-bits 8 --max-tokens 512"
        )

    payload = {
        "model": served_model() or MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _data_uri(rgb)}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        # Greedy decoding makes this 4-bit model degenerate - on one pack it emitted
        # '75555555...' until it hit the token cap, on another just '1'. A little temperature
        # plus a repetition penalty breaks those loops; the same pack then reads back
        # correctly as "Limca Lime 'n' Lemon".
        "temperature": 0.3,
        "repetition_penalty": 1.05,
        "max_tokens": 512,
    }

    req = urllib.request.Request(
        f"{HOST}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    elapsed = time.perf_counter() - t0

    text = body["choices"][0]["message"]["content"].strip()
    h, w = rgb.shape[:2]

    # A VLM returns prose, not geometry: we know WHAT it read, not WHERE. One box for the
    # whole crop is the honest answer - and it is why this tier cannot replace RapidOCR,
    # which is our only source of per-line boxes and confidences.
    lines = [
        OCRLine(text=t, bbox=[(0, 0), (w, 0), (w, h), (0, h)], confidence=None)
        for t in text.splitlines() if t.strip()
    ]
    return OCRResult(lines=lines, engine="local_vlm", elapsed=elapsed, cost_usd=0.0)
