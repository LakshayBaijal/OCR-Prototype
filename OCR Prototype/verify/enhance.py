"""The thing being verified: a rough product photo redrawn as a studio image.

Gemini's native image-generation model takes the photo in and hands a new image back
- white background, corrected lighting, the product upright and centred. It is worth
being precise about what that means, because it is the entire reason the rest of this
package exists: this is not a filter. Nothing here adjusts pixels. The model looks at
the photo, forms an understanding of it, and DRAWS A NEW IMAGE from that
understanding. Every glyph, every digit, every regulatory mark in the output was
re-rendered from what the model believed it saw.

So "preserve the product exactly" is an instruction, not a guarantee, and the prompt
below leans on it hard - the last three requirements exist purely to constrain
invention. They reduce the failure rate; they cannot eliminate it, which is why the
output goes to `verify.report` rather than straight into a catalog.

A classical CV pass runs first (`preprocess`), and it is genuinely a filter: EXIF
rotation, blur-adaptive sharpening, CLAHE, Lanczos upscale, all deterministic, no
model involved. It exists to hand the generator a cleaner image to read, on the
theory that a model that can see the fine print is less likely to invent it.
"""

from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "gemini-3.1-flash-image"   # "Nano Banana 2"
LITE_MODEL = "gemini-3.1-flash-lite-image"

BLUR_SHARPEN_THRESHOLD = 100.0   # variance-of-Laplacian below this reads as "blurry"
UPSCALE_TARGET_LONG_SIDE = 1600  # px; only ever upscales

PROMPT = (
    "Transform this rough product photo into a professional studio-quality "
    "e-commerce product image. Requirements:\n"
    "- Focus on the single product that is the primary subject of the photo. "
    "Ignore the background and any other products visible behind or around it.\n"
    "- If the photo shows multiple products with no single one clearly held or "
    "otherwise the obvious subject (e.g. a direct photo of a shelf containing "
    "several different products), identify the product that is most centrally "
    "positioned according to the image's geometry and treat only that one as "
    "the subject — ignore every other product in the frame.\n"
    "- The final image must contain ONLY that product — nothing else. If the "
    "photo shows a hand holding the product, or the product sitting on a shelf "
    "or other surface, remove the hand, shelf, and every other surface or "
    "background object entirely; do not let any part of them remain in the "
    "output.\n"
    "- Show exactly ONE unit of the product, even if the original photo shows "
    "multiple quantities (e.g. a pack, a stack, or several units together).\n"
    "- Pure clean white background (#FFFFFF), with at most a subtle, natural "
    "soft shadow beneath the product.\n"
    "- Correct the product's orientation so it is upright and squarely framed, "
    "centered in the image.\n"
    "- Even, professional studio lighting — remove harsh shadows, glare, and "
    "color casts from the original photo.\n"
    "- Enhance sharpness, resolution, and overall image quality.\n"
    "- Preserve the product itself EXACTLY as shown — its shape, color, "
    "printed text, logos, and every visible detail must remain completely "
    "unchanged. Do not invent, remove, or alter any feature of the product.\n"
    "- Do not add any new text, watermark, or graphic overlay."
)


@dataclass
class EnhancementResult:
    image: np.ndarray | None      # RGB
    model: str
    preprocess_time_s: float
    gemini_time_s: float
    blur_score: float | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.image is not None


def api_key_available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())


# ---------------------------------------------------------------- classical pass


def _blur_score(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _sharpen(rgb: np.ndarray, amount: float, sigma: float = 1.0) -> np.ndarray:
    """Unsharp mask: sharpened = original*(1+amount) - blurred*amount."""
    blurred = cv2.GaussianBlur(rgb, (0, 0), sigma)
    return cv2.addWeighted(rgb, 1 + amount, blurred, -amount, 0)


def _clahe(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)


def _upscale(rgb: np.ndarray, target: int = UPSCALE_TARGET_LONG_SIDE) -> np.ndarray:
    h, w = rgb.shape[:2]
    long_side = max(h, w)
    if long_side >= target:
        return rgb
    scale = target / long_side
    return cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)


def preprocess(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """Blur-adaptive sharpen -> CLAHE -> Lanczos upscale. Returns (image, blur_score).

    Sharpening strength is driven by the measured blur rather than fixed: a photo that
    is already crisp gets a light touch, one that is soft gets more. A single fixed
    amount would either under-serve the blurry inputs or put halos on the sharp ones.
    """
    blur = _blur_score(rgb)
    out = _sharpen(rgb, amount=1.2 if blur < BLUR_SHARPEN_THRESHOLD else 0.5)
    out = _clahe(out)
    out = _upscale(out)
    return out, blur


# ---------------------------------------------------------------- the model call


def enhance(
    rgb: np.ndarray, model: str = DEFAULT_MODEL, use_preprocess: bool = True
) -> EnhancementResult:
    """One image in, one image out. Returns the result rather than raising, so a
    failed enhancement degrades into a visible error in the UI instead of a traceback.
    """
    started = time.perf_counter()
    blur: float | None = None
    if use_preprocess:
        rgb, blur = preprocess(rgb)
    pre_time = time.perf_counter() - started

    if not api_key_available():
        return EnhancementResult(None, model, pre_time, 0.0, blur,
                                 "GEMINI_API_KEY is not set in OCR Prototype/.env")

    gem_started = time.perf_counter()
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"].strip())
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=95)
        part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")

        response = client.models.generate_content(
            model=model, contents=[PROMPT, part],
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )
    except Exception as exc:
        return EnhancementResult(None, model, pre_time, time.perf_counter() - gem_started,
                                 blur, f"{type(exc).__name__}: {exc}")

    gem_time = time.perf_counter() - gem_started
    candidates = response.candidates or []
    if not candidates or not candidates[0].content or not candidates[0].content.parts:
        return EnhancementResult(None, model, pre_time, gem_time, blur,
                                 "the model returned no content (it may have refused)")

    for part in candidates[0].content.parts:
        if part.inline_data and part.inline_data.data:
            img = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
            return EnhancementResult(np.asarray(img), model, pre_time, gem_time, blur)

    return EnhancementResult(None, model, pre_time, gem_time, blur,
                             "the model replied but returned no image")
