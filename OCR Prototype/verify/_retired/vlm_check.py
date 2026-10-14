"""The catch-all: show Gemini both images at once and ask what changed.

Everything else in this package checks a field we thought to name. This module exists
for the failures we did not think to name - and on a generative enhancer those are the
majority. The field comparison cannot tell you that the model swapped a mango pack for
a strawberry one when both read "Real Fruit Juice", that it redrew the bottle cap the
wrong colour, that it flattened an embossed seal into a printed one, or that it turned
two units into one. Those are all visual facts about the product, and none of them
live in a text field.

Design decisions worth stating, because each one is load-bearing:

  * BOTH images go in ONE call, labelled A and B. Two separate calls, each describing
    one image, would force us to diff two free-text descriptions - and two
    descriptions of the same pack legitimately differ in wording, so that diff is
    mostly noise. Asking one model to compare two images it can see simultaneously
    turns a text-similarity problem into a perception problem, which is what it is.

  * The prompt asks for DIFFERENCES, never for quality. "Is B a good product image"
    is a question the model will happily answer well about a corrupted image - it was
    trained to like clean white backgrounds. The only question here is what changed.

  * Cosmetic changes are explicitly excluded in the prompt. B is SUPPOSED to have a
    white background, better lighting, no hand holding it, and a straightened
    product. A model told to list every difference will dutifully report all of that,
    burying the one real finding under six intended ones.

  * The model is told to say "unclear" rather than guess. A hedged answer routes to a
    human; a confident wrong answer routes to a catalog.

This is a strong check, not a sound one: it is another generative model, susceptible
to the same confident invention it is being used to detect. It carries the same weight
as the classical checks in `verify.report` - a finding here raises a flag for review,
it does not by itself reject an image.

Two backends implement the same prompt and the same `VLMComparison` result:

  * `compare_images_local`  - `mlx_vlm.server` running Gemma-3-4b-it-4bit on this
    machine (see ocr/local_vlm.py). No network call leaves the machine, no per-call
    cost. This is the default for `verify_app.py` and `verify_batch.py`.
  * `compare_images`        - the real Gemini API. Kept as a same-project quality
    reference and for `verify_batch.py --visual-engine gemini`.

`compare_images_local` reaches into `ocr.local_vlm`'s private `_chat_json_with_retry`
and `_data_uri` rather than its public `run_structured` - deliberately, and for one
reason: `run_structured` always puts every image first and appends ONE trailing text
block, so there is no way to interleave "IMAGE A:" / "IMAGE B:" labels between the two
images through that function. Two unlabelled images and a single prompt asking about
"image A" and "image B" leaves the model to guess an ordering convention it was never
actually told; the interleaved labels this module relies on remove that ambiguity
entirely, and `stage2/gemini_direct.py` already reaches into `vlm_direct`'s private
helpers for the identical reason (shared prompt, no public seam for it) - a precedent
already established in this project, not a new pattern.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

_SYSTEM = (
    "You compare two photographs of what should be the SAME retail product. IMAGE A is "
    "an original photograph. IMAGE B was generated from IMAGE A by an AI image model "
    "that was instructed to produce a clean studio product shot while preserving the "
    "product exactly.\n"
    "\n"
    "Your job is to find where IMAGE B departs from IMAGE A in ways that CHANGE THE "
    "FACTS about the product. You are an auditor, not a critic: never comment on which "
    "image looks better.\n"
    "\n"
    "IGNORE these differences entirely - they are the intended effect of the "
    "enhancement, not defects:\n"
    "- background replaced with plain white; background objects, shelves, other "
    "products, or a hand holding the item removed\n"
    "- lighting, exposure, colour cast, glare or shadow changes\n"
    "- the product being rotated upright, re-centred, re-cropped, or made larger\n"
    "- overall sharpness or resolution improvements\n"
    "- text in A that was blurry becoming legible in B, PROVIDED the legible text is "
    "consistent with what is visible in A\n"
    "\n"
    "REPORT these, they are real defects:\n"
    "- any printed text that differs: brand, product name, price/MRP, weight or "
    "volume, licence numbers, dates, ingredients, warnings, any wording at all\n"
    "- text, badges, seals, logos or graphics present in B that are NOT in A "
    "(invention), or present in A and missing from B (loss)\n"
    "- the veg/non-veg mark (small square outline containing a filled circle - green "
    "for vegetarian, brown/maroon for non-vegetarian) being added, removed, moved, or "
    "changed in colour\n"
    "- a barcode that is present in A but missing, blurred beyond scanning, or "
    "structurally different in B\n"
    "- the product itself being a DIFFERENT product, a different variant, flavour, or "
    "colourway than the one in A\n"
    "- a different NUMBER of units shown, or a different package format\n"
    "- the package's shape, proportions, cap/lid, or closure being altered\n"
    "\n"
    "Be specific and quote the actual text when you report a text difference. If you "
    "genuinely cannot tell whether something changed - because A is too blurry, or the "
    "detail is not visible in one of them - say so using the string \"unclear\" rather "
    "than guessing. A hedged answer is useful; a confident wrong answer is not.\n"
    "Respond with ONLY a JSON object, no markdown fence and no commentary."
)

_USER = (
    "Compare IMAGE A (original) with IMAGE B (AI-enhanced). Return this JSON object:\n"
    "{\n"
    '  "same_product": true | false | "unclear",\n'
    '  "same_variant": true | false | "unclear",\n'
    '  "text_differences": [ {"field": "<what it is, e.g. MRP / brand / net weight>", '
    '"in_a": "<text as printed in A, or \'not legible\'>", "in_b": "<text as printed in '
    'B>", "severity": "high" | "medium" | "low"} ],\n'
    '  "added_in_b": [ "<element visible in B but not in A>" ],\n'
    '  "missing_from_b": [ "<element visible in A but not in B>" ],\n'
    '  "veg_mark_in_a": "green" | "brown" | "none" | "unclear",\n'
    '  "veg_mark_in_b": "green" | "brown" | "none" | "unclear",\n'
    '  "barcode_in_a": "present" | "absent" | "unclear",\n'
    '  "barcode_in_b": "present" | "absent" | "unclear",\n'
    '  "unit_count_in_a": <integer or "unclear">,\n'
    '  "unit_count_in_b": <integer or "unclear">,\n'
    '  "verdict": "pass" | "review" | "reject",\n'
    '  "reasoning": "<two sentences at most, explaining the verdict>"\n'
    "}\n"
    "Use empty arrays when there is nothing to report. Set verdict to \"pass\" only if "
    "every factual detail you can check agrees between the two images; \"reject\" if "
    "the product identity, a price, a code, or the veg/non-veg mark differs; "
    "\"review\" for anything you could not resolve."
)


def _norm(s) -> str:
    """Whitespace/case-insensitive compare key, so "1 kg" and "1  KG" are one value."""
    return re.sub(r"\s+", " ", str(s if s is not None else "").strip().lower())


@dataclass
class VLMComparison:
    ok: bool
    raw: dict = field(default_factory=dict)
    model: str = ""
    elapsed: float = 0.0
    error: str | None = None

    # -- convenience accessors, each tolerating the model omitting a key --------

    @property
    def verdict(self) -> str:
        return str(self.raw.get("verdict", "review")).lower()

    @property
    def text_differences(self) -> list[dict]:
        """Reported text differences, with the model's NON-differences filtered out.

        Both backends regularly emit entries whose `in_a` and `in_b` are the same
        string - "Net Quantity 1 kg" -> "Net Quantity 1 kg", marked severity "high".
        Measured on a 6-pair audit: 5 of 9 high-severity entries were this. It is the
        model narrating what it checked rather than what changed, and it is not
        recoverable by prompting (the same run's own `verdict` was "pass" with the
        reasoning "All text elements are identical", while these entries said
        otherwise - the model is contradicting itself within one response, so there is
        no wording that reliably suppresses it).

        This matters beyond tidiness because `report._decide` demotes to REJECT on any
        high-severity text difference: unfiltered, one of these self-cancelling entries
        alone flipped a genuinely clean pair (img40) from PASS to REJECT. A "difference"
        where both sides read identically is definitionally not a difference, so it is
        dropped here, at the single point every consumer reads through.
        """
        out = []
        for d in self.raw.get("text_differences") or []:
            if not isinstance(d, dict):
                continue
            a, b = _norm(d.get("in_a")), _norm(d.get("in_b"))
            if a == b:
                continue  # the model reported a field it checked, not one that changed
            out.append(d)
        return out

    @property
    def added(self) -> list[str]:
        return [str(x) for x in (self.raw.get("added_in_b") or [])]

    @property
    def missing(self) -> list[str]:
        return [str(x) for x in (self.raw.get("missing_from_b") or [])]

    @property
    def reasoning(self) -> str:
        return str(self.raw.get("reasoning", ""))

    def high_severity(self) -> list[dict]:
        return [d for d in self.text_differences
                if str(d.get("severity", "")).lower() == "high"]

    def identity_broken(self) -> bool:
        """Did the model say B is not the same product/variant as A? `unclear` is not
        a yes - it routes to review through the verdict, not to an outright reject."""
        return self.raw.get("same_product") is False or self.raw.get("same_variant") is False


# Confirmed against `client.models.list()` on this project's key, which is the only
# reliable source - there is no `gemini-3.1-flash`, and asking for one 404s. A
# non-lite model is the right default here even though `stage2/gemini_direct.py`
# defaults to lite for extraction: comparing two images for subtle text corruption is
# a markedly harder perception task than reading one image's fields, and the lite
# model was measurably vaguer on it (it reported "significant gibberish" as one
# finding where the full model quoted the actual corrupted strings).
DEFAULT_MODEL = "gemini-3.5-flash"


def compare_images_local(original_rgb, enhanced_rgb) -> "VLMComparison":
    """The local-MLX analogue of `compare_images` below - same prompts, same result
    shape, no network call. See this module's docstring for why it builds the request
    by hand instead of calling `ocr.local_vlm.run_structured` directly.
    """
    import time

    from ocr import local_vlm

    started = time.perf_counter()
    if not local_vlm.available():
        return VLMComparison(
            False, model=f"local_vlm:{local_vlm.MODEL}",
            error=(f"the MLX server is not reachable at {local_vlm.HOST} - start it with "
                  f"`venv/bin/python -m mlx_vlm.server --model {local_vlm.MODEL} "
                  f"--port 8080 --kv-bits 8 --max-tokens 512`"),
        )

    content = [
        {"type": "text", "text": "IMAGE A (the original photograph):"},
        {"type": "image_url", "image_url": {"url": local_vlm._data_uri(original_rgb)}},
        {"type": "text", "text": "IMAGE B (the AI-enhanced version of image A):"},
        {"type": "image_url", "image_url": {"url": local_vlm._data_uri(enhanced_rgb)}},
        {"type": "text", "text": _USER},
    ]

    try:
        parsed = local_vlm._chat_json_with_retry(content, _SYSTEM, max_retries=3)
    except Exception as exc:
        return VLMComparison(False, model=f"local_vlm:{local_vlm.MODEL}",
                             elapsed=time.perf_counter() - started,
                             error=f"{type(exc).__name__}: {exc}")

    elapsed = time.perf_counter() - started
    model_name = f"local_vlm:{local_vlm.served_model() or local_vlm.MODEL}"
    if parsed is None:
        return VLMComparison(
            False, model=model_name, elapsed=elapsed,
            error=("the local VLM returned no parseable JSON after retries - a small "
                  "4-bit model asked for this much structured output at once can "
                  "degenerate (see ocr/local_vlm.py); the Gemini backend is more "
                  "reliable here if billing is available"),
        )
    return VLMComparison(True, raw=parsed, model=model_name, elapsed=elapsed)


def compare_images(
    original_jpeg: bytes, enhanced_jpeg: bytes, model: str = DEFAULT_MODEL,
) -> VLMComparison:
    """One call, both images. Returns a result object rather than raising."""
    started = time.perf_counter()

    from stage2 import gemini_direct

    if not gemini_direct.api_key_available():
        return VLMComparison(False, error="GEMINI_API_KEY is not set in OCR Prototype/.env")

    try:
        from google.genai import types

        client = gemini_direct._client()
        contents = [
            "IMAGE A (the original photograph):",
            types.Part.from_bytes(data=original_jpeg, mime_type="image/jpeg"),
            "IMAGE B (the AI-enhanced version of image A):",
            types.Part.from_bytes(data=enhanced_jpeg, mime_type="image/jpeg"),
            _USER,
        ]
        config = types.GenerateContentConfig(
            system_instruction=_SYSTEM, response_mime_type="application/json",
            temperature=0.0,
        )
        resp = client.models.generate_content(model=model, contents=contents, config=config)
    except Exception as exc:
        return VLMComparison(False, model=model, elapsed=time.perf_counter() - started,
                             error=f"{type(exc).__name__}: {exc}")

    elapsed = time.perf_counter() - started
    text = (resp.text or "").strip()
    if not text:
        return VLMComparison(False, model=model, elapsed=elapsed,
                             error="the model returned an empty response")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return VLMComparison(False, model=model, elapsed=elapsed,
                             error=f"the model returned non-JSON: {text[:200]}")
    if not isinstance(parsed, dict):
        return VLMComparison(False, model=model, elapsed=elapsed,
                             error="the model returned JSON that was not an object")

    return VLMComparison(True, raw=parsed, model=model, elapsed=elapsed)
