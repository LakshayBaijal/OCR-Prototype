"""The router - decides how little we can spend and still read the image.

Order is strictly cheapest-first. A paid tier is touched only when every free option has
been exhausted, and only if its credentials exist:

    0  cache        free   never read the same pixels twice
    1  rapidocr     free   text + boxes + confidence
    2  local_vlm    free   better text, no geometry
    3  google       PAID   disabled without GOOGLE_APPLICATION_CREDENTIALS
    4  claude       PAID   disabled without ANTHROPIC_API_KEY

`allow_paid` defaults to False. Paying is opt-in, never a default, so no run can silently
cost money.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import cache, claude_vision, google_vision, local_vlm, rapid
from .types import OCRResult


@dataclass
class Policy:
    min_confidence: float = 0.65   # mean per-line confidence below this is a weak read
    min_chars: int = 40            # a near-empty read is a failed read
    allow_paid: bool = False       # opt-in. Nothing bills unless this is explicitly True.

    # Needs the MLX server running (see commands.txt). If it is not up, the tier is skipped.
    use_local_vlm: bool = False

    # Run BOTH free engines and merge their text, instead of using the VLM only as a rescue.
    # They fail on different images and neither dominates: on the hard set RapidOCR found 6
    # ground-truth fields to the VLM's 5, but the VLM pulled 2.4x more text and rescued
    # images RapidOCR read as literally 0 characters. Both are free and latency does not
    # matter here, so taking the union strictly dominates picking either one.
    union_free_tiers: bool = False

    # OFF by default, and that is deliberate. "No digits" looks like a missed nutrition
    # panel, but the FRONT of a pack legitimately has none - a tea box reading
    # "Brooke Bond / TAJ MAHAL / Spicy Ginger" at confidence 1.0 is a perfect read, and
    # escalating it would burn a tier (and later, real money) to find digits that do not
    # exist. The digit check only makes sense across a product's whole image set, which is
    # the evaluator's job, not the router's.
    require_digits: bool = False


def weakness(result: OCRResult, policy: Policy) -> str | None:
    """Why this read should not be trusted - or None if it is good enough to keep."""
    if not result.lines:
        return "no text found at all"

    if result.char_count < policy.min_chars:
        return f"only {result.char_count} chars recovered (< {policy.min_chars})"

    conf = result.mean_confidence
    if conf is not None and conf < policy.min_confidence:
        return f"mean confidence {conf:.2f} < {policy.min_confidence}"

    if policy.require_digits and result.digit_lines == 0:
        return "no digits anywhere (MRP/FSSAI/quantity panel likely missed)"

    return None


def _cached_or_run(rgb: np.ndarray, engine_name: str, fn) -> OCRResult:
    hit = cache.get(rgb, engine_name)
    if hit is not None:
        return hit
    result = fn(rgb)
    cache.put(rgb, result)
    return result


def read(rgb: np.ndarray, policy: Policy | None = None) -> OCRResult:
    """Read an image for the least possible cost. Returns the best result plus an audit
    trail explaining every escalation."""
    policy = policy or Policy()
    trail: list[str] = []

    # --- Tier 1: free, and the only source of boxes/confidence.
    best = _cached_or_run(rgb, "rapidocr", rapid.run)
    trail.append(
        f"rapidocr{' (cached)' if best.cached else ''}: {len(best.lines)} lines, "
        f"{best.char_count} chars, conf={best.mean_confidence and round(best.mean_confidence, 2)}"
    )

    # --- Union mode: both free engines on every image, texts merged. RapidOCR contributes
    # the geometry and the fields it is good at; the VLM contributes the text RapidOCR
    # cannot see. Neither is discarded.
    if policy.union_free_tiers and policy.use_local_vlm and local_vlm.available():
        vlm = _cached_or_run(rgb, "local_vlm", local_vlm.run)
        trail.append(f"local_vlm{' (cached)' if vlm.cached else ''}: {vlm.char_count} chars (merged)")
        merged = OCRResult(
            lines=best.lines + vlm.lines,
            engine="rapidocr+local_vlm",
            elapsed=best.elapsed + vlm.elapsed,
            cost_usd=0.0,
            cached=best.cached and vlm.cached,
        )
        merged.trail = trail + ["accepted: union of both free engines"]
        return merged

    why = weakness(best, policy)
    if why is None:
        best.trail = trail + ["accepted: free tier sufficient"]
        return best
    trail.append(f"ESCALATE - {why}")

    # --- Tier 2: still free. Costs only local GPU time, which we have decided is worthless.
    if policy.use_local_vlm and local_vlm.available():
        vlm = _cached_or_run(rgb, "local_vlm", local_vlm.run)
        trail.append(
            f"local_vlm{' (cached)' if vlm.cached else ''}: {len(vlm.lines)} lines, "
            f"{vlm.char_count} chars"
        )
        # A VLM has no confidence to compare, so we judge it on how much it recovered.
        if vlm.char_count > best.char_count:
            trail.append("local_vlm recovered more text; keeping it (still free)")
            best, why = vlm, weakness(vlm, policy)
            if why is None:
                best.trail = trail + ["accepted: free tier sufficient"]
                return best
            trail.append(f"ESCALATE - {why}")
        else:
            trail.append("local_vlm did not improve on rapidocr")
    elif policy.use_local_vlm:
        trail.append("local_vlm unavailable (needs Apple Silicon + mlx-vlm); skipped")

    # --- Everything below here costs money.
    if not policy.allow_paid:
        best.trail = trail + ["STOP: paid tiers disabled (allow_paid=False); keeping best free result"]
        return best

    if google_vision.available():
        g = _cached_or_run(rgb, "google_vision", google_vision.run)
        trail.append(f"google_vision{' (cached)' if g.cached else ''}: {g.char_count} chars, ${g.cost_usd}")
        if weakness(g, policy) is None:
            g.trail = trail + ["accepted: google_vision"]
            return g
        best = g if g.char_count > best.char_count else best
    else:
        trail.append("google_vision unavailable (no credentials); skipped")

    if claude_vision.available():
        c = _cached_or_run(rgb, f"claude:{claude_vision.CHEAP_MODEL}", claude_vision.run)
        trail.append(f"claude{' (cached)' if c.cached else ''}: {c.char_count} chars, ${c.cost_usd:.4f}")
        best = c if c.char_count > best.char_count else best
    else:
        trail.append("claude unavailable (no API key); skipped")

    best.trail = trail + ["accepted: best available"]
    return best
