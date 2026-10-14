"""Run every OCR-only check over one before/after pair and reduce it to a verdict.

No VLM, no LLM, nothing generative. Everything here is either deterministic (regex,
barcode decode, set arithmetic, feature matching) or a task-specific recogniser
(RapidOCR). Run it twice on the same pair and it returns the same answer - which is the
property that makes the output usable as evidence, and the property the generative path
never had.

The check set, most trustworthy first:

    barcode         decode + checksum, both sides            REJECT-capable
    digits          every >=3-digit run, as a set            REJECT-capable
    quantities      every number+unit, canonicalised         REJECT-capable
    labelled fields FSSAI / MRP / net qty / pack qty         REJECT-capable
    prices          every currency-marked amount             review
    prominent text  brand-ish lines, containment test        review
    words           overall token retention / invention      review
    artwork         ORB inlier ratio                         review
    colour          hue-histogram correlation                review

WHAT CHANGED FROM THE PREVIOUS DESIGN, and why the verdict should be trusted more now:
nothing in this pipeline has to decide which line of text is "the Brand". The old
design's noise came almost entirely from that guess (`stage2/heuristic.py`'s
`layout:largest-text`, its own confidence 0.4), not from OCR misreading characters. The
sets in `verify.signals` compare what is present without attributing it, and the only
NAMED fields left are the four the pack labels explicitly (`verify.labeled`).

Three verdicts, for the same reason as before: an image where OCR read nothing has not
passed, and an image whose enhanced version is merely more legible has not failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field

import cv2
import numpy as np

from . import artwork, extract, labeled, signals
from .signals import SetDiff

PASS, REVIEW, REJECT = "PASS", "REVIEW", "REJECT"

# A digit run or quantity present on one side and not the other is the strongest
# evidence this tool produces, but OCR does drop a line on a bad crop - so a SINGLE
# lost item is a review, and it takes either an invention (text that was not there) or
# a co-occurring loss+invention (a value that was REPLACED) to reject. That asymmetry
# is deliberate: losing "500 g" is usually OCR missing it; "500 g" becoming "158 g" is
# the enhancer rewriting it.
WORD_RETENTION_REVIEW = 0.55
WORD_INVENTION_REVIEW = 0.45


@dataclass
class Check:
    """One named check, rendered as a row. `advisory` caps it at REVIEW."""

    name: str
    status: str          # MATCH | CHANGED | LOST | ADDED | ABSENT
    detail: str
    original: str = ""
    enhanced: str = ""
    critical: bool = False
    advisory: bool = False


@dataclass
class VerificationReport:
    checks: list[Check] = _dc_field(default_factory=list)
    verdict: str = REVIEW
    reasons: list[str] = _dc_field(default_factory=list)

    original_reading: extract.ImageReading | None = None
    enhanced_reading: extract.ImageReading | None = None
    original_crop_note: str = ""
    enhanced_crop_note: str = ""
    diffs: dict[str, SetDiff] = _dc_field(default_factory=dict)
    artwork: artwork.ArtworkSimilarity | None = None
    engine: str = "rapidocr"

    def counts(self) -> dict[str, int]:
        out = {k: 0 for k in ("MATCH", "CHANGED", "LOST", "ADDED", "ABSENT")}
        for c in self.checks:
            out[c.status] = out.get(c.status, 0) + 1
        return out

    def reader_errors(self) -> list[str]:
        out = []
        for reading, side in ((self.original_reading, "Original"),
                              (self.enhanced_reading, "Enhanced")):
            if reading is not None and reading.error:
                out.append(f"{side}: {reading.error}")
        return out


def _fmt(items, limit: int = 6) -> str:
    """A readable, bounded rendering of a set for the UI."""
    if not items:
        return "—"
    shown = sorted(str(i) for i in items)[:limit]
    more = len(items) - len(shown)
    return ", ".join(shown) + (f" (+{more} more)" if more > 0 else "")


def _fmt_measures(items, limit: int = 6) -> str:
    if not items:
        return "—"
    shown = sorted(f"{v:g} {u}" for u, v in items)[:limit]
    more = len(items) - len(shown)
    return ", ".join(shown) + (f" (+{more} more)" if more > 0 else "")


# --------------------------------------------------------------------------- #
# individual checks
# --------------------------------------------------------------------------- #


def _barcode_check(o_reading, e_reading) -> Check:
    """Decode, not read. A barcode smoothed into unscannable bars fails this even when
    the digits printed underneath still OCR correctly."""
    o, e = o_reading.barcode, e_reading.barcode
    o_code = o.code if o and o.code else None
    e_code = e.code if e and e.code else None

    if not o_code and not e_code:
        return Check("Barcode", "ABSENT", "no barcode decoded from either image",
                     critical=True)
    if o_code and not e_code:
        return Check("Barcode", "LOST",
                     f"the original decodes to {o_code} but the enhanced image will not "
                     f"scan at all - the bars were destroyed", o_code, "—", critical=True)
    if e_code and not o_code:
        return Check("Barcode", "ADDED",
                     f"the enhanced image decodes to {e_code} but the original would not "
                     f"scan - verify against the physical pack", "—", e_code, critical=True)
    if o_code != e_code:
        return Check("Barcode", "CHANGED",
                     f"the two images decode to DIFFERENT products", o_code, e_code,
                     critical=True)
    checksum = "checksum valid" if (e and e.valid_checksum) else "checksum INVALID"
    return Check("Barcode", "MATCH", f"both images decode identically ({checksum})",
                 o_code, e_code, critical=True)


# A set can be "changed" two very different ways, and they deserve different verdicts.
# A dense nutrition panel rewritten wholesale (img13: 14 of 50 numbers survived) is the
# enhancer destroying the panel. One number differing out of five, on print small enough
# that OCR is shaky on BOTH sides, is more likely the reader than the image. So a
# replacement rejects only when retention falls below this; above it, it asks for a
# human. Calibrated on the audit set, not a labelled study - the most worthwhile number
# in this file to revisit once there is ground truth.
WHOLESALE_REWRITE_RETENTION = 0.6


def _set_check(name: str, diff: SetDiff, critical: bool, fmt=_fmt,
               advisory: bool = False) -> Check:
    """Turn a SetDiff into a row. LOST and ADDED are kept distinct because they have
    different causes: lost is usually OCR missing a line, added is usually the enhancer
    rendering something that was not there."""
    if diff.original_count == 0 and diff.enhanced_count == 0:
        return Check(name, "ABSENT", f"no {diff.label} found in either image",
                     critical=critical, advisory=advisory)
    if diff.lost and diff.invented:
        status = "CHANGED"
        retention = diff.retention if diff.retention is not None else 0.0
        wholesale = retention < WHOLESALE_REWRITE_RETENTION
        detail = (f"{diff.label} were replaced: {fmt(diff.lost)} is gone and "
                  f"{fmt(diff.invented)} appeared instead. {diff.summary()}")
        if not wholesale:
            # Most of the set survived, so this is a few items differing rather than a
            # rewritten panel - the regime where OCR noise on fine print is the more
            # likely explanation. Downgrade to advisory so it reviews instead of rejects.
            advisory = True
            detail += (f" — but {retention:.0%} of them did survive, so this may be OCR "
                      f"noise on small print rather than the image changing")
    elif diff.invented:
        status = "ADDED"
        detail = (f"{diff.label} appear in the enhanced image that are not in the "
                  f"original: {fmt(diff.invented)}. {diff.summary()}")
    elif diff.lost:
        status = "LOST"
        detail = f"{diff.label} missing from the enhanced image: {fmt(diff.lost)}. {diff.summary()}"
    else:
        status = "MATCH"
        detail = f"all {diff.original_count} {diff.label} survived"
    return Check(name, status, detail, fmt(diff.retained | diff.lost),
                 fmt(diff.retained | diff.invented), critical=critical, advisory=advisory)


def _labeled_checks(o_fields, e_fields) -> list[Check]:
    """The four label-gated fields. Compared exactly after OCR-confusion folding."""
    out = []
    for key in labeled.KEYS:
        name = labeled.LABELS[key]
        o, e = o_fields.get(key), e_fields.get(key)
        if o is None and e is None:
            out.append(Check(name, "ABSENT", "not labelled on either image", "—", "—",
                             critical=True))
            continue
        if o is None:
            out.append(Check(name, "ADDED",
                             f"labelled only on the enhanced image (evidence: {e.evidence!r})",
                             "—", e.value, critical=True))
            continue
        if e is None:
            out.append(Check(name, "LOST",
                             f"labelled on the original but not the enhanced image "
                             f"(evidence: {o.evidence!r})", o.value, "—", critical=True))
            continue

        same = signals.fold(o.value) == signals.fold(e.value)
        weak = min(o.confidence, e.confidence) < 0.5
        if same:
            out.append(Check(name, "MATCH", f"identical ({o.value})", o.value, e.value,
                             critical=True))
        else:
            detail = f"the labelled value differs"
            if key == "fssai":
                for tag, v in (("original", o.value), ("enhanced", e.value)):
                    if not labeled.fssai_plausible(v):
                        detail += (f"; the {tag} reading is not a well-formed 14-digit "
                                   f"licence number, so it is likely a mis-scan")
            out.append(Check(name, "CHANGED", detail, o.value, e.value,
                             critical=True, advisory=weak))
    return out


def _prominent_text_check(o_lines, e_text: str) -> Check:
    """Does the original's biggest text still appear in the enhanced image?

    This is the replacement for brand/product-name field comparison, and the reason it
    is phrased as containment: it never has to decide which prominent line was 'the
    brand'. It only asks whether each survived somewhere in the other image's text -
    a question OCR can answer, unlike the naming question that produced the old
    design's `COCO -> Kelloys` noise.
    """
    prominent = signals.prominent_lines(o_lines, top_n=5)
    if not prominent:
        return Check("Prominent text (brand/name)", "ABSENT",
                     "no prominent text lines found in the original")

    found, missing = [], []
    for line in prominent:
        ok, score = signals.contains_fuzzy(line, e_text)
        (found if ok else missing).append(f"{line!r}" + ("" if ok else f" ({score:.0%})"))

    if not missing:
        return Check("Prominent text (brand/name)", "MATCH",
                     f"all {len(found)} prominent lines from the original still appear "
                     f"in the enhanced image", _fmt(prominent), "all present")
    status = "LOST" if len(missing) < len(prominent) else "CHANGED"
    return Check("Prominent text (brand/name)", status,
                 f"{len(missing)} of {len(prominent)} prominent lines from the original "
                 f"are not found in the enhanced image: {', '.join(missing)}",
                 _fmt(prominent), f"{len(found)}/{len(prominent)} found")


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #


def _decide(report: VerificationReport) -> tuple[str, list[str]]:
    severity = {PASS: 0, REVIEW: 1, REJECT: 2}
    verdict, reasons = PASS, []

    def demote(to: str, why: str) -> None:
        nonlocal verdict
        reasons.append(why)
        if severity[to] > severity[verdict]:
            verdict = to

    for reading, side in ((report.original_reading, "original"),
                          (report.enhanced_reading, "enhanced")):
        if reading is not None and reading.error:
            demote(REVIEW, f"the {side} image could not be read — {reading.error}")

    for chk in report.checks:
        if chk.status == "CHANGED":
            if chk.critical and not chk.advisory:
                demote(REJECT, f"{chk.name}: {chk.detail}")
            else:
                demote(REVIEW, f"{chk.name}: {chk.detail}")
        elif chk.status in ("LOST", "ADDED") and chk.critical:
            demote(REVIEW, f"{chk.name}: {chk.detail}")

    # Nothing readable is not a pass. Checks alone (a barcode, an artwork score) do not
    # substitute for having read the pack's text.
    words = report.diffs.get("words")
    if words is None or words.original_count == 0:
        demote(REVIEW, "no text could be read from the original, so nothing was actually "
                       "verified - this is not a pass")

    if verdict == PASS:
        reasons.append("every check that could run agreed; no critical difference found")
    return verdict, reasons


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def run_verification(
    original_rgb: np.ndarray,
    enhanced_rgb: np.ndarray,
    check_artwork: bool = True,
    lang: str = "auto",
    engine: str = "rapidocr",
    progress=None,
) -> VerificationReport:
    """Full OCR-only verification of one image pair.

    `lang` selects the RapidOCR recognition model ("auto" detects it from the original;
    "ch" is the Latin/digit default, "el" Greek, "devanagari" Hindi, and so on). It is
    resolved once and applied to both images - see the comment at the read step. Ignored
    when `engine="paddleocr-vl-1.5"`, which has no separate per-script models to choose
    between.

    `engine` picks the reader: "rapidocr" (default) or "paddleocr-vl-1.5" (local, via
    MLX - see `extract.ocr_read_paddle_vl`). Every check downstream operates on
    `ImageReading.ocr_text` / `.lines_data`, so nothing else in this function branches
    on which engine produced them - EXCEPT that `.lines_data` from the PaddleOCR-VL
    path carries no real per-line geometry, which weakens (does not break) the
    background-line filter and the prominent-text check. See
    `extract.ocr_read_paddle_vl`'s docstring for exactly what that costs.
    """
    def step(msg: str) -> None:
        if progress:
            progress(msg)

    report = VerificationReport()

    step("Cropping both images to the product…")
    o_crop, o_mask, report.original_crop_note = extract.crop_to_product(original_rgb)
    e_crop, e_mask, report.enhanced_crop_note = extract.crop_to_product(enhanced_rgb)

    step("Preparing both images identically for OCR…")
    o_prep = extract.prepare_for_ocr(o_crop)
    e_prep = extract.prepare_for_ocr(e_crop)
    # The mask was computed at the CROP's resolution; prepare_for_ocr may have resized
    # it (upscale-only). Nearest-neighbour to the prepped size keeps a per-line overlap
    # check meaningful. Resize only, not rotation-aware - fine here since deskew is a
    # no-op on this dataset's already-straight crops (see prepare_for_ocr's docstring);
    # a genuinely rotated input would need the mask carried through the same transform.
    o_mask_prep = (cv2.resize(o_mask, (o_prep.shape[1], o_prep.shape[0]),
                              interpolation=cv2.INTER_NEAREST) if o_mask is not None else None)
    e_mask_prep = (cv2.resize(e_mask, (e_prep.shape[1], e_prep.shape[0]),
                              interpolation=cv2.INTER_NEAREST) if e_mask is not None else None)

    # Resolve the recognition language ONCE, from the original, then force both images
    # onto it. Detecting per-image could put the two sides on different recognisers -
    # and two models disagreeing about the same glyph would be reported as the enhancer
    # changing the text, which is the verifier measuring itself again. The original is
    # the reference because it is the ground truth here; the enhanced image is the thing
    # under suspicion.
    if engine == "paddleocr-vl-1.5":
        # No per-script model to resolve, unlike RapidOCR - one model reads
        # everything, so both sides just call it directly with nothing to agree on
        # first.
        step("Reading the original with PaddleOCR-VL-1.5 (local, MLX)…")
        o_reading = extract.ocr_read_paddle_vl(o_prep)
        step("Reading the enhanced image (same model)…")
        e_reading = extract.ocr_read_paddle_vl(e_prep)
    else:
        step("Reading the original with RapidOCR…")
        o_reading = extract.ocr_read(o_prep, lang=lang, product_mask=o_mask_prep)
        resolved = o_reading.lang or "ch"
        step(f"Reading the enhanced image (same model: {resolved})…")
        e_reading = extract.ocr_read(e_prep, lang=resolved, product_mask=e_mask_prep)
    report.original_reading, report.enhanced_reading = o_reading, e_reading
    report.engine = o_reading.engine

    o_text, e_text = o_reading.ocr_text, e_reading.ocr_text

    step("Comparing what the two images say…")
    report.diffs = {
        "digits": signals.diff_sets("numbers", signals.digit_runs(o_text),
                                    signals.digit_runs(e_text), fold_text=True),
        "quantities": signals.diff_measures(signals.measures(o_text),
                                            signals.measures(e_text)),
        "prices": signals.diff_sets("prices", signals.prices(o_text),
                                    signals.prices(e_text)),
        "words": signals.diff_sets("words", signals.tokens(o_text),
                                   signals.tokens(e_text), fold_text=True),
    }

    step("Checking the barcode still scans…")
    report.checks.append(_barcode_check(o_reading, e_reading))

    # Digits and quantities are the sharpest signals this tool has, so they are the
    # ones allowed to reject. Prices are not: a shelf price tag legitimately vanishes
    # when the enhancer removes the background, so a lost price is expected behaviour.
    report.checks.append(_set_check("Numbers on the pack", report.diffs["digits"],
                                     critical=True))
    report.checks.append(_set_check("Quantities", report.diffs["quantities"],
                                     critical=True, fmt=_fmt_measures))
    report.checks.extend(_labeled_checks(labeled.extract(o_reading.lines_data),
                                          labeled.extract(e_reading.lines_data)))
    report.checks.append(_set_check("Prices", report.diffs["prices"], critical=False,
                                     advisory=True))

    step("Checking prominent text survived…")
    report.checks.append(_prominent_text_check(o_reading.lines_data, e_text))

    words = report.diffs["words"]
    retention, invention = words.retention, words.invention
    if retention is None:
        report.checks.append(Check("Overall text", "ABSENT",
                                    "no words were read from the original"))
    else:
        bad = retention < WORD_RETENTION_REVIEW or (invention or 0) > WORD_INVENTION_REVIEW
        report.checks.append(Check(
            "Overall text", "CHANGED" if bad else "MATCH",
            f"{retention:.0%} of the original's words survive; "
            f"{(invention or 0):.0%} of the enhanced image's words are new. {words.summary()}",
            f"{words.original_count} words", f"{words.enhanced_count} words",
            critical=False, advisory=True,
        ))

    if check_artwork:
        step("Comparing the artwork…")
        art = artwork.similarity(o_crop, e_crop)
        report.artwork = art
        if not art.measurable:
            report.checks.append(Check("Artwork", "ABSENT", art.note, advisory=True))
        else:
            report.checks.append(Check(
                "Artwork", "MATCH" if art.same_artwork else "CHANGED", art.note,
                f"{art.keypoints[0]} keypoints", f"{art.keypoints[1]} keypoints",
                critical=False, advisory=True,
            ))
        score, note = artwork.colour_correlation(o_crop, e_crop, o_mask, e_mask)
        report.checks.append(Check(
            "Pack colours", "MATCH" if score >= artwork.COLOUR_CORRELATION_THRESHOLD
            else "CHANGED", note, critical=False, advisory=True,
        ))

    step("Deciding the verdict…")
    report.verdict, report.reasons = _decide(report)
    return report
