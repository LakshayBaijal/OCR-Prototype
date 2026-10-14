"""Is the enhanced image the SAME artwork, re-rendered? Classical CV, no model.

This answers the one genuinely useful thing the VLM used to be asked - "is this still
the same product" - without a generative model, and more reliably than the VLM managed.
Two independent signals:

  * FEATURE MATCH (`similarity`) - ORB keypoints matched between the two product
    regions, filtered by Lowe's ratio test and then by a RANSAC homography fit. The
    inlier ratio answers "is the enhanced image a geometric re-rendering of the same
    printed design". A faithful studio re-shoot of the same pack has many inliers even
    though lighting, background and framing all changed, because ORB keys on corner
    structure rather than absolute brightness. A pack whose artwork was REBUILT - the
    failure case measured on img40, where every text fact survived but the enhancer
    rearranged the layout and moved the veg mark - loses them.

  * COLOUR PROFILE (`colour_distance`) - hue histogram over the product silhouette.
    Catches variant swaps that text and geometry both miss: a red-label flavour redrawn
    in orange is the same words, the same layout, and a different product on a shelf.

Both are deliberately reported as MEASUREMENTS with thresholds applied by the caller,
not as booleans. The thresholds here are the least evidence-backed numbers in the
package - they were set from a handful of pairs, not a labelled study - so they are
named constants with their provenance stated rather than magic numbers buried in an
`if`. Treat a borderline score as "look at it", never as a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Inlier ratio above this reads as "same artwork, re-rendered". Calibrated loosely on
# the audit pairs: faithful re-renders scored well above it, and the one measured
# layout-rebuild case scored below. Not a study - revisit with real labelled data
# before treating a near-threshold score as meaningful.
SAME_ARTWORK_THRESHOLD = 0.15

# Hue-histogram correlation below this reads as a palette change. Generous on purpose:
# the enhancer legitimately changes lighting and white balance, which shifts saturation
# and value a lot but hue comparatively little - which is exactly why this compares
# HUE only, and why the bar sits low enough to ignore ordinary relighting.
COLOUR_CORRELATION_THRESHOLD = 0.55

_MAX_DIM = 900  # both images are downscaled to this before matching, for speed


@dataclass
class ArtworkSimilarity:
    inlier_ratio: float
    inliers: int
    good_matches: int
    keypoints: tuple[int, int]
    note: str

    @property
    def same_artwork(self) -> bool:
        return self.inlier_ratio >= SAME_ARTWORK_THRESHOLD

    @property
    def measurable(self) -> bool:
        """False when there simply was not enough texture to compare - a plain pack, a
        heavy blur. Distinguishing 'no evidence' from 'evidence of difference' is the
        whole reason this flag exists."""
        return self.good_matches >= 10


def _prep(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = min(1.0, _MAX_DIM / max(h, w))
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # The enhancer changes exposure substantially; CLAHE puts both sides on comparable
    # local contrast so ORB is comparing structure rather than brightness.
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def similarity(original_rgb: np.ndarray, enhanced_rgb: np.ndarray) -> ArtworkSimilarity:
    """ORB + Lowe ratio + RANSAC homography. Never raises."""
    try:
        a, b = _prep(original_rgb), _prep(enhanced_rgb)
        orb = cv2.ORB_create(nfeatures=2000)
        kp_a, des_a = orb.detectAndCompute(a, None)
        kp_b, des_b = orb.detectAndCompute(b, None)

        if des_a is None or des_b is None or len(kp_a) < 10 or len(kp_b) < 10:
            return ArtworkSimilarity(0.0, 0, 0, (len(kp_a or []), len(kp_b or [])),
                                     "too few keypoints to compare - a plain or blurred pack")

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw = matcher.knnMatch(des_a, des_b, k=2)
        # Lowe's ratio test: keep a match only when the best candidate is clearly better
        # than the second best, which is what separates a real correspondence from a
        # repeated texture matching everywhere.
        good = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]

        if len(good) < 10:
            return ArtworkSimilarity(0.0, 0, len(good), (len(kp_a), len(kp_b)),
                                     f"only {len(good)} confident feature matches - "
                                     f"not enough to judge either way")

        src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0
        ratio = inliers / len(good) if good else 0.0

        if ratio >= SAME_ARTWORK_THRESHOLD:
            note = (f"{inliers}/{len(good)} feature matches fit one consistent geometry "
                    f"({ratio:.0%}) - the enhanced image is a re-rendering of the same artwork")
        else:
            note = (f"only {inliers}/{len(good)} feature matches fit a consistent geometry "
                    f"({ratio:.0%}) - the printed design may have been rebuilt rather than "
                    f"re-rendered")
        return ArtworkSimilarity(ratio, inliers, len(good), (len(kp_a), len(kp_b)), note)
    except Exception as exc:
        return ArtworkSimilarity(0.0, 0, 0, (0, 0),
                                 f"feature matching failed ({type(exc).__name__})")


def colour_correlation(
    original_rgb: np.ndarray, enhanced_rgb: np.ndarray,
    original_mask: np.ndarray | None = None, enhanced_mask: np.ndarray | None = None,
) -> tuple[float, str]:
    """Hue-histogram correlation over the product silhouette. Returns (score, note).

    Hue only, and masked to the product: the background is white in one image and a
    supermarket aisle in the other, so including it would measure the enhancer doing
    its job. Saturation and value are excluded because relighting moves them
    legitimately.
    """
    try:
        def hue_hist(rgb, mask):
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            m = None
            if mask is not None and mask.shape[:2] == rgb.shape[:2]:
                m = (mask > 0).astype(np.uint8)
            # Ignore near-grey pixels: their hue is numerically unstable and would swamp
            # the histogram of a mostly-white pack.
            sat_ok = (hsv[:, :, 1] > 60).astype(np.uint8)
            m = sat_ok if m is None else cv2.bitwise_and(m, sat_ok)
            hist = cv2.calcHist([hsv], [0], m, [36], [0, 180])
            total = hist.sum()
            return hist / total if total > 0 else hist

        ha = hue_hist(original_rgb, original_mask)
        hb = hue_hist(enhanced_rgb, enhanced_mask)
        if ha.sum() == 0 or hb.sum() == 0:
            return 1.0, "not enough saturated colour on either pack to compare"

        score = float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
        if score >= COLOUR_CORRELATION_THRESHOLD:
            return score, f"pack colours match ({score:.0%} hue correlation)"
        return score, (f"pack colours differ ({score:.0%} hue correlation) - check for a "
                       f"changed flavour/variant colourway")
    except Exception as exc:
        return 1.0, f"colour comparison failed ({type(exc).__name__})"
