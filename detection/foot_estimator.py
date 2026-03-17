"""
detection/foot_estimator.py — Automatic Foot-Point Estimation Under Occlusion

When a person's lower body is hidden by another person's bounding box (crowd
occlusion) the raw bottom-center of the YOLO bbox is NOT the feet — it's
somewhere on the mid-body of the occluder.  Projecting that point through the
homography gives a significantly wrong floor position.

This module provides ``FootEstimator`` which:

  1. **Detects occlusion** — checks whether each person bbox's bottom 30 %
     zone is overlapped horizontally by another person's bbox whose top edge
     is above our target's bottom edge.

  2. **Builds a reference height pool** — uses the pixel heights of all
     *unoccluded* persons in the same frame (weighted by inverse pixel distance
     from the target) to estimate the typical person height at that camera
     depth/perspective.

  3. **Extrapolates the foot** — if occluded, projects the foot downward by:
       corrected_foot_y = y1_target + ref_height
     where ref_height is the weighted-median reference height.

  4. **Computes occlusion_confidence** — a 0.0–1.0 score that tells downstream
     fusion how much to trust this foot projection:
       occlusion_confidence = visible_height / ref_height
     1.0 means feet are unambiguously visible; < 1.0 means the foot was
     estimated (the lower the value, the more was hidden).

Usage
-----
::

    from detection.foot_estimator import FootEstimator, FootEstimate
    from detection.detector import Detection

    estimator = FootEstimator()
    estimates = estimator.estimate(detections)   # list[Detection] → list[FootEstimate]

    for det, est in zip(detections, estimates):
        # Use est.foot_x / est.foot_y for floor projection
        print(det, est.occlusion_confidence, est.was_occluded)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────

# Fraction of a bbox's height that counts as the "bottom zone"
_BOTTOM_ZONE_FRAC     = 0.3

# Minimum horizontal overlap (as fraction of target width) to be considered
# a valid occluder.  Eliminates detections that are side-by-side but not
# actually covering the feet.
_MIN_HORIZ_OVERLAP_FRAC = 0.20

# Minimum fraction of the bottom zone that the occluder must cover vertically
# to be treated as a genuine occlusion (prevents false triggers from tiny clips)
_MIN_VERT_COVER_FRAC  = 0.10

# Fall-back person height used only when < 1 reference detection exists
# (e.g. a single-person frame where that one person is also occluded).
# Set to 0 to disable the hard fallback; the module will then use
# the visible bbox height as the estimate.
_DEFAULT_PERSON_H_PX  = 0          # 0 = auto (use bbox height as fallback)

# When the estimated foot_y is below the *real* y2 of a bbox by more than
# this fraction of the reference height, we cap it (safety net against wild
# extrapolation in extreme crowd scenarios).
_MAX_EXTRAPOLATION_FRAC = 1.5      # at most 150 % of ref height below y1


# ═══════════════════════════════════════════════════════════════════════════
#  Data class
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FootEstimate:
    """
    Corrected foot point for one person detection.

    Attributes
    ----------
    foot_x              : float   — corrected pixel X  (horizontal bbox centre)
    foot_y              : float   — corrected pixel Y  (extrapolated if occluded)
    occlusion_confidence: float   — 1.0 = feet clearly visible and used as-is;
                                    < 1.0 = feet were estimated algorithmically
    was_occluded        : bool    — True if the foot was extrapolated
    ref_height_px       : float   — reference person height used (pixels)
    occluder_bbox       : tuple | None  — (x1,y1,x2,y2) of primary occluder bbox
    """

    foot_x:               float
    foot_y:               float
    occlusion_confidence: float
    was_occluded:         bool
    ref_height_px:        float        = 0.0
    occluder_bbox:        Optional[tuple] = None


# ═══════════════════════════════════════════════════════════════════════════
#  FootEstimator
# ═══════════════════════════════════════════════════════════════════════════

class FootEstimator:
    """
    Per-frame foot-point corrector for crowd occlusion scenarios.

    Instantiate once per camera and call ``estimate(detections)`` every frame.
    No internal state is maintained between frames.
    """

    # ──────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────

    def estimate(self, detections: list) -> List[FootEstimate]:
        """
        Compute corrected foot estimates for all persons in one frame.

        Parameters
        ----------
        detections : list[Detection]
            All person detections from a single camera frame.
            Each must have a ``bbox`` attribute = (x1, y1, x2, y2).

        Returns
        -------
        list[FootEstimate]
            Same length as *detections*, one estimate per person.
            For unoccluded persons ``was_occluded=False`` and
            ``occlusion_confidence=1.0``.
        """
        if not detections:
            return []

        bboxes = np.array(
            [det.bbox for det in detections], dtype=np.float32
        )  # shape (N, 4) — [x1, y1, x2, y2]

        n = len(detections)

        # ── Step 1: find occluder for each detection ───────────────────────
        # occluder_idx[i] = index of the primary occluder of detection i,
        # or -1 if no occlusion.
        occluder_idx   = np.full(n, -1, dtype=np.int32)
        occluder_cover = np.zeros(n, dtype=np.float32)  # vert cover fraction

        for i in range(n):
            x1_i, y1_i, x2_i, y2_i = bboxes[i]
            h_i    = y2_i - y1_i
            w_i    = x2_i - x1_i

            if h_i <= 0 or w_i <= 0:
                continue

            # Bottom zone of person i
            bz_top    = y2_i - _BOTTOM_ZONE_FRAC * h_i
            bz_bottom = y2_i

            best_cover = 0.0
            best_j     = -1

            for j in range(n):
                if i == j:
                    continue

                x1_j, y1_j, x2_j, y2_j = bboxes[j]

                # j must have its top edge above i's bottom (j is in front of i)
                if y1_j >= y2_i:
                    continue

                # Horizontal overlap between i and j
                h_olap = max(0.0, min(x2_i, x2_j) - max(x1_i, x1_j))
                if h_olap / w_i < _MIN_HORIZ_OVERLAP_FRAC:
                    continue

                # Vertical coverage: Does j cross into i's bottom zone?
                # We need j's top to be above i's bottom (already checked y1_j < y2_i)
                # and j's bottom must reach into or past i's bottom zone
                if y2_j < bz_top:
                    continue  # j is entirely floating above i's bottom zone
                
                # We use the raw overlap area as the score to pick the "best" occluder
                cover_frac = y2_j - max(bz_top, y1_j)

                if cover_frac > best_cover:
                    best_cover = cover_frac
                    best_j     = j

            if best_j >= 0:
                occluder_idx[i]   = best_j
                occluder_cover[i] = best_cover

        # ── Step 2: build reference height pool (unoccluded persons) ──────
        # For each occluded person, compute a weighted median of the pixel
        # heights of all nearby unoccluded persons.

        # Heights of each detection (pixels)
        heights = bboxes[:, 3] - bboxes[:, 1]            # y2 - y1
        # Centres of each detection (pixel coords)
        cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0

        is_clear = (occluder_idx == -1)                   # unoccluded mask

        # ── Step 3: compute FootEstimate for every detection ──────────────
        estimates: List[FootEstimate] = []

        for i in range(n):
            x1_i, y1_i, x2_i, y2_i = bboxes[i]
            foot_x_raw = (x1_i + x2_i) / 2.0   # horizontal centre always stable

            if occluder_idx[i] == -1:
                # Unoccluded — use real bottom-centre
                estimates.append(FootEstimate(
                    foot_x               = float(foot_x_raw),
                    foot_y               = float(y2_i),
                    occlusion_confidence = 1.0,
                    was_occluded         = False,
                    ref_height_px        = float(heights[i]),
                    occluder_bbox        = None,
                ))
                continue

            # ── Occluded person: extrapolate foot ────────────────────────
            # Reference height: weighted median of unoccluded persons,
            # weighted by 1 / (Euclidean distance + 1).
            ref_h = self._reference_height(
                i, cx, cy, heights, is_clear
            )

            # How much of person i's height is visible above the occluder?
            j = occluder_idx[i]
            x1_j, y1_j, x2_j, y2_j = bboxes[j]
            visible_height = max(0.0, y1_j - y1_i)

            if ref_h <= 0 or visible_height >= ref_h * 0.95:
                # Occluder barely clips the bottom — no meaningful correction
                estimates.append(FootEstimate(
                    foot_x               = float(foot_x_raw),
                    foot_y               = float(y2_i),
                    occlusion_confidence = 1.0,
                    was_occluded         = False,
                    ref_height_px        = float(ref_h),
                    occluder_bbox        = None,
                ))
                continue

            # Corrected foot y = top of person + full estimated height
            corrected_foot_y = y1_i + ref_h

            # Safety cap: don't extrapolate beyond a reasonable limit
            max_foot_y = y1_i + ref_h * _MAX_EXTRAPOLATION_FRAC
            corrected_foot_y = min(corrected_foot_y, max_foot_y)

            # Occlusion confidence: fraction of expected height that is visible
            occ_conf = float(np.clip(visible_height / ref_h, 0.05, 1.0))

            occluder_box = (
                float(bboxes[j, 0]), float(bboxes[j, 1]),
                float(bboxes[j, 2]), float(bboxes[j, 3]),
            )

            logger.debug(
                "FootEstimator: bbox[%d] occluded by bbox[%d]  "
                "visible=%.1fpx  ref_h=%.1fpx  foot_y %.1f→%.1f  conf=%.2f",
                i, j, visible_height, ref_h, y2_i, corrected_foot_y, occ_conf,
            )

            estimates.append(FootEstimate(
                foot_x               = float(foot_x_raw),
                foot_y               = float(corrected_foot_y),
                occlusion_confidence = occ_conf,
                was_occluded         = True,
                ref_height_px        = float(ref_h),
                occluder_bbox        = occluder_box,
            ))

        return estimates

    # ──────────────────────────────────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _reference_height(
        target_idx: int,
        cx: np.ndarray,
        cy: np.ndarray,
        heights: np.ndarray,
        is_clear: np.ndarray,
    ) -> float:
        """
        Weighted median height of unoccluded persons near *target_idx*.

        Weights are inversely proportional to pixel distance from the target's
        bbox centre, so persons at similar depth/perspective count more.

        Falls back to:
          1. Unweighted median of all clear heights (if distance is uniform)
          2. Median of ALL heights (if no clear persons at all)
          3. _DEFAULT_PERSON_H_PX (if even that is unavailable)
        """
        clear_indices = np.where(is_clear)[0]

        # Remove the target itself (it's being estimated)
        clear_indices = clear_indices[clear_indices != target_idx]

        if len(clear_indices) == 0:
            # Fall back: use median of everyone's height (incl. occluded)
            all_h = heights[heights > 0]
            if len(all_h) > 0:
                return float(np.median(all_h))
            return float(_DEFAULT_PERSON_H_PX) if _DEFAULT_PERSON_H_PX > 0 else float(heights[target_idx])

        # Euclidean pixel distances from target centre
        dx = cx[clear_indices] - cx[target_idx]
        dy = cy[clear_indices] - cy[target_idx]
        dists = np.sqrt(dx * dx + dy * dy)

        # Weights = 1 / (distance + 1) — +1 avoids division by zero and
        # prevents a co-located detection from getting infinite weight
        weights = 1.0 / (dists + 1.0)
        ref_heights = heights[clear_indices]

        # Weighted median via sorted cumulative weights
        sort_idx     = np.argsort(ref_heights)
        sorted_h     = ref_heights[sort_idx]
        sorted_w     = weights[sort_idx]
        cumulative_w = np.cumsum(sorted_w)
        half_w       = cumulative_w[-1] / 2.0
        median_idx   = np.searchsorted(cumulative_w, half_w)
        median_idx   = min(median_idx, len(sorted_h) - 1)

        return float(sorted_h[median_idx])


# ═══════════════════════════════════════════════════════════════════════════
#  Module-level convenience
# ═══════════════════════════════════════════════════════════════════════════

_default_estimator: Optional[FootEstimator] = None


def get_default_estimator() -> FootEstimator:
    """Return the module-level shared FootEstimator instance."""
    global _default_estimator
    if _default_estimator is None:
        _default_estimator = FootEstimator()
    return _default_estimator
