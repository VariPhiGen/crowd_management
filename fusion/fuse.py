"""
fuse.py — Multi-Camera Detection Fusion

Deduplicates persons seen by overlapping cameras using the Hungarian
algorithm (scipy.optimize.linear_sum_assignment) to find the globally
optimal 1-to-1 pairing between detections from different cameras inside
each overlap zone.

Algorithm (four-step)
---------------------
Step 1 — PASS-THROUGH (no overlap)
    Cameras not present in any overlap zone have all their detections
    forwarded unchanged as single-camera FusedDetections.

Step 2 — PER-ZONE HUNGARIAN MATCHING
    For each overlap zone (e.g. cam_1 ∩ cam_2):

    a. Split each camera's detections:
         inside_zone  = zone.contains_point(x, y, use_buffer=True)
         outside_zone = everything else

    b. outside_zone detections → forwarded as singles.

    c. inside_zone detections from both cameras → Hungarian matching:
         cost_matrix[i][j] = Euclidean distance between cam_a[i] and cam_b[j]
         row_ind, col_ind  = linear_sum_assignment(cost_matrix)

         If distance < zone.distance_threshold:
             MERGE → highest-confidence camera's position
             FusedDetection(is_fused=True, source_cameras=[cam_a, cam_b])
         Else:
             TOO FAR → both kept as separate singles

    d. Unmatched detections (|cam_a| ≠ |cam_b|) → forwarded as singles.

Step 3 — COLLECT + FLOOR NMS
    All FusedDetections are collected then passed through a greedy
    floor-space NMS that suppresses residual duplicates closer than
    FLOOR_NMS_MIN_SEP_M metres (default 0.5 m).  Fused detections take
    priority over singles; ties broken by confidence.

Step 4 — EMA POSITION SMOOTHING
    DetectionFuser maintains a lightweight track list across frames.
    Each track is matched to the nearest new detection within
    EMA_MATCH_RADIUS_M (default 1.5 m) using greedy nearest-neighbour.
    Matched positions are updated with an exponential moving average
    (alpha=EMA_ALPHA, default 0.55).  Tracks not matched for up to
    EMA_OCCLUSION_FRAMES (default 2) frames are still emitted (occlusion
    tolerance); tracks missing for > EMA_MAX_MISSED (default 8) frames
    are dropped.

Claim tracking
--------------
A per-detection "claimed" set prevents a detection from being used in
multiple zone matchings when a camera appears in more than one overlap zone
(e.g. cam_1 in zone_AB and zone_AC).

Accepted detection types
------------------------
``FloorDetection``  (detection/detector.py)   — primary interface
Legacy Detection dataclass with floor_point   — backward compat
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from fusion.overlap import OverlapZone, load_overlap_zones

logger = logging.getLogger(__name__)

# ── Tuning constants ─────────────────────────────────────────────────────────
# Floor-space NMS
FLOOR_NMS_MIN_SEP_M   = 0.5   # metres — suppress duplicates closer than this

# EMA position smoother
EMA_ALPHA             = 0.45  # weight of new observation (lower = smoother/more lag-tolerant)
EMA_MATCH_RADIUS_M    = 3.0   # metres — max gap allowed to still re-associate a detection
                               #   (raised so a person briefly missed doesn't spawn a new ID)
EMA_OCCLUSION_FRAMES  = 8     # frames to keep emitting a track with no new detection
                               #   (~0.5 s @ 15 FPS — hides brief full-body occlusions)
EMA_MAX_MISSED        = 25    # frames before a track is permanently dropped
                               #   (~1.7 s @ 15 FPS — persists through longer occlusions)

# Track confirmation gate — a new track must be observed for this many
# consecutive frames before it is promoted to a confirmed (named) track.
# Un-confirmed tracks use track_id = -1 and are invisible in the TSV log.
# This prevents a single-frame false positive from consuming a track ID.
EMA_MIN_CONFIRM_FRAMES = 3


# ═══════════════════════════════════════════════════════════════════════════
#  FusedDetection
# ═══════════════════════════════════════════════════════════════════════════

class FusedDetection:
    """
    Best-estimate position of one physical person on the factory floor.

    Produced by ``DetectionFuser.fuse()``.  May represent either a single
    camera observation or a merged observation from two cameras.

    Attributes
    ----------
    floor_x          : float  — metres (X = right from origin)
    floor_y          : float  — metres (Y = up from origin)
    confidence       : float  — max(cam_confidences) after merge
    source_cameras   : list[str]
        ``["cam_3"]`` for a single-camera detection.
        ``["cam_1", "cam_2"]`` for a cross-camera merge.
    is_fused         : bool   — True only when two cameras contributed
    fusion_distance  : float  — metres between the two original foot-points
                                (0.0 for single-camera detections)
    """

    __slots__ = [
        'floor_x', 'floor_y', 'confidence',
        'source_cameras', 'is_fused', 'fusion_distance', 'track_id',
    ]

    def __init__(
        self,
        floor_x: float,
        floor_y: float,
        confidence: float,
        source_cameras: list[str],
        is_fused: bool,
        fusion_distance: float,
        track_id: int = -1,
    ) -> None:
        self.floor_x         = floor_x
        self.floor_y         = floor_y
        self.confidence      = confidence
        self.source_cameras  = source_cameras
        self.is_fused        = is_fused
        self.fusion_distance = fusion_distance
        self.track_id        = track_id

    # Convenience property so legacy code using .floor_point still works
    @property
    def floor_point(self) -> tuple[float, float]:
        return (self.floor_x, self.floor_y)

    def __repr__(self) -> str:
        tag  = "FUSED" if self.is_fused else "single"
        cams = "+".join(self.source_cameras)
        dist = f"  Δ={self.fusion_distance:.2f}m" if self.is_fused else ""
        tid  = f" id={self.track_id}" if self.track_id >= 0 else ""
        return (
            f"FusedDetection({tag}, cam=[{cams}]{dist}{tid}, "
            f"floor=({self.floor_x:.2f},{self.floor_y:.2f})m, "
            f"conf={self.confidence:.2f})"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  DetectionFuser
# ═══════════════════════════════════════════════════════════════════════════

class DetectionFuser:
    """
    Merges per-camera floor detections into a de-duplicated person list.

    Parameters
    ----------
    overlap_zones : list[OverlapZone]
        Loaded from ``config/overlap_zones.json`` via
        ``fusion.overlap.load_overlap_zones()``.
    """

    def __init__(self, overlap_zones: list[OverlapZone]) -> None:
        self.overlap_zones = overlap_zones

        # Pre-index: which cameras appear in at least one zone?
        self._cameras_in_zones: set[str] = {
            cam_id
            for zone in overlap_zones
            for cam_id in zone.camera_ids
        }

        # EMA track state — each entry is a dict:
        #   id          : confirmed integer track ID, or None while unconfirmed
        #   age         : consecutive frames this track has been matched
        #   x, y        : smoothed floor position (m)
        #   conf        : latest confidence
        #   cameras     : latest source_cameras list
        #   is_fused    : latest is_fused flag
        #   dist        : latest fusion_distance
        #   missed      : consecutive frames without a match
        self._tracks: list[dict] = []
        self._next_track_id: int = 1   # never reused; increments on each confirmed track

    # ──────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────

    def fuse(
        self,
        detections_by_camera: dict[str, list],
    ) -> list[FusedDetection]:
        """
        Fuse detections from all cameras into a unique-person list.

        Parameters
        ----------
        detections_by_camera : dict[str, list]
            Keys are camera IDs; values are lists of FloorDetection
            (or legacy Detection dataclass objects with floor_point set).

        Returns
        -------
        list[FusedDetection]
            Sorted by floor_y descending (bottom of scene first).
        """
        # ── Flatten all detections into a workable list ───────────────────
        # Each entry: (cam_id, det, floor_x, floor_y, confidence, occ_confidence)
        all_items: list[tuple[str, object, float, float, float, float]] = []

        for cam_id, dets in detections_by_camera.items():
            for det in dets:
                fx, fy = _get_floor_xy(det)
                if fx is None:
                    continue
                conf    = float(det.confidence)
                occ     = float(getattr(det, 'occlusion_confidence', 1.0))
                all_items.append((cam_id, det, fx, fy, conf, occ))

        if not all_items:
            return []

        # claimed[i] = True once item i has been used in a merge or matched
        claimed: set[int] = set()
        result:  list[FusedDetection] = []

        # ── Step 1: pass-through cameras not in any overlap zone ──────────
        for i, (cam_id, det, fx, fy, conf, occ) in enumerate(all_items):
            if cam_id not in self._cameras_in_zones:
                result.append(FusedDetection(
                    floor_x        = fx,
                    floor_y        = fy,
                    confidence     = conf,
                    source_cameras = [cam_id],
                    is_fused       = False,
                    fusion_distance = 0.0,
                ))
                claimed.add(i)        # mark so they aren't doubled in step 3

        # ── Step 2: per-zone Hungarian matching ───────────────────────────
        for zone in self.overlap_zones:
            if len(zone.camera_ids) < 2:
                logger.warning("Zone %s has < 2 cameras — skipped.", zone.id)
                continue

            cam_a, cam_b = zone.camera_ids[0], zone.camera_ids[1]

            # Collect unclaimed detections from each camera inside the zone
            idx_a: list[int] = []
            idx_b: list[int] = []

            for i, (cam_id, det, fx, fy, conf, occ) in enumerate(all_items):
                if i in claimed:
                    continue
                if not zone.contains_point(fx, fy, use_buffer=True):
                    continue
                if cam_id == cam_a:
                    idx_a.append(i)
                elif cam_id == cam_b:
                    idx_b.append(i)

            # Detections from these cameras OUTSIDE the zone → pass through
            # (will be caught by the unclaimed sweep in step 3)

            if not idx_a or not idx_b:
                # One or both cameras have no detections inside this zone;
                # all their items remain unclaimed and flow to step 3.
                continue

            # ── Build Euclidean cost matrix ───────────────────────────────
            pts_a = np.array(
                [(all_items[i][2], all_items[i][3]) for i in idx_a],
                dtype=np.float64,
            )
            pts_b = np.array(
                [(all_items[i][2], all_items[i][3]) for i in idx_b],
                dtype=np.float64,
            )

            n_a, n_b = len(idx_a), len(idx_b)
            cost = np.zeros((n_a, n_b), dtype=np.float64)
            for r in range(n_a):
                for c in range(n_b):
                    dx = pts_a[r, 0] - pts_b[c, 0]
                    dy = pts_a[r, 1] - pts_b[c, 1]
                    cost[r, c] = (dx * dx + dy * dy) ** 0.5

            # ── Hungarian algorithm ───────────────────────────────────────
            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind, col_ind):
                dist     = float(cost[r, c])
                global_a = idx_a[r]
                global_b = idx_b[c]

                _, _, fx_a, fy_a, conf_a, occ_a = all_items[global_a]
                _, _, fx_b, fy_b, conf_b, occ_b = all_items[global_b]

                if dist < zone.distance_threshold:
                    # ── MERGE: occlusion-confidence-weighted position ──────
                    # Effective trust in each camera's floor projection:
                    #   eff = raw_detection_confidence × occlusion_confidence
                    # occlusion_confidence = 1.0  → feet clearly visible
                    # occlusion_confidence < 1.0  → foot was algorithmically
                    #   estimated (FootEstimator predicted it from crowd context)
                    #
                    # This ensures the camera that can actually see the feet
                    # always wins position authority, regardless of raw YOLO
                    # confidence which is unrelated to foot visibility.
                    eff_a = conf_a * occ_a
                    eff_b = conf_b * occ_b

                    if abs(eff_a - eff_b) < 0.02:
                        # Effectively equal — weighted average by effective trust
                        w_sum = eff_a + eff_b
                        wx = (eff_a * fx_a + eff_b * fx_b) / w_sum
                        wy = (eff_a * fy_a + eff_b * fy_b) / w_sum
                    elif eff_a >= eff_b:
                        wx, wy = fx_a, fy_a   # cam_a has better projection
                    else:
                        wx, wy = fx_b, fy_b   # cam_b has better projection

                    result.append(FusedDetection(
                        floor_x         = wx,
                        floor_y         = wy,
                        confidence      = max(conf_a, conf_b),
                        source_cameras  = [cam_a, cam_b],
                        is_fused        = True,
                        fusion_distance = dist,
                    ))
                    claimed.add(global_a)
                    claimed.add(global_b)
                    logger.debug(
                        "Zone %s: merged %s(eff=%.2f)+%s(eff=%.2f)  "
                        "Δ=%.2f m  pos=(%.2f,%.2f)",
                        zone.id, cam_a, eff_a, cam_b, eff_b, dist, wx, wy,
                    )
                else:
                    # ── TOO FAR: keep as separate singles ─────────────────
                    logger.debug(
                        "Zone %s: dist %.2f m > threshold %.2f m — kept separate.",
                        zone.id, dist, zone.distance_threshold,
                    )
                    # Leave unclaimed so they flow to step 3

        # ── Step 3: remaining unclaimed detections → singles ──────────────
        for i, (cam_id, det, fx, fy, conf, occ) in enumerate(all_items):
            if i not in claimed:
                result.append(FusedDetection(
                    floor_x         = fx,
                    floor_y         = fy,
                    confidence      = conf,
                    source_cameras  = [cam_id],
                    is_fused        = False,
                    fusion_distance = 0.0,
                ))

        # ── Step 4a: floor-space NMS — remove residual duplicates ────────────
        result = _floor_nms(result, min_sep_m=FLOOR_NMS_MIN_SEP_M)

        # ── Step 4b: EMA position smoothing across frames ─────────────────
        result = self._ema_smooth(result)

        # Sort by floor_y descending (persons furthest into scene last, useful
        # for rendering with painter's algorithm)
        result.sort(key=lambda fd: fd.floor_y, reverse=True)
        return result

    # ──────────────────────────────────────────────────────────────────────
    #  Statistics
    # ──────────────────────────────────────────────────────────────────────

    def get_stats(self, fused: list[FusedDetection]) -> dict:
        """
        Return a summary dict for the given fused detection list.

        Returns
        -------
        dict with keys:
            ``total_persons``  — total unique persons
            ``fused_count``    — detections merged from ≥2 cameras
            ``single_count``   — detections from exactly 1 camera
            ``by_camera``      — dict[camera_id, contribution_count]
        """
        fused_count  = sum(1 for f in fused if f.is_fused)
        single_count = len(fused) - fused_count

        by_camera: dict[str, int] = {}
        for fd in fused:
            for cam_id in fd.source_cameras:
                by_camera[cam_id] = by_camera.get(cam_id, 0) + 1

        return {
            "total_persons": len(fused),
            "fused_count":   fused_count,
            "single_count":  single_count,
            "by_camera":     by_camera,
        }

    # ──────────────────────────────────────────────────────────────────────
    #  EMA position smoother (internal, called by fuse())
    # ──────────────────────────────────────────────────────────────────────

    def _ema_smooth(
        self,
        raw: list[FusedDetection],
    ) -> list[FusedDetection]:
        """
        Smooth detection positions across frames using an exponential moving
        average (EMA) and a simple greedy nearest-neighbour track matcher.

        Tracks are seeded from new unmatched detections and aged out after
        EMA_MAX_MISSED consecutive missed frames.  Tracks with ≤
        EMA_OCCLUSION_FRAMES missed frames are still emitted so that brief
        occlusions don't cause flickering disappearances.
        """
        # ── Match raw detections to existing tracks ───────────────────────
        matched_raw: set[int] = set()
        matched_trk: set[int] = set()

        if self._tracks and raw:
            # Build (N_raw × N_trk) distance matrix
            n_r = len(raw)
            n_t = len(self._tracks)
            dist_mat = np.full((n_r, n_t), np.inf, dtype=np.float64)
            for ri, fd in enumerate(raw):
                for ti, t in enumerate(self._tracks):
                    dx = fd.floor_x - t['x']
                    dy = fd.floor_y - t['y']
                    d  = (dx * dx + dy * dy) ** 0.5
                    if d < EMA_MATCH_RADIUS_M:
                        dist_mat[ri, ti] = d

            # Greedy nearest-neighbour matching (fast for small N)
            while True:
                if np.isinf(dist_mat).all():
                    break
                ri, ti = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)
                if dist_mat[ri, ti] >= EMA_MATCH_RADIUS_M:
                    break
                # EMA update
                fd = raw[ri]
                t  = self._tracks[ti]
                a  = EMA_ALPHA
                t['x']       = a * fd.floor_x + (1.0 - a) * t['x']
                t['y']       = a * fd.floor_y + (1.0 - a) * t['y']
                t['conf']    = fd.confidence
                t['cameras'] = fd.source_cameras
                t['is_fused']= fd.is_fused
                t['dist']    = fd.fusion_distance
                t['missed']  = 0
                t['age']     = t.get('age', 0) + 1
                # Promote to confirmed track once seen for enough consecutive frames
                if t['id'] is None and t['age'] >= EMA_MIN_CONFIRM_FRAMES:
                    t['id'] = self._next_track_id
                    self._next_track_id += 1
                matched_raw.add(ri)
                matched_trk.add(ti)
                # Remove row and column so they aren't reused
                dist_mat[ri, :] = np.inf
                dist_mat[:, ti] = np.inf

        # ── Unmatched raw detections → new (unconfirmed) tracks ──────────
        for ri, fd in enumerate(raw):
            if ri not in matched_raw:
                self._tracks.append({
                    'id':      None,   # confirmed only after EMA_MIN_CONFIRM_FRAMES
                    'age':     1,      # first observation
                    'x':       fd.floor_x,
                    'y':       fd.floor_y,
                    'conf':    fd.confidence,
                    'cameras': fd.source_cameras,
                    'is_fused':fd.is_fused,
                    'dist':    fd.fusion_distance,
                    'missed':  0,
                })

        # ── Age unmatched existing tracks ─────────────────────────────────
        for ti, t in enumerate(self._tracks):
            if ti not in matched_trk and t['missed'] == 0:
                # Only age tracks that weren't just created (missed still 0
                # for brand-new tracks added above in the same frame)
                pass   # handled by initialisation above; will age next frame

        # Age tracks that existed before this frame and weren't matched
        n_old = len(self._tracks) - (len(raw) - len(matched_raw))
        for ti in range(n_old):
            if ti not in matched_trk:
                self._tracks[ti]['missed'] += 1

        # ── Drop stale tracks ─────────────────────────────────────────────
        self._tracks = [t for t in self._tracks
                        if t['missed'] <= EMA_MAX_MISSED]

        # ── Build output from active / recently-occluded tracks ───────────
        # Only emit confirmed tracks (id is not None) during occlusion windows;
        # unconfirmed tracks are emitted without an ID so the floor renderer can
        # still show a "?"-labelled dot, but the crossing log ignores them.
        out: list[FusedDetection] = []
        for t in self._tracks:
            if t['missed'] > EMA_OCCLUSION_FRAMES:
                continue
            # During confirmed occlusion, only persist tracks that have an ID
            if t['missed'] > 0 and t['id'] is None:
                continue
            out.append(FusedDetection(
                floor_x         = t['x'],
                floor_y         = t['y'],
                confidence      = t['conf'],
                source_cameras  = t['cameras'],
                is_fused        = t['is_fused'],
                fusion_distance = t['dist'],
                track_id        = t['id'] if t['id'] is not None else -1,
            ))
        return out

    def reset_tracks(self) -> None:
        """Clear all EMA tracks (call when video resets or camera changes).

        Note: ``_next_track_id`` is intentionally NOT reset so that new tracks
        created after a reset receive IDs that are globally unique for the
        session (important for crossing-event TSV logs).
        """
        self._tracks = []


# ═══════════════════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════════════════

def build_fuser(
    overlap_config_path: Optional[str | Path] = None,
) -> DetectionFuser:
    """
    Convenience factory: load overlap zones from JSON → return DetectionFuser.

    Parameters
    ----------
    overlap_config_path : str | Path | None
        Path to ``overlap_zones.json``.  Defaults to
        ``config/overlap_zones.json`` in the project root.

    Returns
    -------
    DetectionFuser
    """
    if overlap_config_path is not None:
        zones = load_overlap_zones(overlap_config_path)
    else:
        zones = load_overlap_zones()
    return DetectionFuser(zones)


# ═══════════════════════════════════════════════════════════════════════════
#  Private helpers
# ═══════════════════════════════════════════════════════════════════════════

def _floor_nms(
    detections: list[FusedDetection],
    min_sep_m: float = FLOOR_NMS_MIN_SEP_M,
) -> list[FusedDetection]:
    """
    Greedy floor-space non-maximum suppression.

    Removes detections that are within *min_sep_m* metres of a better
    detection.  Priority order: fused > single, then higher confidence.
    This catches residual double-counts that survive zone matching (e.g.
    same person seen by both cameras but their projected positions are
    just beyond distance_threshold_m).

    Parameters
    ----------
    detections : list[FusedDetection]
    min_sep_m  : float  — suppression radius in metres (default 0.5 m)

    Returns
    -------
    list[FusedDetection]  — deduplicated, original order preserved for kept items
    """
    if len(detections) <= 1:
        return detections

    # Sort: fused detections first, then by confidence descending
    ordered = sorted(
        detections,
        key=lambda d: (int(d.is_fused), d.confidence),
        reverse=True,
    )

    kept: list[FusedDetection] = []
    for cand in ordered:
        suppressed = False
        for k in kept:
            dx = cand.floor_x - k.floor_x
            dy = cand.floor_y - k.floor_y
            if (dx * dx + dy * dy) ** 0.5 < min_sep_m:
                suppressed = True
                logger.debug(
                    "Floor NMS: suppressed %.2f,%.2f (conf=%.2f) near %.2f,%.2f",
                    cand.floor_x, cand.floor_y, cand.confidence,
                    k.floor_x, k.floor_y,
                )
                break
        if not suppressed:
            kept.append(cand)

    return kept


def _get_floor_xy(det) -> tuple[Optional[float], Optional[float]]:
    """
    Extract (floor_x, floor_y) from any supported detection type.

    Handles:
    • FloorDetection    (detection/detector.py)   — det.floor_x / det.floor_y
    • Legacy Detection  (old dataclass style)     — det.floor_point np.ndarray
    """
    # FloorDetection (__slots__: floor_x, floor_y)
    if hasattr(det, 'floor_x'):
        fx = det.floor_x
        fy = det.floor_y
        if fx is not None and fy is not None:
            return float(fx), float(fy)
        return None, None

    # Legacy Detection dataclass with floor_point as np.ndarray
    fp = getattr(det, 'floor_point', None)
    if fp is not None:
        try:
            return float(fp[0]), float(fp[1])
        except (IndexError, TypeError):
            return None, None

    return None, None


# ═══════════════════════════════════════════════════════════════════════════
#  Retained utility (used by phase_4 & tests)
# ═══════════════════════════════════════════════════════════════════════════

def pairwise_floor_distances(detections: list) -> Optional[np.ndarray]:
    """
    Return an (N×N) pairwise Euclidean distance matrix for floor positions.

    Parameters
    ----------
    detections : list  — any supported detection type

    Returns
    -------
    np.ndarray shape (N, N) or None if fewer than 2 valid detections.
    """
    from scipy.spatial.distance import cdist

    pts = []
    for det in detections:
        fx, fy = _get_floor_xy(det)
        if fx is not None:
            pts.append([fx, fy])

    if len(pts) < 2:
        return None
    arr = np.array(pts, dtype=np.float64)
    return cdist(arr, arr, metric="euclidean")
