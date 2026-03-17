"""
detector.py — YOLO Person Detection and Floor Projection

Three-layer per-camera pipeline:

  Layer 1  PersonDetector  — YOLOv8n inference → list[Detection]
                              (bbox, confidence, foot_point, class_id)

  Layer 2  Track-point rule — foot_point = BOTTOM-CENTRE of bbox
                              The bottom edge of the box is used as the
                              tracking point because it approximates the
                              person's feet on the floor plane and maps
                              correctly onto the shared floor plane.

  Layer 3  CameraProcessor — orchestrates:
                               raw frame → [undistort] → detect → floor-map
                              Passes already_undistorted=True to HomographyMapper
                              when lens correction was applied, preventing
                              double-undistortion of foot_point coordinates.

Classes
-------
Detection       — raw pixel detection (__slots__, 4 fields)
FloorDetection  — floor-mapped detection (__slots__, 6 fields)
PersonDetector  — thin YOLOv8 wrapper (detect / detect_and_draw)
CameraProcessor — per-camera capture + undistort + detect + project pipeline

Backward-compat
---------------
Detector        — legacy wrapper; kept so phase_4() and run_live() compile
draw_detections — overlay helper; accepts Detection, FloorDetection, or legacy
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np

try:
    from shapely.geometry import Point as _SPoint, Polygon as _SPolygon
    _SHAPELY_OK = True
except ImportError:          # pragma: no cover
    _SHAPELY_OK = False

logger = logging.getLogger(__name__)

# Outward buffer applied to the coverage polygon before filtering.
# Allows detections that are legitimately on the edge of the polygon
# (homography/projection error can place them slightly outside).
_COVERAGE_BUFFER_M = 0.5


# ═══════════════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════════════

class Detection:
    """
    Single YOLO detection from one camera frame.

    Coordinates are in the frame that was passed to PersonDetector.detect()
    — which may be the raw (distorted) image or an already-undistorted copy
    depending on what the caller did upstream.

    Attributes
    ----------
    bbox       : (x1, y1, x2, y2) in pixels
    confidence : float  in [0, 1]
    foot_point : (x, y) in pixels — BOTTOM-CENTRE of bounding box
    class_id   : int    (0 = person in COCO)
    """

    __slots__ = ['bbox', 'confidence', 'foot_point', 'class_id', 'track_id']

    def __init__(
        self,
        bbox: tuple,
        confidence: float,
        foot_point: tuple,
        class_id: int,
        track_id: int = -1,
    ) -> None:
        self.bbox       = bbox        # (x1, y1, x2, y2) floats
        self.confidence = confidence
        self.foot_point = foot_point  # (fx, fy) — bottom-centre
        self.class_id   = class_id
        self.track_id   = track_id    # ByteTrack ID ≥ 1, or -1 (detect-only)

    def __repr__(self) -> str:
        x1, y1, x2, y2 = self.bbox
        fx, fy = self.foot_point
        tid = f" tid={self.track_id}" if self.track_id >= 0 else ""
        return (
            f"Detection(cls={self.class_id}, conf={self.confidence:.2f},{tid} "
            f"bbox=({x1:.0f},{y1:.0f}→{x2:.0f},{y2:.0f}), "
            f"foot=({fx:.1f},{fy:.1f}))"
        )


class FloorDetection:
    """
    Person detection projected onto the factory floor coordinate system.

    floor_x / floor_y are in metres.  Origin and axis orientation follow
    floor_config.json (default: bottom-left origin, X = right, Y = up).

    Attributes
    ----------
    camera_id  : str
    floor_x    : float  — metres
    floor_y    : float  — metres
    confidence : float
    pixel_bbox : (x1, y1, x2, y2) in source-frame pixels
    pixel_foot : (x, y)            in source-frame pixels — bottom-centre of bbox
    """

    __slots__ = ['camera_id', 'floor_x', 'floor_y', 'confidence',
                 'pixel_bbox', 'pixel_foot', 'occlusion_confidence']

    def __init__(
        self,
        camera_id: str,
        floor_x: float,
        floor_y: float,
        confidence: float,
        pixel_bbox: tuple,
        pixel_foot: tuple,
        occlusion_confidence: float = 1.0,
    ) -> None:
        self.camera_id            = camera_id
        self.floor_x              = floor_x
        self.floor_y              = floor_y
        self.confidence           = confidence
        self.pixel_bbox           = pixel_bbox  # (x1, y1, x2, y2)
        self.pixel_foot           = pixel_foot  # (fx, fy)
        self.occlusion_confidence = occlusion_confidence  # 1.0=feet visible, <1=estimated

    def __repr__(self) -> str:
        occ = f"  occ={self.occlusion_confidence:.2f}" if self.occlusion_confidence < 1.0 else ""
        return (
            f"FloorDetection(cam={self.camera_id}, "
            f"floor=({self.floor_x:.2f}, {self.floor_y:.2f})m, "
            f"conf={self.confidence:.2f}{occ})"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  PersonDetector
# ═══════════════════════════════════════════════════════════════════════════

# COCO class ID → human-readable name for every class we support.
# Add more entries here if other classes are needed in future.
COCO_CLASS_NAMES: dict[int, str] = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
}

# Default classes detected when none are specified
_DEFAULT_CLASSES: list[int] = [0, 2, 3, 7]   # person, car, motorcycle, truck


def parse_classes(spec: "str | list[int] | None") -> list[int]:
    """
    Convert a flexible class specification to a sorted list of COCO class IDs.

    Accepted formats
    ----------------
    None / ""         → default classes (person, car, motorcycle, truck)
    "all"             → all COCO classes (no filter)
    "person,truck"    → names resolved to IDs  [0, 7]
    "0,2,3,7"         → already numeric strings → [0, 2, 3, 7]
    [0, 2, 3, 7]      → list of ints passed through unchanged

    Raises ValueError for unknown class names.
    """
    if spec is None or spec == "":
        return list(_DEFAULT_CLASSES)

    if isinstance(spec, list):
        return sorted(set(int(c) for c in spec))

    if spec.lower() == "all":
        return []   # empty list = no filter in Ultralytics

    _name_to_id = {v: k for k, v in COCO_CLASS_NAMES.items()}
    result: list[int] = []
    for token in spec.replace(" ", "").split(","):
        if token.isdigit():
            result.append(int(token))
        elif token in _name_to_id:
            result.append(_name_to_id[token])
        else:
            raise ValueError(
                f"Unknown class '{token}'. "
                f"Known names: {list(_name_to_id.keys())}. "
                f"Or use a numeric COCO class ID."
            )
    return sorted(set(result))


class PersonDetector:
    """
    YOLOv8 multi-class detector (person, car, motorcycle, truck by default).

    Downloads weights from Ultralytics on first use if not cached.

    Parameters
    ----------
    model_name : str
        YOLOv8 variant, e.g. ``"yolov8m.pt"`` (default), ``"yolov8n.pt"``.
    confidence : float
        Minimum confidence threshold [0, 1].  Default 0.5.
    device : str
        ``"auto"`` (default) selects CUDA → MPS → CPU automatically.
    track_point : str
        Floor-projection anchor: ``"bottom"`` (default), ``"center"``, ``"top"``.
    target_classes : str | list[int] | None
        Classes to detect.  Accepts names (``"person,car,truck"``), COCO IDs
        (``"0,2,7"``), or ``None`` / ``""`` for the default set
        (person, car, motorcycle, truck).  Pass ``"all"`` to detect everything.
    """

    def __init__(
        self,
        model_name: str = "yolov8m.pt",
        confidence: float = 0.50,
        device: str = "auto",
        track_point: str = "bottom",
        target_classes: "str | list[int] | None" = None,
    ) -> None:
        self.model_name     = model_name
        self.confidence     = confidence
        self.track_point    = track_point   # "bottom" | "center" | "top"
        self.target_classes = parse_classes(target_classes)
        self.model          = None

        # Auto-select best available device
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda:0"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            except Exception:
                device = "cpu"
        self.device = device

        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            _class_names = (
                [COCO_CLASS_NAMES.get(c, str(c)) for c in self.target_classes]
                if self.target_classes else ["all"]
            )
            logger.info(
                "PersonDetector: loaded '%s' on device=%s — tracking classes: %s",
                model_name, device, _class_names,
            )
        except ImportError:
            logger.warning(
                "ultralytics not installed — PersonDetector running in stub mode "
                "(returns empty list).  Install with: pip install ultralytics"
            )
        except Exception as exc:
            logger.error(
                "Failed to load YOLO model '%s': %s",
                model_name, exc,
            )

    # ──────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on *frame* and return all person detections.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (OpenCV format).  May be raw or already undistorted —
            the detector does not care; the caller is responsible for
            tracking that state (see CameraProcessor).

        Returns
        -------
        list[Detection]
            Sorted by confidence descending.  Empty list on failure or if no
            persons detected.
        """
        if self.model is None:
            return []

        try:
            results = self.model(
                frame,
                conf         = self.confidence,
                iou          = 0.4,
                agnostic_nms = True,
                classes      = self.target_classes or None,  # None = all classes
                device       = self.device,
                verbose      = False,
            )
        except Exception as exc:
            logger.error("YOLO inference error: %s", exc)
            return []

        return self._parse(results)

    def track(self, frame: np.ndarray) -> list[Detection]:
        """
        Run ByteTrack on *frame* and return person detections with stable
        track IDs.

        Uses YOLO's built-in ``model.track(persist=True)`` which runs
        ByteTracker internally and assigns a consistent ``track_id`` to
        each Detection across frames.  The tracker state is stored on this
        model instance, so each camera must have its **own** PersonDetector
        to avoid ID collisions between cameras.

        Returns
        -------
        list[Detection]
            Same as detect(), but Detection.track_id is set to the ByteTrack
            ID (int ≥ 1) instead of -1.
        """
        if self.model is None:
            return []

        try:
            results = self.model.track(
                frame,
                conf         = self.confidence,
                iou          = 0.4,
                agnostic_nms = True,
                classes      = self.target_classes or None,  # None = all classes
                device       = self.device,
                verbose      = False,
                persist      = True,
                tracker      = "bytetrack.yaml",
            )
        except Exception as exc:
            logger.error("YOLO track error: %s — falling back to detect()", exc)
            return self.detect(frame)

        return self._parse(results, use_track_ids=True)

    def detect_and_draw(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, list[Detection]]:
        """
        Detect persons and return ``(annotated_frame, detections)``.

        The annotated frame is a copy of *frame* with green bounding boxes,
        confidence labels, and red foot-point circles drawn on it.
        """
        detections = self.detect(frame)
        annotated  = draw_detections(frame, detections)
        return annotated, detections

    # ──────────────────────────────────────────────────────────────────────
    #  Private
    # ──────────────────────────────────────────────────────────────────────

    def _parse(self, results, use_track_ids: bool = False) -> list[Detection]:
        """Convert raw Ultralytics results to Detection objects.

        Parameters
        ----------
        results        : Ultralytics Results list
        use_track_ids  : bool — if True, read box.id (ByteTrack ID) into
                         Detection.track_id.  Should be True only when
                         called from track().
        """
        detections: list[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                try:
                    xyxy   = box.xyxy[0].cpu().numpy()
                    conf   = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    x1 = float(xyxy[0])
                    y1 = float(xyxy[1])
                    x2 = float(xyxy[2])
                    y2 = float(xyxy[3])

                    # ── Track point (configurable) ───────────────────────
                    # foot_x: horizontal centre of the bounding box
                    # foot_y: bottom (y2), vertical centre, or top (y1)
                    #         controlled by self.track_point
                    foot_x = (x1 + x2) / 2.0
                    if self.track_point == "top":
                        foot_y = y1
                    elif self.track_point == "center":
                        foot_y = (y1 + y2) / 2.0
                    else:                        # "bottom"  (default)
                        foot_y = y2

                    # ByteTrack ID — only available when called from track()
                    tid = -1
                    if use_track_ids and box.id is not None:
                        try:
                            tid = int(box.id[0].cpu().numpy())
                        except Exception:
                            tid = -1

                    detections.append(Detection(
                        bbox       = (x1, y1, x2, y2),
                        confidence = conf,
                        foot_point = (foot_x, foot_y),
                        class_id   = cls_id,
                        track_id   = tid,
                    ))
                except Exception as exc:
                    logger.debug("Box parse error: %s", exc)

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


# ═══════════════════════════════════════════════════════════════════════════
#  CameraProcessor
# ═══════════════════════════════════════════════════════════════════════════

class CameraProcessor:
    """
    Per-camera pipeline: capture → undistort → track → project to floor.

    Integrates ``LensCorrector`` (loaded from the HomographyMapper),
    ``PersonDetector``, and ``HomographyMapper`` into a single call:
    ``process_frame() → (list[FloorDetection], frame | None)``.

    Each CameraProcessor owns its **own** PersonDetector instance so that
    YOLO ByteTrack's internal state (track IDs, Kalman filters) remains
    per-camera and IDs do not collide across cameras.

    Undistortion awareness
    ----------------------
    When lens intrinsics are available the raw frame is undistorted *before*
    detection.  The resulting foot_point coordinates are therefore in
    undistorted pixel space.  ``pixel_to_floor_batch`` is called with
    ``already_undistorted=True`` to prevent double-undistortion inside
    ``HomographyMapper``.

    Parameters
    ----------
    camera_config : dict
        One camera entry from ``cameras.json``
        (must have ``"id"`` and ``"source"`` keys).
    homography : HomographyMapper
        The pre-loaded mapper for this camera.
    detector : PersonDetector
        Detector instance whose model/confidence settings to clone.
        Each CameraProcessor creates its own internal PersonDetector
        from these settings so ByteTrack state stays isolated.
    """

    def __init__(
        self,
        camera_config: dict,
        homography,           # HomographyMapper — avoid circular import
        detector: PersonDetector,
        track_point: str = "bottom",
    ) -> None:
        self.camera_id      = camera_config["id"]
        self.homography     = homography

        # Per-camera track_point: prefer cameras.json entry, fall back to arg
        _cfg_tp = camera_config.get("track_point", track_point)

        # Each camera gets its own PersonDetector so ByteTrack track IDs
        # stay per-camera and don't bleed across cameras.
        self.detector = PersonDetector(
            model_name     = detector.model_name,
            confidence     = detector.confidence,
            device         = detector.device,
            track_point    = _cfg_tp,
            target_classes = detector.target_classes,
        )

        # Foot estimator — corrects foot positions when crowd occludes bbox bottom
        from detection.foot_estimator import FootEstimator
        self._foot_estimator = FootEstimator()

        # Reuse the LensCorrector already loaded inside the mapper
        self.lens_corrector = homography.lens_corrector

        # ── Coverage polygon (used to reject out-of-view projections) ────────
        # Built from floor_coverage_polygon in cameras.json, expanded by
        # _COVERAGE_BUFFER_M to tolerate small homography projection errors at
        # the edges.  Falls back to the loose bounding-box check when Shapely
        # is unavailable or the polygon is missing / degenerate.
        self._coverage_poly = None
        raw_poly = camera_config.get("floor_coverage_polygon", [])
        if _SHAPELY_OK and len(raw_poly) >= 3:
            try:
                poly = _SPolygon(raw_poly)
                if poly.is_valid:
                    self._coverage_poly = poly.buffer(_COVERAGE_BUFFER_M)
                else:
                    self._coverage_poly = poly.buffer(0).buffer(_COVERAGE_BUFFER_M)
                logger.debug(
                    "[%s] Coverage polygon loaded (%d pts, buffer=%.1f m)",
                    camera_config["id"], len(raw_poly), _COVERAGE_BUFFER_M,
                )
            except Exception as exc:
                logger.warning("[%s] Could not build coverage polygon: %s", camera_config["id"], exc)

        # Persist source so reset() can re-open it
        self._source            = camera_config["source"]
        self.cap                = self._open_cap(self._source)

        # State for annotation without re-running inference
        self._last_detections:  list[Detection]      = []
        self._last_frame:       Optional[np.ndarray] = None

    # ──────────────────────────────────────────────────────────────────────
    #  Main pipeline
    # ──────────────────────────────────────────────────────────────────────

    def process_frame(
        self,
    ) -> tuple[list[FloorDetection], Optional[np.ndarray]]:
        """
        Read one frame, optionally undistort, detect, project to floor.

        Returns
        -------
        (floor_detections, frame)

        ``floor_detections`` may be empty if:
          • the capture failed,
          • no persons were detected, or
          • the homography is not yet calibrated.

        ``frame`` is the (possibly undistorted) BGR image used for
        detection, or ``None`` if the capture failed.
        """
        if not self.is_active():
            return [], None

        ret, raw = self.cap.read()
        if not ret or raw is None:
            logger.warning("[%s] Frame read failed.", self.camera_id)
            return [], None

        # ── Step 2: optional lens undistortion ───────────────────────────
        if self.lens_corrector.is_calibrated:
            frame = self.lens_corrector.undistort_frame(raw)
        else:
            frame = raw

        self._last_frame = frame

        # ── Step 3: person detection + ByteTrack tracking ────────────────
        # Use track() so each detection carries a stable track_id across
        # frames for the same physical person.  Falls back to detect() on
        # error (handled inside PersonDetector.track()).
        detections = self.detector.track(frame)
        self._last_detections = detections          # always save for annotation

        if not detections:
            return [], frame

        if not self.homography.is_calibrated:
            # Can annotate but cannot floor-map
            return [], frame

        # ── Step 4: occlusion-corrected foot points ───────────────────────
        # FootEstimator checks whether each detection's bottom is hidden by
        # another person's bbox and extrapolates the true foot position if so.
        # For unoccluded persons it returns the raw bbox bottom unchanged.
        foot_estimates = self._foot_estimator.estimate(detections)

        foot_points = np.array(
            [(est.foot_x, est.foot_y) for est in foot_estimates],
            dtype=np.float64,
        )

        # ── Step 5: batch floor projection ───────────────────────────────
        # already_undistorted=True  ← we undistorted the frame in step 2,
        #   so foot_points are in undistorted pixel space.  Tell the mapper
        #   NOT to undistort them again.
        # already_undistorted=False ← raw frame; mapper may still undistort
        #   the points if _lens_corrected=True in the stored homography.
        already_undistorted = self.lens_corrector.is_calibrated
        floor_coords = self.homography.pixel_to_floor_batch(
            foot_points,
            already_undistorted=already_undistorted,
        )

        if floor_coords is None:
            return [], frame

        # ── Step 6: build FloorDetection objects (with coverage filter) ──────
        # Primary filter: reject projections outside the camera's own
        # floor_coverage_polygon (+ _COVERAGE_BUFFER_M outward expansion).
        # Fallback: loose axis-aligned bounding box when the polygon is
        # unavailable (Shapely not installed or no polygon in config).
        _FX_MIN, _FX_MAX = -2.0, 15.0   # hard-limit safety net (floor + 2 m)
        _FY_MIN, _FY_MAX = -2.0, 38.0

        floor_detections: list[FloorDetection] = []
        for det, fxy, est in zip(detections, floor_coords, foot_estimates):
            fx, fy = float(fxy[0]), float(fxy[1])

            # Hard bounding-box check first (fast, catches wild artefacts)
            if not (_FX_MIN <= fx <= _FX_MAX and _FY_MIN <= fy <= _FY_MAX):
                logger.debug(
                    "[%s] Discarding far-out-of-bounds projection (%.2f, %.2f)",
                    self.camera_id, fx, fy,
                )
                continue

            # Precise coverage-polygon check (rejects detections outside the
            # camera's actual field-of-view on the floor)
            if self._coverage_poly is not None:
                if not self._coverage_poly.contains(_SPoint(fx, fy)):
                    logger.debug(
                        "[%s] Discarding projection outside coverage polygon (%.2f, %.2f)",
                        self.camera_id, fx, fy,
                    )
                    continue

            floor_detections.append(FloorDetection(
                camera_id            = self.camera_id,
                floor_x              = fx,
                floor_y              = fy,
                confidence           = det.confidence,
                pixel_bbox           = det.bbox,
                pixel_foot           = (est.foot_x, est.foot_y),
                occlusion_confidence = est.occlusion_confidence,
            ))

        return floor_detections, frame

    # ──────────────────────────────────────────────────────────────────────
    #  Annotation
    # ──────────────────────────────────────────────────────────────────────

    def get_annotated_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the most recent detections onto *frame* and return the copy.

        Uses the Detection objects stored during the last ``process_frame()``
        call; does NOT re-run inference.  Returns an unmodified copy if no
        detections have been stored yet.
        """
        if not self._last_detections:
            return frame.copy()
        return draw_detections(frame, self._last_detections)

    def draw_grid_overlay(
        self,
        frame: np.ndarray,
        floor_w: float,
        floor_h: float,
        step_m: float  = 1.0,
        major_every: float = 5.0,
        alpha: float   = 0.45,
    ) -> np.ndarray:
        """
        Project the floor coordinate grid back onto *frame* using the inverse
        homography, returning an annotated copy.

        Each floor grid line (vertical: x = n·step_m, horizontal: y = n·step_m)
        is projected to two pixel end-points and drawn as a line.  OpenCV clips
        lines that extend outside the image boundary automatically.

        Minor lines (1 m) are drawn in semi-transparent green.
        Major lines (5 m) are slightly thicker and brighter.

        If the homography is not calibrated the frame is returned unchanged.

        Parameters
        ----------
        frame       : BGR source image (not modified).
        floor_w     : floor width  in metres  (from floor_config.json).
        floor_h     : floor height in metres  (from floor_config.json).
        step_m      : grid step in metres (default 1.0).
        major_every : metres between major (thicker) lines (default 5.0).
        alpha       : blend factor — 0 = invisible, 1 = fully opaque (default 0.45).
        """
        if not self.homography.is_calibrated:
            return frame.copy()

        out     = frame.copy()
        overlay = frame.copy()

        img_h, img_w = frame.shape[:2]

        # Color palette (BGR)
        _MINOR = (20,  240,  20)   # bright green (same as major)
        _MAJOR = (20,  240,  20)   # bright green for 5 m lines
        _LABEL = (255, 255, 255)   # white text

        major_step_n = max(1, int(round(major_every / step_m)))

        def _project(fx: float, fy: float):
            """Floor metres → pixel int-tuple, or None if out of range."""
            uv = self.homography.floor_to_pixel(fx, fy)
            if uv is None:
                return None
            return (int(round(uv[0])), int(round(uv[1])))

        n_x = int(round(floor_w / step_m))
        n_y = int(round(floor_h / step_m))

        # ── Vertical lines: x = i * step_m ───────────────────────────────
        for i in range(n_x + 1):
            fx   = i * step_m
            pt1  = _project(fx, 0.0)
            pt2  = _project(fx, floor_h)
            if pt1 is None or pt2 is None:
                continue
            is_major = (i % major_step_n == 0)
            col      = _MAJOR if is_major else _MINOR
            thick    = 2      if is_major else 1
            cv2.line(overlay, pt1, pt2, col, thick, cv2.LINE_AA)
            # Label every major vertical line at the bottom of the image
            if is_major and 0 <= pt2[0] < img_w:
                lbl = f"{int(fx)}m"
                cv2.putText(overlay, lbl,
                            (pt2[0] + 3, min(img_h - 8, pt2[1] + 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, _LABEL, 1, cv2.LINE_AA)

        # ── Horizontal lines: y = j * step_m ─────────────────────────────
        for j in range(n_y + 1):
            fy   = j * step_m
            pt1  = _project(0.0,   fy)
            pt2  = _project(floor_w, fy)
            if pt1 is None or pt2 is None:
                continue
            is_major = (j % major_step_n == 0)
            col      = _MAJOR if is_major else _MINOR
            thick    = 2      if is_major else 1
            cv2.line(overlay, pt1, pt2, col, thick, cv2.LINE_AA)
            # Label every major horizontal line on the left edge
            if is_major and 0 <= pt1[1] < img_h:
                lbl = f"{int(fy)}m"
                cv2.putText(overlay, lbl,
                            (max(0, pt1[0] + 3), pt1[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, _LABEL, 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)
        return out

    # ──────────────────────────────────────────────────────────────────────
    #  Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def release(self) -> None:
        """Release the VideoCapture handle."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def reset(self) -> None:
        """Release and re-open the video source from the beginning."""
        self.release()
        self._last_detections = []
        self._last_frame      = None
        self.cap = self._open_cap(self._source)
        # Reset ByteTrack state so IDs restart cleanly after a video reset
        if self.detector.model is not None:
            try:
                self.detector.model.reset_tracker()
            except Exception:
                pass   # not all YOLO versions expose reset_tracker()

    def is_active(self) -> bool:
        """True if the VideoCapture is open and ready to read."""
        return self.cap is not None and self.cap.isOpened()

    # ──────────────────────────────────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _open_cap(source: str) -> cv2.VideoCapture:
        """Open a VideoCapture with RTSP-over-TCP for network cameras."""
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        if str(source).isdigit():
            return cv2.VideoCapture(int(source))
        return cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)


# ═══════════════════════════════════════════════════════════════════════════
#  Annotation helper
# ═══════════════════════════════════════════════════════════════════════════

def draw_detections(
    frame: np.ndarray,
    detections: list,
    color: tuple[int, int, int] = (0, 210, 0),
) -> np.ndarray:
    """
    Draw bounding boxes, labels, and foot-point circles on a copy of *frame*.

    Accepts ``Detection``, ``FloorDetection``, or legacy dataclass-style
    detection objects.

    Parameters
    ----------
    frame      : np.ndarray  — BGR source image (not modified)
    detections : list        — any mix of Detection / FloorDetection
    color      : BGR tuple   — bounding box / label colour

    Returns
    -------
    np.ndarray  — annotated copy
    """
    out = frame.copy()

    for det in detections:
        # ── Extract fields based on type ─────────────────────────────────
        if isinstance(det, Detection):
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            fx,  fy         = int(det.foot_point[0]), int(det.foot_point[1])
            conf            = det.confidence
            label           = f"person {conf:.2f}"
            floor_str       = None

        elif isinstance(det, FloorDetection):
            x1, y1, x2, y2 = [int(v) for v in det.pixel_bbox]
            fx,  fy         = int(det.pixel_foot[0]), int(det.pixel_foot[1])
            conf            = det.confidence
            floor_str       = f"({det.floor_x:.2f},{det.floor_y:.2f})m"
            label           = f"person {conf:.2f}  {floor_str}"

        else:
            # Legacy dataclass Detection (backward compat)
            try:
                x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
                fx,  fy         = det.foot_pixel
                conf            = det.confidence
                label           = f"person {conf:.2f}"
                floor_str       = None
                if getattr(det, "floor_point", None) is not None:
                    fp = det.floor_point
                    floor_str = f"({fp[0]:.2f},{fp[1]:.2f})m"
                    label = f"{label}  {floor_str}"
            except AttributeError:
                continue

        # ── Bounding box ─────────────────────────────────────────────────
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # ── Label with background ─────────────────────────────────────────
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
        )
        lbl_y = max(y1 - 4, th + 6)
        cv2.rectangle(
            out,
            (x1, lbl_y - th - 5),
            (x1 + tw + 6, lbl_y + 1),
            color, -1,
        )
        cv2.putText(
            out, label, (x1 + 3, lbl_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA,
        )

        # ── Foot-point marker (white ring + red fill) ─────────────────────
        cv2.circle(out, (fx, fy), 7, (255, 255, 255), 2)
        cv2.circle(out, (fx, fy), 4, (0, 30, 210), -1)

    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Backward-compatibility shim  (phase_4, run_live use Detector / detect_batch)
# ═══════════════════════════════════════════════════════════════════════════

class Detector:
    """
    Legacy wrapper around PersonDetector.

    Kept so that ``phase_4()`` and ``run_live()`` continue to compile without
    modification.  New code should use ``PersonDetector`` + ``CameraProcessor``
    directly.

    API difference from new classes
    --------------------------------
    ``detect(frame, camera_id, mapper)`` returns ``list[FloorDetection]`` when
    a calibrated mapper is provided, ``list[Detection]`` otherwise — the
    caller (fuser, renderer) needs to handle both.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        target_classes: Optional[list[int]] = None,
        confidence_threshold: float = 0.45,
        use_tracking: bool = True,
        device: str = "cpu",
    ) -> None:
        self._pd = PersonDetector(
            model_name  = model_path,
            confidence  = confidence_threshold,
            device      = device,
        )
        self.use_tracking = use_tracking
        logger.info(
            "Detector (compat wrapper) initialised.  "
            "use_tracking=%s (tracking not yet implemented in PersonDetector).",
            use_tracking,
        )

    def detect(
        self,
        frame: np.ndarray,
        camera_id: str = "",
        mapper=None,
    ) -> list:
        """Detect persons; optionally project to floor via *mapper*."""
        dets = self._pd.detect(frame)

        if mapper is not None and mapper.is_calibrated and dets:
            foot_pts = np.array(
                [(d.foot_point[0], d.foot_point[1]) for d in dets],
                dtype=np.float64,
            )
            floor_coords = mapper.pixel_to_floor_batch(
                foot_pts, already_undistorted=False
            )
            if floor_coords is not None:
                return [
                    FloorDetection(
                        camera_id  = camera_id,
                        floor_x    = float(fxy[0]),
                        floor_y    = float(fxy[1]),
                        confidence = d.confidence,
                        pixel_bbox = d.bbox,
                        pixel_foot = d.foot_point,
                    )
                    for d, fxy in zip(dets, floor_coords)
                ]
        return dets

    def detect_batch(
        self,
        frames: dict[str, np.ndarray],
        mappers: Optional[dict] = None,
    ) -> dict[str, list]:
        """Run detect() on every entry in *frames*."""
        return {
            cam_id: self.detect(
                frame,
                camera_id = cam_id,
                mapper    = mappers.get(cam_id) if mappers else None,
            )
            for cam_id, frame in frames.items()
        }
