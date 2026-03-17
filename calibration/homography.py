"""
homography.py — High-Accuracy Floor Homography

Transforms camera pixel coordinates → factory floor coordinates (metres)
using a 4-layer accuracy pipeline:

  Layer 0  Lens undistortion   (optional — uses LensCorrector if --intrinsic
                                was run; reduces edge errors from 0.5–2 m to
                                0.05–0.1 m)
  Layer 1  Hartley normalisation  (prevents numerical instability when pixel
                                   coords ~1920 and floor coords ~50 have very
                                   different scales)
  Layer 2  USAC_MAGSAC / RANSAC  (robust outlier rejection; USAC_MAGSAC is
                                  self-tuning on OpenCV 4.5+)
  Layer 3  Levenberg-Marquardt   (nonlinear refinement on inliers to squeeze
                                  sub-mm accuracy)

Calibrated homography is saved to:
  • config/homography_{cam_id}.npz   — full-precision, fast reload
  • config/cameras.json              — homography_matrix field (backward-compat)

Usage
-----
    from calibration.homography import HomographyMapper
    mapper = HomographyMapper("cam_1")
    fx, fy = mapper.pixel_to_floor(px, py)
    report = mapper.get_reprojection_error()
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Quality thresholds (metres)
_EXCELLENT  = 0.05
_GOOD       = 0.10
_ACCEPTABLE = 0.30


# ===========================================================================
# HomographyMapper
# ===========================================================================

class HomographyMapper:
    """
    Per-camera pixel → floor homography with a 4-layer accuracy pipeline.

    Parameters
    ----------
    camera_id : str
    config_path : str | Path
        Directory containing cameras.json and where .npz files are saved.
        Accepts either a string (e.g. ``"config/"``) or a :class:`Path`.
    """

    def __init__(
        self,
        camera_id: str,
        config_path: Union[str, Path] = "config/",
    ) -> None:
        self.camera_id   = camera_id
        self.config_path = str(config_path).rstrip("/") + "/"

        self.H:            Optional[np.ndarray] = None   # 3×3  pixels → floor
        self.H_inv:        Optional[np.ndarray] = None   # 3×3  floor → pixels
        self.image_points: Optional[np.ndarray] = None   # (N,2) raw pixel coords
        self.floor_points: Optional[np.ndarray] = None   # (N,2) floor metres
        self.inlier_mask:  Optional[np.ndarray] = None   # (N,1) uint8

        self._working_points: Optional[np.ndarray] = None  # post-undistortion
        self._lens_corrected: bool  = False
        self._method_used:    str   = "unknown"
        self._error_report:   Optional[dict] = None
        # True when calibration points were collected on an already-undistorted
        # frame (e.g. by CalibrationTool when lens_corrector.is_calibrated).
        # Layer 0 must skip re-undistorting them, but _lens_corrected stays True
        # so that pixel_to_floor() undistorts live camera pixels at runtime.
        self._pts_are_undistorted: bool = False

        # ------------------------------------------------------------------
        # Attach LensCorrector (may be uncalibrated — that is fine)
        # ------------------------------------------------------------------
        from calibration.lens_correction import LensCorrector
        self.lens_corrector = LensCorrector(camera_id, self.config_path)

        self._load()

    # ------------------------------------------------------------------
    # Load / initialise
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """
        Try to load from .npz first (fastest path).
        Fall back to computing from calibration_points in cameras.json.
        """
        npz_path = self._npz_path()
        if os.path.exists(npz_path):
            if self._load_from_npz(npz_path):
                return

        # Compute from cameras.json calibration points
        self._load_points_from_config()
        if (
            self.image_points is not None
            and self.floor_points is not None
            and len(self.image_points) >= 4
        ):
            self._compute_high_accuracy_homography()

    def _load_from_npz(self, npz_path: str) -> bool:
        """Load pre-computed homography from .npz. Returns True on success."""
        try:
            data = np.load(npz_path, allow_pickle=True)
            self.H               = data["H"]
            self.H_inv           = data["H_inv"]
            self.image_points    = data["image_points"]
            self.floor_points    = data["floor_points"]
            self._working_points = data["working_points"]
            self.inlier_mask     = data["inlier_mask"]
            self._lens_corrected = bool(data["lens_corrected"])
            self._method_used    = str(data["method_used"])

            # Rebuild error report from saved metrics
            self._error_report = self._compute_error_report(
                self.H, self._working_points, self.floor_points, self.inlier_mask
            )

            logger.info(
                "[%s] Homography loaded from %s  (mean=%.4f m, lens=%s)",
                self.camera_id, npz_path,
                self._error_report.get("mean_error_m", 0),
                self._lens_corrected,
            )
            return True
        except Exception as exc:
            logger.warning("[%s] Failed to load %s: %s", self.camera_id, npz_path, exc)
            return False

    def _load_points_from_config(self) -> None:
        """
        Read calibration_points from cameras.json for this camera.

        Resolution scaling
        ------------------
        If the operator used a screenshot (or any frame with a different
        resolution than the live feed), the saved pixel coordinates are in
        the *calibration* image space.  This method reads the optional
        ``calibration_frame_size`` field and, if the live video has a
        different resolution (but same aspect ratio), scales all image_points
        so that the homography is computed in live-frame pixel space.
        """
        json_path = os.path.join(self.config_path, "cameras.json")
        if not os.path.exists(json_path):
            logger.warning("[%s] cameras.json not found at %s", self.camera_id, json_path)
            return
        try:
            with open(json_path) as f:
                config = json.load(f)
            entry = next(
                (c for c in config["cameras"] if c["id"] == self.camera_id), None
            )
            if entry is None:
                logger.error("[%s] Camera not found in cameras.json", self.camera_id)
                return
            img_raw = entry["calibration_points"]["image_points"]
            flr_raw = entry["calibration_points"]["floor_points"]
            if img_raw and flr_raw:
                self.image_points = np.array(img_raw, dtype=np.float64)
                self.floor_points = np.array(flr_raw, dtype=np.float64)

                # ── Denormalize if stored as [0, 1] fractions ─────────────
                # calibrate.py writes coordinate_format="normalized" and
                # calibration_frame_size=[w, h] alongside the points.
                # We restore them to calibration-space pixels here so that
                # the subsequent live-resolution scaling step works exactly
                # as it did for the legacy (raw-pixel) format.
                coord_fmt  = entry["calibration_points"].get("coordinate_format")
                calib_size = entry["calibration_points"].get("calibration_frame_size")
                if coord_fmt == "normalized" and calib_size and len(calib_size) == 2:
                    calib_w, calib_h = float(calib_size[0]), float(calib_size[1])
                    if calib_w > 0 and calib_h > 0:
                        self.image_points[:, 0] *= calib_w
                        self.image_points[:, 1] *= calib_h
                        logger.info(
                            "[%s] Denormalized image_points from [0,1] → %dx%d px space",
                            self.camera_id, int(calib_w), int(calib_h),
                        )
                elif calib_size and len(calib_size) == 2:
                    # Legacy raw-pixel path: calib_size present, no normalisation flag
                    calib_size = entry["calibration_points"].get("calibration_frame_size")

                # ── Resolution scaling (existing path) ────────────────────
                # If calibration was done at a different resolution than the
                # live feed, rescale image_points to live-frame pixel space.
                calib_size = entry["calibration_points"].get("calibration_frame_size")
                if calib_size and len(calib_size) == 2:
                    calib_w, calib_h = float(calib_size[0]), float(calib_size[1])
                    # Peek at the live-source frame size
                    live_w, live_h = self._get_live_frame_size(entry)
                    if live_w and live_h:
                        sx = live_w / calib_w
                        sy = live_h / calib_h
                        if abs(sx - 1.0) > 0.01 or abs(sy - 1.0) > 0.01:
                            self.image_points[:, 0] *= sx
                            self.image_points[:, 1] *= sy
                            logger.info(
                                "[%s] Rescaled calibration points from "
                                "%dx%d → %dx%d  (sx=%.3f sy=%.3f)",
                                self.camera_id,
                                int(calib_w), int(calib_h),
                                int(live_w),  int(live_h),
                                sx, sy,
                            )
                            print(
                                f"  ℹ  [{self.camera_id}] Calibration was at "
                                f"{int(calib_w)}×{int(calib_h)}, "
                                f"live feed is {int(live_w)}×{int(live_h)} — "
                                f"pixel coords auto-scaled (sx={sx:.3f}, sy={sy:.3f})."
                            )

            # Flag set by CalibrationTool when points were clicked on an
            # already-undistorted frame — skip Layer 0 to avoid double-undistortion.
            self._pts_are_undistorted = bool(
                entry["calibration_points"].get("points_are_undistorted", False)
            )
        except Exception as exc:
            logger.warning("[%s] Error reading cameras.json: %s", self.camera_id, exc)

    def _get_live_frame_size(self, cam_entry: dict) -> tuple:
        """
        Return (width, height) of a real frame from the camera source.

        Uses cv2.VideoCapture for videos / RTSP, or cv2.imread for images.
        Returns (None, None) on any failure — caller ignores scaling in that case.
        """
        import cv2  # local import — cv2 may not be available at module load
        source = cam_entry.get("source", "")
        if not source:
            return None, None
        # Resolve relative paths against the project root (parent of config/)
        src_path = os.path.join(os.path.dirname(self.config_path.rstrip("/")), source)
        if not os.path.exists(src_path):
            src_path = source  # try as-is (absolute path or URL)

        # Still image
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        if os.path.splitext(src_path)[1].lower() in img_exts:
            img = cv2.imread(src_path)
            if img is not None:
                return float(img.shape[1]), float(img.shape[0])
            return None, None

        # Video / RTSP
        try:
            cap = cv2.VideoCapture(src_path)
            if cap.isOpened():
                w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                cap.release()
                if w > 0 and h > 0:
                    return float(w), float(h)
        except Exception:
            pass
        return None, None

    # ------------------------------------------------------------------
    # 4-Layer accuracy pipeline
    # ------------------------------------------------------------------

    def _compute_high_accuracy_homography(self) -> None:
        """
        4-layer pipeline: undistort → normalise → robust fit → LM refinement.
        Stores H, H_inv, inlier_mask, error report, and saves to .npz.
        """
        img_pts = self.image_points
        flr_pts = self.floor_points
        n = len(img_pts)

        print(f"\n  ── Homography Computation [{self.camera_id}] ──────────────")
        print(f"  Calibration points : {n}")

        # ------------------------------------------------------------------
        # Layer 0: Lens undistortion
        # ------------------------------------------------------------------
        if self._pts_are_undistorted:
            # CalibrationTool already showed an undistorted frame and the
            # operator clicked on that frame, so the stored pixels are already
            # in undistorted image space — re-undistorting them would corrupt
            # the data.  We still set _lens_corrected so that pixel_to_floor()
            # undistorts live camera pixels at runtime.
            working = img_pts.copy()
            self._lens_corrected = self.lens_corrector.is_calibrated
            if self._lens_corrected:
                lc_rms = self.lens_corrector.get_reprojection_error()
                print(f"  L0 Lens undistortion : SKIPPED  (points already undistorted, "
                      f"intrinsic RMS={lc_rms:.4f} px — runtime will undistort live pixels)")
            else:
                print("  L0 Lens undistortion : SKIPPED  (points already undistorted, no intrinsics)")
        elif self.lens_corrector.is_calibrated:
            working = self.lens_corrector.undistort_points_batch(img_pts)
            self._lens_corrected = True
            lc_rms = self.lens_corrector.get_reprojection_error()
            print(f"  L0 Lens undistortion : APPLIED  (intrinsic RMS={lc_rms:.4f} px)")
        else:
            working = img_pts.copy()
            self._lens_corrected = False
            print("  L0 Lens undistortion : SKIPPED  (run --intrinsic for +0.1–2 m accuracy)")

        self._working_points = working

        # ------------------------------------------------------------------
        # Layer 1: Hartley normalisation
        # ------------------------------------------------------------------
        img_norm, T_img = _normalize_points(working)
        flr_norm, T_flr = _normalize_points(flr_pts)
        print("  L1 Hartley normalisation : done")

        # ------------------------------------------------------------------
        # Layer 2: USAC_MAGSAC → RANSAC fallback
        # ------------------------------------------------------------------
        H_norm, self.inlier_mask, self._method_used = _robust_homography(
            img_norm, flr_norm
        )

        if H_norm is None:
            logger.error("[%s] Robust homography failed.", self.camera_id)
            print("  ✗  Homography computation failed.")
            return

        # Denormalise: H = inv(T_flr) @ H_norm @ T_img
        H = np.linalg.inv(T_flr) @ H_norm @ T_img
        H /= H[2, 2]

        inlier_count = int(self.inlier_mask.sum()) if self.inlier_mask is not None else n
        print(f"  L2 {self._method_used:12s}        : {inlier_count}/{n} inliers")

        # ------------------------------------------------------------------
        # Layer 3: Levenberg-Marquardt refinement on inliers
        # ------------------------------------------------------------------
        H_final, lm_status = _lm_refine(H, working, flr_pts, self.inlier_mask)
        print(f"  L3 LM refinement         : {lm_status}")

        self.H     = H_final
        self.H_inv = np.linalg.inv(self.H)

        # ------------------------------------------------------------------
        # Error report
        # ------------------------------------------------------------------
        self._error_report = self._compute_error_report(
            self.H, self._working_points, self.floor_points, self.inlier_mask
        )
        _print_error_report(self._error_report, self.camera_id)

        # ------------------------------------------------------------------
        # Save
        # ------------------------------------------------------------------
        self._save()

    # ------------------------------------------------------------------
    # Public API — new interface
    # ------------------------------------------------------------------

    def pixel_to_floor(
        self,
        u: float,
        v: float,
        already_undistorted: bool = False,
    ) -> Optional[tuple[float, float]]:
        """
        Transform a single pixel coordinate to floor metres.

        Parameters
        ----------
        u, v : float
            Pixel column and row.
        already_undistorted : bool
            Set True if the source frame was already passed through
            ``lens_corrector.undistort_frame()`` before detection.
            Prevents double-undistortion.

        Returns
        -------
        (x, y) floor metres, or None if not calibrated.
        """
        if self.H is None:
            return None

        pu, pv = float(u), float(v)

        if self._lens_corrected and not already_undistorted:
            pu, pv = self.lens_corrector.undistort_point(pu, pv)

        pt = np.array([[[pu, pv]]], dtype=np.float64)
        result = cv2.perspectiveTransform(pt, self.H)
        xy = result[0, 0]
        return (float(xy[0]), float(xy[1]))

    def pixel_to_floor_batch(
        self,
        points: np.ndarray,
        already_undistorted: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Vectorised pixel → floor for (N, 2) array.  ~100× faster than a loop.

        Parameters
        ----------
        points : np.ndarray, shape (N, 2)
        already_undistorted : bool

        Returns
        -------
        np.ndarray, shape (N, 2), or None.
        """
        if self.H is None:
            return None

        pts = points.astype(np.float64)

        if self._lens_corrected and not already_undistorted:
            pts = self.lens_corrector.undistort_points_batch(pts)

        pts_cv = pts.reshape(-1, 1, 2)
        result = cv2.perspectiveTransform(pts_cv, self.H)
        return result.reshape(-1, 2)

    def floor_to_pixel(
        self,
        x: float,
        y: float,
    ) -> Optional[tuple[float, float]]:
        """
        Inverse transform: floor metres → pixel.

        Used to project known floor grid points onto the camera image (e.g.
        for overlay visualisation or checking calibration accuracy visually).

        Parameters
        ----------
        x, y : float
            Floor coordinates in metres.

        Returns
        -------
        (u, v) pixel, or None if not calibrated.
        """
        if self.H_inv is None:
            return None
        pt = np.array([[[x, y]]], dtype=np.float64)
        result = cv2.perspectiveTransform(pt, self.H_inv)
        uv = result[0, 0]
        return (float(uv[0]), float(uv[1]))

    def get_reprojection_error(self) -> Optional[dict]:
        """
        Full reprojection error report over all calibration points.

        Returns
        -------
        dict or None ::

            {
                "mean_error_m":       float,   # metres
                "max_error_m":        float,
                "per_point_errors_m": list[float],
                "worst_point_idx":    int,
                "inlier_count":       int,
                "total_points":       int,
                "quality":            str,   # EXCELLENT / GOOD / ACCEPTABLE / POOR
                "lens_corrected":     bool,
                "method":             str,
            }
        """
        if self.H is None or self._working_points is None:
            return None
        if self._error_report is not None:
            return self._error_report
        self._error_report = self._compute_error_report(
            self.H, self._working_points, self.floor_points, self.inlier_mask
        )
        return self._error_report

    def is_valid(self) -> bool:
        """
        Return True if the homography exists and mean reprojection error < 0.3 m.
        """
        report = self.get_reprojection_error()
        if report is None:
            return False
        return report["mean_error_m"] < _ACCEPTABLE

    @property
    def is_calibrated(self) -> bool:
        """True if the homography matrix H has been computed or loaded."""
        return self.H is not None

    # ------------------------------------------------------------------
    # Backward-compatibility aliases (used by calibrate.py / detector.py)
    # ------------------------------------------------------------------

    def compute(self, apply_undistortion: bool = True, **_kwargs) -> bool:
        """
        (Re-)compute the homography from the current calibration points.
        Alias kept for compatibility with ``calibration/calibrate.py``.

        The ``apply_undistortion`` parameter is honoured: if False, the
        LensCorrector is temporarily bypassed for this run.
        """
        # Reload the latest points from cameras.json before re-computing
        self._load_points_from_config()
        if self.image_points is None or len(self.image_points) < 4:
            logger.error("[%s] Not enough calibration points.", self.camera_id)
            return False

        # Temporarily skip lens correction if caller asked for that
        _orig_lc = self.lens_corrector.is_calibrated
        if not apply_undistortion:
            # Monkey-patch is_calibrated to False for this computation
            self.lens_corrector.camera_matrix = None
            self.lens_corrector.dist_coeffs   = None

        self._compute_high_accuracy_homography()

        if not apply_undistortion and _orig_lc:
            # Restore (reload from config)
            from calibration.lens_correction import LensCorrector
            self.lens_corrector = LensCorrector(self.camera_id, self.config_path)

        return self.is_calibrated

    def reprojection_error(self) -> Optional[float]:
        """
        Returns mean reprojection error in metres.
        Alias kept for compatibility with ``calibration/calibrate.py``.
        """
        report = self.get_reprojection_error()
        return report["mean_error_m"] if report else None

    def image_to_floor(
        self, image_point: tuple[float, float]
    ) -> Optional[np.ndarray]:
        """Alias for ``pixel_to_floor``, returns np.ndarray shape (2,)."""
        result = self.pixel_to_floor(image_point[0], image_point[1])
        return np.array(result, dtype=np.float32) if result is not None else None

    def floor_to_image(
        self, floor_point: tuple[float, float]
    ) -> Optional[np.ndarray]:
        """Alias for ``floor_to_pixel``, returns np.ndarray shape (2,)."""
        result = self.floor_to_pixel(floor_point[0], floor_point[1])
        return np.array(result, dtype=np.float32) if result is not None else None

    def image_to_floor_batch(self, points: np.ndarray) -> Optional[np.ndarray]:
        """Alias for ``pixel_to_floor_batch``."""
        return self.pixel_to_floor_batch(points)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Save H, points, mask to .npz and update cameras.json."""
        if self.H is None:
            return

        # 1. npz
        npz_path = self._npz_path()
        os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
        np.savez(
            npz_path,
            H              = self.H,
            H_inv          = self.H_inv,
            image_points   = self.image_points,
            floor_points   = self.floor_points,
            working_points = self._working_points,
            inlier_mask    = self.inlier_mask if self.inlier_mask is not None
                             else np.ones((len(self.image_points), 1), dtype=np.uint8),
            lens_corrected = np.array(self._lens_corrected),
            method_used    = np.array(self._method_used),
        )
        logger.info("[%s] Homography saved → %s", self.camera_id, npz_path)

        # 2. cameras.json homography_matrix (backward compat)
        json_path = os.path.join(self.config_path, "cameras.json")
        if not os.path.exists(json_path):
            return
        try:
            with open(json_path) as f:
                config = json.load(f)
            entry = next(
                (c for c in config["cameras"] if c["id"] == self.camera_id), None
            )
            if entry is not None:
                entry["homography_matrix"] = self.H.tolist()
                with open(json_path, "w") as f:
                    json.dump(config, f, indent=2)
        except Exception as exc:
            logger.warning("[%s] Failed to update cameras.json: %s", self.camera_id, exc)

    def _npz_path(self) -> str:
        return os.path.join(self.config_path, f"homography_{self.camera_id}.npz")

    # ------------------------------------------------------------------
    # Error computation (shared by live compute + npz reload)
    # ------------------------------------------------------------------

    def _compute_error_report(
        self,
        H: np.ndarray,
        working_pts: np.ndarray,
        floor_pts: np.ndarray,
        inlier_mask: Optional[np.ndarray],
    ) -> dict:
        """Compute per-point reprojection errors and quality grade."""
        if H is None or working_pts is None or floor_pts is None:
            return {}

        n = len(working_pts)
        pts_cv = working_pts.reshape(-1, 1, 2).astype(np.float64)
        proj   = cv2.perspectiveTransform(pts_cv, H).reshape(-1, 2)
        errors = np.linalg.norm(proj - floor_pts, axis=1).tolist()

        mean_err = float(np.mean(errors))
        max_err  = float(np.max(errors))
        worst    = int(np.argmax(errors))

        inlier_count = (
            int(inlier_mask.sum())
            if inlier_mask is not None
            else n
        )

        if mean_err < _EXCELLENT:
            quality = "EXCELLENT"
        elif mean_err < _GOOD:
            quality = "GOOD"
        elif mean_err < _ACCEPTABLE:
            quality = "ACCEPTABLE"
        else:
            quality = "POOR"

        return {
            "mean_error_m":       mean_err,
            "max_error_m":        max_err,
            "per_point_errors_m": [round(e, 5) for e in errors],
            "worst_point_idx":    worst,
            "inlier_count":       inlier_count,
            "total_points":       n,
            "quality":            quality,
            "lens_corrected":     self._lens_corrected,
            "method":             self._method_used,
        }


# ===========================================================================
# Layer helpers (module-level, reusable)
# ===========================================================================

def _normalize_points(
    pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hartley normalisation: translate centroid to origin, scale so mean
    distance from origin = sqrt(2).

    Parameters
    ----------
    pts : np.ndarray, shape (N, 2)

    Returns
    -------
    pts_norm : np.ndarray, shape (N, 2)
    T        : np.ndarray, shape (3, 3)  — the normalisation transform
    """
    centroid  = np.mean(pts, axis=0)
    shifted   = pts - centroid
    avg_dist  = np.mean(np.sqrt(np.sum(shifted ** 2, axis=1)))
    scale     = np.sqrt(2) / (avg_dist + 1e-10)

    T = np.array(
        [
            [scale,     0, -scale * centroid[0]],
            [0,     scale, -scale * centroid[1]],
            [0,         0,                    1],
        ],
        dtype=np.float64,
    )

    pts_h  = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
    pts_n  = (T @ pts_h.T).T
    return pts_n[:, :2], T


def _robust_homography(
    img_norm: np.ndarray,
    flr_norm: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Attempt USAC_MAGSAC (OpenCV 4.5+), fall back to RANSAC.

    Parameters
    ----------
    img_norm, flr_norm : np.ndarray, shape (N, 2) — normalised coordinates

    Returns
    -------
    H_norm : (3,3) or None
    mask   : (N,1) uint8 inlier mask or None
    method : str label
    """
    pts_src = img_norm.astype(np.float32)
    pts_dst = flr_norm.astype(np.float32)

    # USAC_MAGSAC (self-tuning, best on OpenCV ≥4.5)
    try:
        H, mask = cv2.findHomography(
            pts_src, pts_dst,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=1.0,
            maxIters=10000,
            confidence=0.9999,
        )
        if H is not None:
            return H, mask, "USAC_MAGSAC"
    except (AttributeError, cv2.error):
        pass

    # RANSAC fallback
    H, mask = cv2.findHomography(
        pts_src, pts_dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000,
        confidence=0.999,
    )
    return H, mask, "RANSAC"


def _lm_refine(
    H: np.ndarray,
    img_pts: np.ndarray,
    flr_pts: np.ndarray,
    inlier_mask: Optional[np.ndarray],
) -> tuple[np.ndarray, str]:
    """
    Levenberg-Marquardt refinement of the homography on inlier points.

    Minimises the floor-space reprojection residuals (metres) to sub-mm
    accuracy.  Only accepts the refined H if it improves the mean error.

    Parameters
    ----------
    H           : (3,3) initial homography
    img_pts     : (N,2) undistorted (or raw) image points
    flr_pts     : (N,2) floor points in metres
    inlier_mask : (N,1) uint8 or None

    Returns
    -------
    H_final : (3,3) — refined (or original if refinement made it worse)
    status  : str description
    """
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return H, "SKIPPED (scipy not installed)"

    # Select inlier points
    if inlier_mask is not None:
        mask = inlier_mask.ravel().astype(bool)
        in_img = img_pts[mask]
        in_flr = flr_pts[mask]
    else:
        in_img = img_pts
        in_flr = flr_pts

    if len(in_img) < 4:
        return H, "SKIPPED (< 4 inliers)"

    def residuals(h_flat: np.ndarray) -> np.ndarray:
        Hm   = h_flat.reshape(3, 3)
        ones = np.ones((len(in_img), 1), dtype=np.float64)
        hpts = np.hstack([in_img, ones])          # (N, 3)
        proj = (Hm @ hpts.T).T                    # (N, 3)
        # Avoid division by zero
        w    = proj[:, 2:3]
        w    = np.where(np.abs(w) < 1e-10, 1e-10 * np.sign(w + 1e-20), w)
        pred = proj[:, :2] / w                    # (N, 2) floor coords
        return (pred - in_flr).ravel()            # (2N,)

    err_before = float(np.mean(np.linalg.norm(
        np.array(residuals(H.flatten())).reshape(-1, 2), axis=1
    )))

    try:
        result = least_squares(
            residuals,
            H.flatten(),
            method="lm",
            ftol=1e-12,
            xtol=1e-12,
            max_nfev=5000,
        )
        H_refined      = result.x.reshape(3, 3)
        H_refined     /= H_refined[2, 2]

        err_after = float(np.mean(np.linalg.norm(
            np.array(residuals(H_refined.flatten())).reshape(-1, 2), axis=1
        )))

        if err_after <= err_before:
            improvement = (err_before - err_after) * 1000  # mm
            return H_refined, f"OK  ({err_before*1000:.2f}→{err_after*1000:.2f} mm, Δ={improvement:.2f} mm)"
        else:
            return H, f"REVERTED (refined was worse: {err_after:.4f} > {err_before:.4f} m)"

    except Exception as exc:
        return H, f"FAILED ({exc})"


def _mean_floor_error(H: np.ndarray, img_pts: np.ndarray, flr_pts: np.ndarray) -> float:
    """Compute mean Euclidean floor error (metres) of H over given point pairs."""
    ones = np.ones((len(img_pts), 1), dtype=np.float64)
    hpts = np.hstack([img_pts, ones])
    proj = (H @ hpts.T).T
    w    = proj[:, 2:3]
    w    = np.where(np.abs(w) < 1e-10, 1e-10, w)
    pred = proj[:, :2] / w
    return float(np.mean(np.linalg.norm(pred - flr_pts, axis=1)))


def _print_error_report(report: dict, camera_id: str) -> None:
    """Print a formatted quality report to stdout."""
    if not report:
        return
    q        = report["quality"]
    mean_m   = report["mean_error_m"]
    max_m    = report["max_error_m"]
    n_in     = report["inlier_count"]
    n_tot    = report["total_points"]
    worst    = report["worst_point_idx"]
    per_pt   = report["per_point_errors_m"]
    method   = report["method"]
    lc       = "YES" if report["lens_corrected"] else "NO"

    q_col = {"EXCELLENT": "✓✓", "GOOD": "✓", "ACCEPTABLE": "⚠", "POOR": "✗"}.get(q, "?")
    print(f"  ── Quality Report ───────────────────────────────────────")
    print(f"  {q_col}  {q}  (mean={mean_m*100:.2f} cm,  max={max_m*100:.2f} cm)")
    print(f"  Inliers         : {n_in}/{n_tot}")
    print(f"  Lens corrected  : {lc}")
    print(f"  Method          : {method}")
    print(f"  Worst point     : #{worst+1}  ({max_m*100:.2f} cm)")
    print(f"  Per-point (cm)  : {[round(e*100,2) for e in per_pt]}")
    print(f"  ─────────────────────────────────────────────────────────")

    if q == "POOR":
        print("  ⚠  Poor accuracy.  Suggestions:")
        print("     • Add more calibration points (target ≥10, well spread)")
        print("     • Run --intrinsic first to remove lens distortion")
        print("     • Check that floor_points are correct (X=right, Y=up, metres)")
    elif not report["lens_corrected"]:
        print("  ℹ  Run --intrinsic for this camera to potentially improve accuracy.")
    print()


# ===========================================================================
# Convenience factory: load all cameras
# ===========================================================================

def load_all_homographies(
    cameras_config_path: Union[str, Path] = "config/cameras.json",
) -> dict[str, HomographyMapper]:
    """
    Instantiate a :class:`HomographyMapper` for every camera in cameras.json.

    Mappers are loaded from .npz if available, otherwise computed from the
    calibration_points in cameras.json.

    Parameters
    ----------
    cameras_config_path : str | Path
        Full path to cameras.json  OR  a directory (``config/``).

    Returns
    -------
    dict[str, HomographyMapper]
    """
    p = Path(cameras_config_path)
    if p.is_dir():
        json_path  = p / "cameras.json"
        config_dir = str(p)
    else:
        json_path  = p
        config_dir = str(p.parent)

    with open(json_path) as f:
        config = json.load(f)

    mappers: dict[str, HomographyMapper] = {}
    for cam in config["cameras"]:
        cam_id = cam["id"]
        mappers[cam_id] = HomographyMapper(cam_id, config_dir)
    return mappers
