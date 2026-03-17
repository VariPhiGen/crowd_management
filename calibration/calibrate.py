"""
calibrate.py — Interactive Floor-Point Calibration  (--calibrate <cam_id>)

Guides the operator through picking corresponding point pairs:
  · Click a known floor marker visible in the camera frame  → pixel coordinate
  · Type the measured real-world floor coordinate (X, Y) in metres

The collected pairs are persisted to cameras.json and a high-accuracy
homography is immediately computed and saved so the live pipeline can use it.

If lens-intrinsic calibration has been run beforehand (--intrinsic), the
reference frame is automatically undistorted; all clicked coordinates are in
the corrected image space, which removes lens-distortion error from the
homography computation.

Typical usage
-------------
    python main.py --calibrate cam_1
    python main.py --calibrate cam_1 --source /path/to/video.mp4
    python main.py --calibrate cam_1 --source /path/to/frame.jpg
    python main.py --calibrate cam_1 --source /path/to/frame.png

Image files (.jpg .jpeg .png .bmp .tiff .webp) are supported directly —
no need for a video.  The tool reads the single image and uses it as the
reference frame for the entire session.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Image file extensions treated as static frames (not video)
_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# Maximum display dimensions — keeps window on-screen on typical monitors.
# The frame content is NOT scaled; only the window chrome is resized.
_MAX_WIN_W = 1400
_MAX_WIN_H = 900


def _is_image_path(source: str) -> bool:
    """Return True if *source* is a path to a still image file."""
    return Path(source).suffix.lower() in _IMAGE_EXTS


def _win_size(frame_w: int, frame_h: int) -> tuple[int, int]:
    """
    Return (win_w, win_h) that fits the frame on-screen without distorting it.
    The window is sized to the frame's native resolution, capped at
    (_MAX_WIN_W × _MAX_WIN_H) proportionally so it always fits on the monitor.
    """
    if frame_w <= 0 or frame_h <= 0:
        return _MAX_WIN_W, _MAX_WIN_H
    scale = min(_MAX_WIN_W / frame_w, _MAX_WIN_H / frame_h, 1.0)
    return max(1, int(frame_w * scale)), max(1, int(frame_h * scale))


# ═══════════════════════════════════════════════════════════════════════════
#  CalibrationTool
# ═══════════════════════════════════════════════════════════════════════════

class CalibrationTool:
    """
    Interactive per-camera floor calibration session.

    Parameters
    ----------
    camera_id : str
    source : str
        Video file path, RTSP URL, or integer webcam index as string.
    config_path : str
        Directory containing cameras.json, overlap_zones.json, floor_config.json.
    """

    _WINDOW_TITLE_TPL = (
        "CALIBRATE: {cam_id}  |  Click floor markers  |  "
        "Min 8 pts  |  Q=done  U=undo  V=verify  ESC=quit"
    )
    _MIN_REQUIRED    = 4
    _MIN_RECOMMENDED = 8

    # ──────────────────────────────────────────────────────
    #  Construction
    # ──────────────────────────────────────────────────────

    def __init__(
        self,
        camera_id: str,
        source: str,
        config_path: str = "config/",
    ) -> None:
        self.camera_id   = camera_id
        self.source      = source
        self.config_path = str(config_path).rstrip("/") + "/"

        self.image_points: list[list[float]] = []
        self.floor_points: list[list[float]] = []
        self.current_frame:  Optional[np.ndarray] = None
        self.display_frame:  Optional[np.ndarray] = None

        # Internal rendering state
        self._frame_h:    int   = 0
        self._frame_w:    int   = 0
        self._disp_scale: float = 1.0   # scale applied to the display frame
        self._show_verify: bool = False
        self._temp_H_inv:  Optional[np.ndarray] = None

        # Load floor dimensions for grid rendering
        self._floor_w: float = 50.0
        self._floor_h: float = 30.0
        _floor_cfg = os.path.join(self.config_path, "floor_config.json")
        if os.path.exists(_floor_cfg):
            try:
                with open(_floor_cfg) as _f:
                    _fc = json.load(_f)
                self._floor_w = float(_fc.get("floor_width_m",  50))
                self._floor_h = float(_fc.get("floor_height_m", 30))
            except Exception:
                pass

        # Load LensCorrector (may be uncalibrated — that is fine)
        from calibration.lens_correction import LensCorrector
        self.lens_corrector = LensCorrector(camera_id, self.config_path)

        # Load overlap zone info
        self._overlap_info: dict = self._get_overlap_info()

    # ──────────────────────────────────────────────────────
    #  Overlap zone helpers
    # ──────────────────────────────────────────────────────

    def _get_overlap_info(self) -> dict:
        """
        Check whether this camera is involved in any overlap zone.

        Returns a dict with keys:
          zone_id, other_camera, polygon, x_min, x_max, y_min, y_max
        or an empty dict if no overlap zone is found.
        """
        path = os.path.join(self.config_path, "overlap_zones.json")
        if not os.path.exists(path):
            return {}
        try:
            with open(path) as f:
                data = json.load(f)
            for zone in data.get("overlap_zones", []):
                cams = zone.get("cameras", [])
                if self.camera_id in cams:
                    other = next(c for c in cams if c != self.camera_id)
                    poly  = zone.get("floor_polygon", [])
                    xs    = [p[0] for p in poly]
                    ys    = [p[1] for p in poly]
                    return {
                        "zone_id":      zone["id"],
                        "other_camera": other,
                        "polygon":      poly,
                        "x_min": min(xs), "x_max": max(xs),
                        "y_min": min(ys), "y_max": max(ys),
                    }
        except Exception as exc:
            logger.warning("Could not load overlap_zones.json: %s", exc)
        return {}

    def _is_in_overlap_zone(self, fx: float, fy: float) -> bool:
        """Return True if floor coord (fx, fy) is inside the overlap polygon."""
        poly = self._overlap_info.get("polygon", [])
        return bool(poly) and _point_in_polygon(fx, fy, poly)

    def _count_overlap_points(self) -> int:
        return sum(1 for fp in self.floor_points if self._is_in_overlap_zone(fp[0], fp[1]))

    # ──────────────────────────────────────────────────────
    #  Video capture
    # ──────────────────────────────────────────────────────

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        # Force RTSP-over-TCP for Hikvision / Dahua cameras.
        # Setting the env var before VideoCapture is the recommended approach;
        # it is a no-op for local video files and webcam indices.
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        if self.source.isdigit():
            cap = cv2.VideoCapture(int(self.source))
        else:
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"  ✗  Cannot open source: {self.source}")
            return None
        return cap

    def _grab_frame(self, cap: cv2.VideoCapture) -> Optional[np.ndarray]:
        """Try several times to grab a valid frame."""
        for _ in range(10):
            ret, frame = cap.read()
            if ret and frame is not None:
                return frame
        return None

    def _read_source_frame(self) -> Optional[np.ndarray]:
        """
        Read a single frame from *self.source*.

        Supports:
          • Still image files  (.jpg .jpeg .png .bmp .tiff .webp)
          • Video files        (.mp4 .avi .mkv …)
          • RTSP / webcam      (rtsp://… or digit string)

        Returns the frame as a BGR numpy array, or None on failure.
        """
        # ── Still image ───────────────────────────────────────────────────
        if _is_image_path(self.source):
            frame = cv2.imread(self.source)
            if frame is None:
                print(f"  ✗  Cannot read image file: {self.source}")
                return None
            print(f"  ✓  Loaded image: {self.source}  "
                  f"({frame.shape[1]}×{frame.shape[0]})")
            return frame

        # ── Video / RTSP / webcam ─────────────────────────────────────────
        cap = self._open_capture()
        if cap is None:
            return None
        frame = self._grab_frame(cap)
        cap.release()
        if frame is None:
            print(f"  ✗  Could not grab a frame from: {self.source}")
        return frame

    # ──────────────────────────────────────────────────────
    #  Console helpers
    # ──────────────────────────────────────────────────────

    def _print_tips(self) -> None:
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║  TIPS FOR BEST ACCURACY:                                     ║")
        print("║  1. Spread points across the ENTIRE visible floor            ║")
        print("║     (all four corners + centre + mid-edges)                  ║")
        print("║  2. Include points near ALL image edges — edges matter most! ║")
        print("║  3. Use 8-12 points (more = better, esp. without lens calib) ║")
        print("║  4. Mark floor features you can measure precisely:           ║")
        print("║     tape marks, tile corners, machine bolt holes, paint lines ║")
        print("║  5. Measure floor coords carefully — tape measure or laser   ║")
        print("║  6. Overlap cameras: 3+ SHARED points with SAME floor coords ║")
        print("╚══════════════════════════════════════════════════════════════╝\n")

    def _print_overlap_warning(self) -> None:
        if not self._overlap_info:
            return
        oi = self._overlap_info
        print("═" * 63)
        print(f"  ⚠  OVERLAP: {self.camera_id} overlaps with {oi['other_camera']}")
        print(f"  ⚠  Place 3+ calibration points INSIDE the overlap zone")
        print(f"  ⚠  Overlap zone:  x=[{oi['x_min']:.1f} – {oi['x_max']:.1f}]  "
              f"y=[{oi['y_min']:.1f} – {oi['y_max']:.1f}]  metres")
        print(f"  ⚠  Use IDENTICAL floor coordinates when calibrating {oi['other_camera']}")
        print("═" * 63 + "\n")

    # ──────────────────────────────────────────────────────
    #  Drawing / display
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _draw_crosshair(
        img: np.ndarray,
        cx: int, cy: int,
        color: tuple,
        size: int = 20,
        thick: int = 1,
    ) -> None:
        h, w = img.shape[:2]
        cv2.line(img, (max(0, cx - size), cy), (min(w-1, cx + size), cy), color, thick, cv2.LINE_AA)
        cv2.line(img, (cx, max(0, cy - size)), (cx, min(h-1, cy + size)), color, thick, cv2.LINE_AA)

    def _redraw(self) -> np.ndarray:
        """
        Compose the full display frame: base image + verification grid + points + HUD.
        Returns the frame scaled to fit the screen (using self._disp_scale).
        """
        if self.current_frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        display = self.current_frame.copy()
        h, w = display.shape[:2]

        # Verification grid (drawn below points so points appear on top)
        if self._show_verify and self._temp_H_inv is not None:
            self._draw_verification_grid(display)

        # Calibration points
        for i, (ip, fp) in enumerate(zip(self.image_points, self.floor_points)):
            px, py = int(round(ip[0])), int(round(ip[1]))

            # Precision crosshair (subtle grey)
            self._draw_crosshair(display, px, py, (160, 160, 160), size=22, thick=1)

            # Marker: white border + red fill
            cv2.circle(display, (px, py), 9,  (255, 255, 255), 2)
            cv2.circle(display, (px, py), 6,  (0,   0,   210), -1)

            # Point index label
            label = str(i + 1)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            lx, ly = px + 13, py - 10
            cv2.rectangle(display, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), (0, 0, 0), -1)
            cv2.putText(display, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Floor coordinate (green)
            coord_str = f"({fp[0]:.2f}, {fp[1]:.2f})m"
            cv2.putText(display, coord_str, (px + 13, py + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 220, 60), 1, cv2.LINE_AA)

        # ── Bottom HUD bar ──────────────────────────────
        n = len(self.image_points)
        if   n <  self._MIN_REQUIRED:    bar_col = (0,   70, 220)   # red
        elif n <  self._MIN_RECOMMENDED: bar_col = (0,  155, 255)   # orange
        else:                            bar_col = (40, 190,  40)   # green

        status = (
            f"[{self.camera_id}]  Points: {n}"
            f" (min {self._MIN_REQUIRED}, rec. {self._MIN_RECOMMENDED})"
            f"  |  U=undo  V=verify  Q=done  ESC=quit"
        )
        cv2.rectangle(display, (0, h - 30), (w, h), (10, 10, 10), -1)
        cv2.putText(display, status, (8, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, bar_col, 1, cv2.LINE_AA)

        # ── Verify grid indicator ───────────────────────
        if self._show_verify:
            cv2.putText(display, "GRID ON", (w - 110, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 60), 2, cv2.LINE_AA)

        # ── Lens correction indicator ───────────────────
        if self.lens_corrector.is_calibrated:
            cv2.putText(display, "LENS ✓", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (60, 220, 60), 2, cv2.LINE_AA)

        # ── Scale to fit screen ──────────────────────────
        if self._disp_scale < 1.0:
            dw = max(1, int(w * self._disp_scale))
            dh = max(1, int(h * self._disp_scale))
            display = cv2.resize(display, (dw, dh), interpolation=cv2.INTER_AREA)

        return display

    def _draw_verification_grid(self, display: np.ndarray) -> None:
        """Project a 5 m floor grid onto the display using the temp homography inverse."""
        if self._temp_H_inv is None:
            return

        h, w = display.shape[:2]
        H_inv = self._temp_H_inv
        step  = 5.0

        def _proj(fx: float, fy: float) -> Optional[tuple[int, int]]:
            pt  = np.array([[[fx, fy]]], dtype=np.float64)
            res = cv2.perspectiveTransform(pt, H_inv)
            u, v = float(res[0, 0, 0]), float(res[0, 0, 1])
            if -300 < u < w + 300 and -300 < v < h + 300:
                return int(round(u)), int(round(v))
            return None

        grid_col  = (0, 200, 60)
        label_col = (0, 240, 100)

        # Vertical grid lines (constant X)
        x_vals = np.arange(0, self._floor_w + step, step)
        for x_f in x_vals:
            prev: Optional[tuple[int, int]] = None
            for y_frac in np.linspace(0, self._floor_h, 50):
                pt = _proj(float(x_f), float(y_frac))
                if pt and prev:
                    cv2.line(display, prev, pt, grid_col, 1, cv2.LINE_AA)
                prev = pt

        # Horizontal grid lines (constant Y)
        y_vals = np.arange(0, self._floor_h + step, step)
        for y_f in y_vals:
            prev = None
            for x_frac in np.linspace(0, self._floor_w, 70):
                pt = _proj(float(x_frac), float(y_f))
                if pt and prev:
                    cv2.line(display, prev, pt, grid_col, 1, cv2.LINE_AA)
                prev = pt

        # Intersection dot + label at grid nodes
        for x_f in x_vals:
            for y_f in y_vals:
                pt = _proj(float(x_f), float(y_f))
                if pt:
                    u, v = pt
                    if 0 <= u < w and 0 <= v < h:
                        cv2.circle(display, (u, v), 3, label_col, -1)
                        cv2.putText(display, f"{int(x_f)},{int(y_f)}",
                                    (u + 4, v - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, label_col, 1)

    def _compute_temp_homography(self) -> bool:
        """
        Quick RANSAC homography for the verification grid overlay.
        Returns True on success; sets self._temp_H_inv.
        """
        if len(self.image_points) < 4:
            self._temp_H_inv = None
            return False

        img = np.array(self.image_points, dtype=np.float32)
        flr = np.array(self.floor_points,  dtype=np.float32)

        H, _ = cv2.findHomography(img, flr, cv2.RANSAC, 5.0)
        if H is None:
            self._temp_H_inv = None
            return False

        try:
            self._temp_H_inv = np.linalg.inv(H)
            return True
        except np.linalg.LinAlgError:
            self._temp_H_inv = None
            return False

    # ──────────────────────────────────────────────────────
    #  Quality analysis
    # ──────────────────────────────────────────────────────

    def _quality_analysis(self) -> list[str]:
        """
        Perform geometric quality checks on the collected points.

        Checks:
          1. Convex hull coverage (< 25 % of image area → too clustered)
          2. Near / far balance  (all in one image half → unbalanced)
          3. Overlap zone coverage (< 3 pts in overlap → fusion will suffer)

        Returns
        -------
        list[str]  — warning messages (empty = all good).
        """
        warnings: list[str] = []
        n = len(self.image_points)
        if n < self._MIN_REQUIRED or self.current_frame is None:
            return warnings

        h, w = self.current_frame.shape[:2]
        img_area = h * w

        # ── 1. Convex-hull coverage ──────────────────────────────
        pts_cv = np.array(self.image_points, dtype=np.float32).reshape(-1, 1, 2)
        hull   = cv2.convexHull(pts_cv)
        hull_area = cv2.contourArea(hull)
        coverage  = hull_area / img_area * 100.0

        if coverage < 25.0:
            warnings.append(
                f"⚠  Points too clustered — hull covers only {coverage:.0f}% of image area. "
                f"Spread points to all four corners and image edges."
            )

        # ── 2. Near / far balance ────────────────────────────────
        top = sum(1 for ip in self.image_points if ip[1] < h * 0.5)
        bot = n - top
        if top == 0:
            warnings.append(
                "⚠  All points are in the bottom half (near field). "
                "Add points near the TOP of the image for better far-field accuracy."
            )
        elif bot == 0:
            warnings.append(
                "⚠  All points are in the top half (far field). "
                "Add points near the BOTTOM of the image for near-field accuracy."
            )

        # ── 3. Overlap zone coverage ─────────────────────────────
        if self._overlap_info:
            n_zone = self._count_overlap_points()
            if n_zone < 3:
                oi = self._overlap_info
                warnings.append(
                    f"⚠  Only {n_zone} point(s) inside overlap zone "
                    f"(need 3+ for reliable fusion with {oi['other_camera']})."
                )

        return warnings

    # ──────────────────────────────────────────────────────
    #  Persistence helpers
    # ──────────────────────────────────────────────────────

    def _save_to_config(
        self,
        img_pts: list,
        flr_pts: list,
        pts_are_undistorted: bool,
    ) -> str:
        """
        Write the calibration points (and undistortion flag) into cameras.json.

        Image points are stored as **normalized [0, 1] fractions** relative to
        the calibration frame size.  ``calibration_frame_size`` is also recorded
        so ``HomographyMapper`` can denormalize them correctly, and to support the
        legacy pixel-scaling path for configs that predate normalization.

        Returns the original JSON text so the caller can roll back on failure.
        """
        json_path = os.path.join(self.config_path, "cameras.json")
        with open(json_path) as f:
            config = json.load(f)
        original = json.dumps(config, indent=2)   # for rollback

        entry = next(
            (c for c in config["cameras"] if c["id"] == self.camera_id), None
        )
        if entry is None:
            raise ValueError(f"Camera '{self.camera_id}' not found in cameras.json")

        # Normalize image_points to [0, 1] fractions so the config is
        # resolution-independent.  HomographyMapper denormalizes them at load
        # time using calibration_frame_size + the live video size.
        fw, fh = float(self._frame_w), float(self._frame_h)
        normalized_img_pts = [
            [round(px / fw, 6), round(py / fh, 6)]
            for px, py in img_pts
        ]

        entry["calibration_points"]["image_points"]          = normalized_img_pts
        entry["calibration_points"]["floor_points"]          = flr_pts
        entry["calibration_points"]["points_are_undistorted"] = pts_are_undistorted
        entry["calibration_points"]["coordinate_format"]      = "normalized"
        # Reference frame size — lets HomographyMapper denormalize back to
        # calibration-space pixels before applying the pixel-scaling path.
        entry["calibration_points"]["calibration_frame_size"] = [
            self._frame_w, self._frame_h
        ]

        with open(json_path, "w") as f:
            json.dump(config, f, indent=2)
        return original

    def _restore_config(self, original_json: str) -> None:
        json_path = os.path.join(self.config_path, "cameras.json")
        with open(json_path, "w") as f:
            f.write(original_json)
        logger.info("[%s] cameras.json rolled back.", self.camera_id)

    def _delete_old_npz(self) -> None:
        """Remove stale homography .npz to force a fresh compute on next load."""
        npz = os.path.join(self.config_path, f"homography_{self.camera_id}.npz")
        if os.path.exists(npz):
            os.remove(npz)
            logger.info("[%s] Deleted stale homography .npz", self.camera_id)

    # ──────────────────────────────────────────────────────
    #  Error visualisation
    # ──────────────────────────────────────────────────────

    def _draw_error_visualization(
        self,
        display: np.ndarray,
        report: dict,
    ) -> np.ndarray:
        """
        Overlay per-point reprojection errors on *display*.

        Circle radius is proportional to error relative to the mean.
        Colour:  green ≤ mean   |   orange ≤ 3× mean   |   red > 3× mean.
        Points > 3× mean also get a prominent "!#N" flag.
        """
        errors  = report.get("per_point_errors_m", [])
        mean_e  = max(report.get("mean_error_m", 1e-9), 1e-9)
        out     = display.copy()

        for i, (ip, err) in enumerate(zip(self.image_points, errors)):
            px, py = int(round(ip[0])), int(round(ip[1]))
            radius = max(6, min(60, int(err / mean_e * 12)))

            if   err <= mean_e:       col = (40,  200, 40)
            elif err <= 3 * mean_e:   col = (0,   165, 255)
            else:                     col = (0,   40,  210)

            cv2.circle(out, (px, py), radius, col, 2)
            cv2.putText(out, f"{err * 100:.1f}cm",
                        (px + radius + 4, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

            if err > 3 * mean_e:
                cv2.putText(out, f"! #{i+1}",
                            (px - 22, py - radius - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 40, 210), 2)

        # Legend
        cv2.rectangle(out, (0, 0), (330, 36), (0, 0, 0), -1)
        cv2.putText(out,
                    "Error visualisation — green ≤ mean  orange ≤ 3×  red > 3×",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
        return out

    # ──────────────────────────────────────────────────────
    #  Finalisation
    # ──────────────────────────────────────────────────────

    def _finalize(self) -> str:
        """
        Save points, compute homography, show verification, prompt user.

        Returns
        -------
        'accept'  — saved and accepted
        'discard' — user chose to discard
        'redo'    — user wants to redo from scratch
        """
        n = len(self.image_points)

        # ── Warn about limited accuracy ─────────────────────────
        if self._MIN_REQUIRED <= n < self._MIN_RECOMMENDED:
            print(f"\n  ⚠  WARNING: {n} point pairs gives limited accuracy. "
                  f"{self._MIN_RECOMMENDED}+ recommended.")

        # ── Quality analysis warnings ────────────────────────────
        warnings = self._quality_analysis()
        if warnings:
            print("\n  ── Distribution Warnings " + "─" * 36)
            for w in warnings:
                print(f"  {w}")
            print()

        # ── Save to cameras.json ─────────────────────────────────
        pts_are_undistorted = self.lens_corrector.is_calibrated
        try:
            original_json = self._save_to_config(
                self.image_points,
                self.floor_points,
                pts_are_undistorted,
            )
        except Exception as exc:
            print(f"  ✗  Failed to save calibration points: {exc}")
            return "discard"

        self._delete_old_npz()
        print(f"  ✓  Saved {n} point pairs to cameras.json  "
              f"(undistorted={pts_are_undistorted})")

        # ── Compute high-accuracy homography ─────────────────────
        print("  Computing high-accuracy homography …\n")
        from calibration.homography import HomographyMapper
        mapper = HomographyMapper(self.camera_id, self.config_path)

        if not mapper.is_calibrated:
            print("  ✗  Homography computation failed.")
            print("     Check that your floor_points are correct (X=right, Y=up, metres).")
            self._restore_config(original_json)
            return "discard"

        # ── Verification visualisation ───────────────────────────
        report = mapper.get_reprojection_error()
        if report and self.current_frame is not None:
            vis = self.current_frame.copy()

            # Project mapper's verified grid (using the refined H_inv)
            if mapper.H_inv is not None:
                self._temp_H_inv = mapper.H_inv
                self._draw_verification_grid(vis)

            # Overlay per-point errors
            vis = self._draw_error_visualization(vis, report)

            win_v = f"Verification — {self.camera_id}  (press any key)"
            cv2.imshow(win_v, vis)
            cv2.waitKey(1)
            print("  Verification window open — check that grid lines align with the floor.")
            print("  Press any key in the verification window to continue …")
            cv2.waitKey(0)
            cv2.destroyWindow(win_v)

        # ── Accept / discard / redo prompt ───────────────────────
        print()
        while True:
            choice = input("  Accept calibration? [y]es / [n]o / [r]edo: ").strip().lower()
            if choice in ("y", "yes", ""):
                print(f"\n  ✓  Calibration for '{self.camera_id}' accepted and saved.\n")
                return "accept"
            elif choice in ("n", "no"):
                print("  ✗  Calibration discarded.")
                self._restore_config(original_json)
                self._delete_old_npz()
                return "discard"
            elif choice in ("r", "redo"):
                print("  ↺  Restarting calibration …\n")
                self._restore_config(original_json)
                self._delete_old_npz()
                return "redo"
            else:
                print("  Please enter  y, n,  or  r.")

    # ──────────────────────────────────────────────────────
    #  Main interactive loop
    # ──────────────────────────────────────────────────────

    def run(self) -> bool:
        """
        Launch the interactive calibration GUI.

        Returns True if calibration was accepted and saved successfully.
        """
        self._print_tips()
        self._print_overlap_warning()

        while True:  # outer loop handles 'redo'
            # ── Reset state ──────────────────────────────────────
            self.image_points = []
            self.floor_points = []
            self._show_verify  = False
            self._temp_H_inv   = None

            # ── Grab reference frame (image file or video/RTSP) ───────────
            raw_frame = self._read_source_frame()
            if raw_frame is None:
                return False

            # ── Optionally undistort frame ────────────────────────
            if self.lens_corrector.is_calibrated:
                self.current_frame = self.lens_corrector.undistort_frame(raw_frame)
                print("  ✓  Lens correction active — calibrating on undistorted image")
                print("     All clicked pixel coordinates are in the corrected image space.\n")
            else:
                self.current_frame = raw_frame
                print("  ⚠  No lens calibration — for better accuracy, run --intrinsic first\n")

            self._frame_h, self._frame_w = self.current_frame.shape[:2]

            # ── Create OpenCV window ──────────────────────────────
            #    MUST follow the pattern:
            #      namedWindow → imshow → waitKey(1) → setMouseCallback
            #    Calling setMouseCallback before imshow leaves the window
            #    handle NULL on some backends (Qt/GTK) and crashes.
            win = self._WINDOW_TITLE_TPL.format(cam_id=self.camera_id)

            # ── Compute display scale so the image fits the screen ────────────
            # Scale the image content to fit _MAX_WIN bounds; window is then
            # exactly that size (WINDOW_AUTOSIZE).  Qt5 opens this correctly
            # every time without any resizeWindow timing issues.
            scale = min(
                _MAX_WIN_W / max(self._frame_w, 1),
                _MAX_WIN_H / max(self._frame_h, 1),
                1.0,
            )
            self._disp_scale = scale

            # Shared mutable click state (list so closure can mutate it)
            _pending_click: list[Optional[tuple[int, int]]] = [None]

            def _mouse_cb(event: int, x: int, y: int, _flags, _param) -> None:
                if event == cv2.EVENT_LBUTTONDOWN:
                    # Unscale: convert display-pixel → original-frame-pixel
                    _pending_click[0] = (
                        int(round(x / self._disp_scale)),
                        int(round(y / self._disp_scale)),
                    )

            # WINDOW_AUTOSIZE: Qt5 opens the window at exact image size.
            # Pattern: namedWindow → imshow → waitKey(1) → setMouseCallback
            cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(win, self._redraw())   # draws correctly-scaled frame
            # Pump Qt event loop several times — 1 ms is not always enough
            # for Qt5 to fully register the native window handle.
            for _ in range(5):
                cv2.waitKey(20)
            cv2.setMouseCallback(win, _mouse_cb)

            print(f"  Window: '{win}'")
            print("  Left-click on a known floor point in the image window,")
            print("  then type its real-world X Y coordinates (metres) here.\n")

            # ── Inner event loop ──────────────────────────────────
            done   = False
            result = "discard"

            while not done:
                cv2.imshow(win, self._redraw())
                key = cv2.waitKey(30) & 0xFF

                # ── ESC: force quit without saving ────────────────
                if key == 27:
                    print("\n  ESC — exiting without saving.")
                    cv2.destroyWindow(win)
                    return False

                # ── Q: finish (attempt to finalise) ──────────────
                elif key in (ord("q"), ord("Q")):
                    n = len(self.image_points)
                    if n < self._MIN_REQUIRED:
                        print(
                            f"\n  ✗  Need at least {self._MIN_REQUIRED} points "
                            f"(have {n}).  ESC to quit without saving."
                        )
                    else:
                        cv2.destroyWindow(win)
                        done   = True
                        result = self._finalize()

                # ── U: undo last point ────────────────────────────
                elif key in (ord("u"), ord("U")):
                    if self.image_points:
                        self.image_points.pop()
                        self.floor_points.pop()
                        print(f"  ↩  Undo — {len(self.image_points)} point(s) remaining.")
                        if self._show_verify:
                            self._compute_temp_homography()
                    else:
                        print("  Nothing to undo.")

                # ── V: toggle verification grid ───────────────────
                elif key in (ord("v"), ord("V")):
                    if len(self.image_points) < 4:
                        print(
                            f"  V: need ≥4 points for grid "
                            f"(have {len(self.image_points)})."
                        )
                    else:
                        self._show_verify = not self._show_verify
                        if self._show_verify:
                            ok = self._compute_temp_homography()
                            if ok:
                                print("  V: Grid ON — green lines should align with the real floor.")
                            else:
                                print("  V: Could not compute temp homography.")
                                self._show_verify = False
                        else:
                            print("  V: Grid OFF.")

                # ── Mouse click: collect floor coordinate ─────────
                click = _pending_click[0]
                if click is not None:
                    _pending_click[0] = None
                    u, v    = click
                    n_next  = len(self.image_points) + 1

                    print(f"\n  Point {n_next}: pixel ({u}, {v})")

                    # Terminal prompt with retry
                    while True:
                        raw = input("  Enter floor X, Y in metres (e.g. 12.5  8.0): ").strip()
                        if not raw:
                            print("  Skipped (blank input).")
                            break
                        parts = raw.replace(",", " ").split()
                        if len(parts) >= 2:
                            try:
                                fx, fy = float(parts[0]), float(parts[1])
                            except ValueError:
                                print("  ✗  Cannot parse numbers — try again.")
                                continue

                            self.image_points.append([float(u), float(v)])
                            self.floor_points.append([fx, fy])
                            print(
                                f"  ✓  Pair #{n_next} added: "
                                f"pixel=({u}, {v})  →  floor=({fx:.3f}, {fy:.3f}) m"
                            )

                            if self._is_in_overlap_zone(fx, fy):
                                oi = self._overlap_info
                                print(
                                    f"     ✓  Inside overlap zone — "
                                    f"use same coords when calibrating {oi['other_camera']}"
                                )

                            if self._show_verify:
                                self._compute_temp_homography()
                            break
                        else:
                            print("  ✗  Expected two numbers — try again (e.g. 12.5  8.0).")

            # ── Handle result from _finalize() ───────────────────
            if result == "redo":
                print()
                continue            # restart outer while
            elif result == "accept":
                return True
            else:
                return False        # discard


# ═══════════════════════════════════════════════════════════════════════════
#  Module-level helpers
# ═══════════════════════════════════════════════════════════════════════════

def _point_in_polygon(px: float, py: float, polygon: list) -> bool:
    """
    Ray-casting point-in-polygon test.

    Parameters
    ----------
    px, py   : float — query point
    polygon  : list of [x, y] — closed polygon vertices

    Returns
    -------
    bool
    """
    n       = len(polygon)
    inside  = False
    j       = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ═══════════════════════════════════════════════════════════════════════════
#  Legacy function — kept for backward compatibility
# ═══════════════════════════════════════════════════════════════════════════

def run_floor_calibration(
    camera_id: str,
    cameras_config_path=None,   # unused — kept for API compat
    source: Optional[str] = None,
    config_path: str = "config/",
) -> bool:
    """
    Backward-compatible entry point.

    Reads the camera source from cameras.json (or uses *source* override)
    and delegates to :class:`CalibrationTool`.
    """
    json_path = os.path.join(config_path, "cameras.json")
    try:
        with open(json_path) as f:
            cfg = json.load(f)
        entry = next((c for c in cfg["cameras"] if c["id"] == camera_id), None)
        if entry is None:
            print(f"  ✗  Camera '{camera_id}' not found in cameras.json")
            return False
        cam_source = source if source else entry["source"]
    except Exception as exc:
        print(f"  ✗  Cannot read cameras.json: {exc}")
        return False

    tool = CalibrationTool(camera_id, cam_source, config_path)
    return tool.run()
