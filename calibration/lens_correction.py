"""
lens_correction.py — Camera Intrinsic Calibration & Lens Distortion Removal

WHY THIS MATTERS
----------------
CCTV camera lenses introduce radial and tangential distortion, especially near
image edges.  A person at the edge of the frame can be displaced by 10–30 px
from their true position.  Applying homography on distorted pixels produces
floor-coordinate errors of 0.5–2.0 m at the edges.

By removing lens distortion FIRST, then applying homography on undistorted
pixels, accuracy improves from ~0.5 m to ~0.05–0.1 m.

TYPICAL WORKFLOW
----------------
  1. python main.py --intrinsic cam_1                     # live RTSP, chessboard
  2. python main.py --intrinsic cam_1 --source video.mp4  # from video file
  3. python main.py --intrinsic cam_1 --method lines      # no chessboard needed

NOTES FOR CCTV CAMERAS
-----------------------
- Most CCTV cameras use varifocal lenses.  If the zoom/focus changes after
  calibration, you MUST recalibrate.
- For fixed-lens cameras (most factory CCTV), calibrate once and it is
  permanent.
- Print a 9×6 inner-corner chessboard on A3 paper, tape it to a rigid board.
  Hold it in front of the camera at various angles for 20+ captures.  The
  entire board must be visible in each frame.
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

# ---------------------------------------------------------------------------
# Display window helper
# ---------------------------------------------------------------------------
_MAX_WIN_W = 1400
_MAX_WIN_H = 900


def _win_size(frame_w: int, frame_h: int) -> tuple[int, int]:
    """
    Return (win_w, win_h) that fits the frame on the screen without upscaling.
    Proportional fit: both dimensions scaled by the same factor.
    """
    if frame_w <= 0 or frame_h <= 0:
        return _MAX_WIN_W, _MAX_WIN_H
    scale = min(_MAX_WIN_W / frame_w, _MAX_WIN_H / frame_h, 1.0)
    return max(1, int(frame_w * scale)), max(1, int(frame_h * scale))


# ---------------------------------------------------------------------------
# Image-file detection helper
# ---------------------------------------------------------------------------
_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def _is_image_source(source: str) -> bool:
    """Return True when *source* is a path to a still image file."""
    return Path(source).suffix.lower() in _IMAGE_EXTS


def _imread_source(source: str) -> Optional[np.ndarray]:
    """
    Load a still image from *source* and return a BGR ndarray.
    Prints an error and returns None on failure.
    """
    frame = cv2.imread(source)
    if frame is None:
        logger.error("Cannot read image file: %s", source)
        print(f"  ✗  Cannot read image file: {source}")
        return None
    print(f"  ✓  Loaded image: {source}  ({frame.shape[1]}×{frame.shape[0]})")
    return frame


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)
_DEFAULT_BOARD   = (9, 6)          # inner corners (w, h)
_DEFAULT_SQUARE  = 25.0            # mm
_DEFAULT_FRAMES  = 20              # minimum captures for chessboard calibration
_VIDEO_INTERVAL  = 30              # grab every N-th frame in video mode
_LINES_FRAMES    = 60              # frames to average for line-based calibration


# ===========================================================================
# LensCorrector
# ===========================================================================

class LensCorrector:
    """
    Handles camera intrinsic calibration and lens distortion removal.

    Uses Zhang's chessboard method (cv2.calibrateCamera) as the primary path,
    and a straight-line fitting method as a no-pattern fallback.

    Calibrated parameters are persisted in two places:
      • ``config/intrinsics_{camera_id}.npz``  (full precision, fast reload)
      • ``config/cameras.json``                (inline, for homography.py)

    Parameters
    ----------
    camera_id : str
    config_path : str
        Directory that contains cameras.json and where .npz files are saved.
    """

    def __init__(self, camera_id: str, config_path: str = "config/") -> None:
        self.camera_id    = camera_id
        self.config_path  = config_path

        self.camera_matrix:     Optional[np.ndarray] = None  # 3×3 (fx,fy,cx,cy)
        self.dist_coeffs:       Optional[np.ndarray] = None  # (k1,k2,p1,p2,k3)
        self.new_camera_matrix: Optional[np.ndarray] = None  # after undistortion
        self.roi:               Optional[tuple]      = None  # valid pixel region
        self.map1:              Optional[np.ndarray] = None  # remap X (CV_16SC2)
        self.map2:              Optional[np.ndarray] = None  # remap Y

        self._image_size:  Optional[tuple[int, int]] = None  # (w, h)
        self._rms:         float = 0.0                        # reprojection error

        self._load()

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """
        Load saved intrinsics from .npz (preferred) or cameras.json (fallback).
        Precomputes undistortion maps if intrinsics are found.
        """
        npz_path = self._npz_path()
        if os.path.exists(npz_path):
            try:
                data = np.load(npz_path)
                self.camera_matrix     = data["camera_matrix"]
                self.dist_coeffs       = data["dist_coeffs"]
                self.new_camera_matrix = data["new_camera_matrix"]
                self._image_size       = tuple(data["image_size"].tolist())
                self._rms              = float(data.get("rms", 0.0))
                self._compute_undistort_maps(self._image_size)
                logger.info(
                    "[%s] Intrinsics loaded from %s  (RMS=%.4f px)",
                    self.camera_id, npz_path, self._rms,
                )
                return
            except Exception as exc:
                logger.warning("[%s] Failed to load %s: %s", self.camera_id, npz_path, exc)

        # Fallback: read cameras.json inline block
        json_path = os.path.join(self.config_path, "cameras.json")
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    config = json.load(f)
                entry = next(
                    (c for c in config["cameras"] if c["id"] == self.camera_id), None
                )
                if entry and entry["intrinsics"].get("calibrated"):
                    cm = entry["intrinsics"].get("camera_matrix")
                    dc = entry["intrinsics"].get("dist_coeffs")
                    if cm and dc:
                        self.camera_matrix = np.array(cm, dtype=np.float64)
                        self.dist_coeffs   = np.array(dc, dtype=np.float64)
                        self._rms = float(entry["intrinsics"].get("rms_px", 0.0))
                        logger.info(
                            "[%s] Intrinsics loaded from cameras.json (no size → maps deferred)",
                            self.camera_id,
                        )
                        # Maps will be computed on first undistort_frame() call
            except Exception as exc:
                logger.warning("[%s] cameras.json read error: %s", self.camera_id, exc)

    def _save(self) -> None:
        """Persist intrinsics to .npz AND update cameras.json."""
        # 1. npz
        npz_path = self._npz_path()
        os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
        np.savez(
            npz_path,
            camera_matrix     = self.camera_matrix,
            dist_coeffs       = self.dist_coeffs,
            new_camera_matrix = self.new_camera_matrix,
            image_size        = np.array(self._image_size),
            rms               = np.array(self._rms),
        )
        logger.info("[%s] Intrinsics saved → %s", self.camera_id, npz_path)

        # 2. cameras.json inline
        json_path = os.path.join(self.config_path, "cameras.json")
        if not os.path.exists(json_path):
            return
        with open(json_path) as f:
            config = json.load(f)
        entry = next(
            (c for c in config["cameras"] if c["id"] == self.camera_id), None
        )
        if entry is not None:
            # Create the "intrinsics" sub-dict if this camera was added without one
            if "intrinsics" not in entry:
                entry["intrinsics"] = {}
            entry["intrinsics"]["camera_matrix"]     = self.camera_matrix.tolist()
            entry["intrinsics"]["dist_coeffs"]       = self.dist_coeffs.tolist()
            entry["intrinsics"]["new_camera_matrix"] = self.new_camera_matrix.tolist()
            entry["intrinsics"]["image_size"]        = list(self._image_size)
            entry["intrinsics"]["calibrated"]        = True
            entry["intrinsics"]["rms_px"]            = round(self._rms, 6)
            with open(json_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info("[%s] cameras.json updated.", self.camera_id)

    def _npz_path(self) -> str:
        return os.path.join(self.config_path, f"intrinsics_{self.camera_id}.npz")

    # ------------------------------------------------------------------
    # Undistort map computation
    # ------------------------------------------------------------------

    def _compute_undistort_maps(self, image_size: tuple[int, int]) -> None:
        """
        Precompute remap arrays for fast runtime undistortion.

        Uses alpha=1 so the entire original image is kept (black borders appear
        at the edges after undistortion).  Uses CV_16SC2 maps which are the
        fastest format for cv2.remap.

        Parameters
        ----------
        image_size : (width, height)
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            return

        self._image_size = image_size
        w, h = image_size

        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), alpha=1, newImgSize=(w, h)
        )

        # CV_16SC2 maps are 3× faster in remap() vs float32 maps
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            None,                      # no rectification (monocular)
            self.new_camera_matrix,
            (w, h),
            cv2.CV_16SC2,
        )
        logger.debug("[%s] Undistortion maps computed for %dx%d.", self.camera_id, w, h)

    # ------------------------------------------------------------------
    # Calibration — chessboard (Zhang's method)
    # ------------------------------------------------------------------

    def calibrate_from_chessboard(
        self,
        source: str,
        board_size: tuple[int, int] = _DEFAULT_BOARD,
        square_size_mm: float = _DEFAULT_SQUARE,
        num_frames: int = _DEFAULT_FRAMES,
    ) -> bool:
        """
        Calibrate camera intrinsics using a chessboard pattern.

        Procedure
        ---------
        1. Open the video source (RTSP URL or local file).
        2. Capture *num_frames* frames:
           - **Video file**: auto-capture every :data:`_VIDEO_INTERVAL` frames.
           - **Live feed** (RTSP / integer index): user presses ``SPACE``.
        3. For each frame:
           a. Convert to grayscale.
           b. ``cv2.findChessboardCorners`` with adaptive threshold + normalise.
           c. Refine to sub-pixel with ``cv2.cornerSubPix``.
           d. Accumulate (objpoints, imgpoints).
        4. ``cv2.calibrateCamera`` → RMS + intrinsics.
        5. Compute per-view reprojection error; warn if > 1.0 px.
        6. ``cv2.getOptimalNewCameraMatrix`` (alpha=1).
        7. ``cv2.initUndistortRectifyMap`` (CV_16SC2 for speed).
        8. Save to .npz and cameras.json.

        Parameters
        ----------
        source : str
            RTSP URL, local video file path, or ``"0"`` / ``"1"`` for webcam.
        board_size : (cols, rows)
            Number of *inner* corners along width and height.
        square_size_mm : float
            Physical side length of one square in millimetres.
        num_frames : int
            Minimum successful detections required before calibrating.

        Returns
        -------
        bool
            True if calibration succeeded with mean reprojection error < 1.0 px.
        """
        # ── Image file: not useful for chessboard (needs 20+ different views) ──
        if _is_image_source(source):
            print(f"\n  ✗  --method chessboard requires multiple frames from a video "
                  f"or live feed — a single image is not sufficient.")
            print(f"     Use --method lines instead:  "
                  f"python main.py --intrinsic {self.camera_id} "
                  f"--method lines --source {source}")
            return False

        is_live   = _is_live_source(source)
        source_cv = int(source) if source.isdigit() else source

        # Force RTSP-over-TCP for Hikvision / Dahua cameras (no-op for local files)
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        cap = (
            cv2.VideoCapture(source_cv)
            if isinstance(source_cv, int)
            else cv2.VideoCapture(source_cv, cv2.CAP_FFMPEG)
        )
        if not cap.isOpened():
            logger.error("[%s] Cannot open source: %s", self.camera_id, source)
            return False

        # Prepare 3-D object points for one board view
        square_m = square_size_mm / 1000.0
        objp = _build_object_points(board_size[0], board_size[1], square_m)

        obj_points: list[np.ndarray] = []
        img_points: list[np.ndarray] = []
        image_size: Optional[tuple[int, int]] = None

        window = f"Intrinsic Calibration — {self.camera_id}"

        print(f"\n{'='*60}")
        print(f"  Intrinsic Calibration — {self.camera_id}")
        print(f"{'='*60}")
        print(f"  Board   : {board_size[0]}×{board_size[1]} inner corners")
        print(f"  Square  : {square_size_mm:.0f} mm")
        print(f"  Target  : {num_frames} valid captures")
        if is_live:
            print("  Mode    : LIVE — press  SPACE  to capture,  q  to quit")
        else:
            print(f"  Mode    : VIDEO — auto-capture every {_VIDEO_INTERVAL} frames")
        print()

        frame_idx     = 0
        auto_armed    = True   # for video mode
        _disp_scale   = None   # computed on first frame
        _win_created  = False  # WINDOW_AUTOSIZE: create window on first frame

        try:
            while len(obj_points) < num_frames:
                ret, frame = cap.read()
                if not ret:
                    if not is_live:
                        print("  End of video reached before collecting enough frames.")
                        break
                    logger.warning("[%s] Frame read failed — retrying", self.camera_id)
                    continue

                frame_idx += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if image_size is None:
                    image_size = (gray.shape[1], gray.shape[0])

                display  = frame.copy()
                n_so_far = len(obj_points)

                # Decide whether to attempt detection this frame
                attempt = False
                if is_live:
                    # Always show live; detection triggered by SPACE below
                    pass
                else:
                    if auto_armed and (frame_idx % _VIDEO_INTERVAL == 0):
                        attempt = True

                found    = False
                corners2 = None

                if attempt or is_live:
                    found, corners = cv2.findChessboardCorners(
                        gray, board_size,
                        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
                    )
                    if found:
                        corners2 = cv2.cornerSubPix(
                            gray, corners, (11, 11), (-1, -1), _SUBPIX_CRITERIA
                        )

                # Overlay
                if found and corners2 is not None:
                    cv2.drawChessboardCorners(display, board_size, corners2, found)
                    status_col = (0, 255, 0)
                    status_txt = f"FOUND  ({n_so_far+1}/{num_frames})"
                else:
                    status_col = (0, 100, 255)
                    status_txt = f"Searching … ({n_so_far}/{num_frames})"

                cv2.putText(display, status_txt, (12, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_col, 2)
                if is_live:
                    cv2.putText(display, "SPACE=capture  q=quit", (12, display.shape[0] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                # Scale display for WINDOW_AUTOSIZE (no resizeWindow needed)
                if _disp_scale is None:
                    _disp_scale = min(
                        _MAX_WIN_W / max(display.shape[1], 1),
                        _MAX_WIN_H / max(display.shape[0], 1),
                        1.0,
                    )
                if _disp_scale < 1.0:
                    dw = max(1, int(display.shape[1] * _disp_scale))
                    dh = max(1, int(display.shape[0] * _disp_scale))
                    display = cv2.resize(display, (dw, dh), interpolation=cv2.INTER_AREA)
                if not _win_created:
                    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
                    _win_created = True
                cv2.imshow(window, display)
                key = cv2.waitKey(1 if not is_live else 30) & 0xFF

                if key == ord("q"):
                    print("  Quit by user.")
                    break

                # Live mode: SPACE triggers capture if board detected
                if is_live and key == ord(" "):
                    attempt = True
                    if not found:
                        # Re-run detection on current frame
                        found, corners = cv2.findChessboardCorners(
                            gray, board_size,
                            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
                        )
                        if found:
                            corners2 = cv2.cornerSubPix(
                                gray, corners, (11, 11), (-1, -1), _SUBPIX_CRITERIA
                            )

                # Accept valid detection
                if attempt and found and corners2 is not None:
                    obj_points.append(objp.copy())
                    img_points.append(corners2)
                    n_so_far = len(obj_points)
                    print(f"  ✓  Sample {n_so_far}/{num_frames} captured")
                elif attempt and not found:
                    print("  ✗  Chessboard not detected — adjust angle/lighting and retry")

        finally:
            cap.release()
            cv2.destroyWindow(window)

        return self._run_calibration(obj_points, img_points, image_size, source)

    # ------------------------------------------------------------------
    # Calibration — from image files
    # ------------------------------------------------------------------

    def calibrate_from_images(
        self,
        image_paths: list[str],
        board_size: tuple[int, int] = _DEFAULT_BOARD,
        square_size_mm: float = _DEFAULT_SQUARE,
    ) -> bool:
        """
        Calibrate from a list of pre-captured image file paths.

        Parameters
        ----------
        image_paths : list[str]
            Paths to PNG/JPG images showing the chessboard.
        board_size : (cols, rows)
        square_size_mm : float

        Returns
        -------
        bool
        """
        square_m = square_size_mm / 1000.0
        objp = _build_object_points(board_size[0], board_size[1], square_m)

        obj_points: list[np.ndarray] = []
        img_points: list[np.ndarray] = []
        image_size: Optional[tuple[int, int]] = None

        print(f"\n[Intrinsic / images] Camera: {self.camera_id} — {len(image_paths)} files")
        for i, path in enumerate(image_paths):
            img = cv2.imread(path)
            if img is None:
                logger.warning("Cannot read image: %s", path)
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if image_size is None:
                image_size = (gray.shape[1], gray.shape[0])

            found, corners = cv2.findChessboardCorners(
                gray, board_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if found:
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), _SUBPIX_CRITERIA
                )
                obj_points.append(objp.copy())
                img_points.append(corners2)
                print(f"  ✓  {Path(path).name}  ({len(obj_points)} valid so far)")
            else:
                print(f"  ✗  {Path(path).name}  (board not found)")

        return self._run_calibration(obj_points, img_points, image_size, "image_files")

    # ------------------------------------------------------------------
    # Shared calibration runner
    # ------------------------------------------------------------------

    def _run_calibration(
        self,
        obj_points: list[np.ndarray],
        img_points: list[np.ndarray],
        image_size: Optional[tuple[int, int]],
        source_label: str,
    ) -> bool:
        """
        Run cv2.calibrateCamera, compute per-view errors, build maps, save.

        Returns
        -------
        bool
            True if mean reprojection error < 1.0 px.
        """
        if len(obj_points) < 4:
            logger.error(
                "[%s] Only %d valid captures — need at least 4.  Calibration aborted.",
                self.camera_id, len(obj_points),
            )
            print(f"\n  ✗  Not enough valid captures ({len(obj_points)}).  Need ≥4.\n")
            return False

        if image_size is None:
            logger.error("[%s] image_size unknown.", self.camera_id)
            return False

        print(f"\n  Running cv2.calibrateCamera with {len(obj_points)} samples …")

        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, image_size, None, None
        )

        # ------------------------------------------------------------------
        # Per-view reprojection errors
        # ------------------------------------------------------------------
        per_view_errors = []
        for i, (objp_i, imgp_i, rvec, tvec) in enumerate(
            zip(obj_points, img_points, rvecs, tvecs)
        ):
            projected, _ = cv2.projectPoints(objp_i, rvec, tvec, camera_matrix, dist_coeffs)
            err = float(cv2.norm(imgp_i, projected, cv2.NORM_L2) / len(projected))
            per_view_errors.append(err)

        mean_err = float(np.mean(per_view_errors))
        max_err  = float(np.max(per_view_errors))
        worst_idx = int(np.argmax(per_view_errors))

        # ------------------------------------------------------------------
        # Print results
        # ------------------------------------------------------------------
        print(f"\n  ── Calibration Results ────────────────────────────────")
        print(f"  RMS reprojection error  : {rms:.4f} px")
        print(f"  Mean per-view error     : {mean_err:.4f} px")
        print(f"  Max per-view error      : {max_err:.4f} px  (view #{worst_idx+1})")
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        print(f"  Camera matrix           : fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")
        print(f"  Distortion coefficients : {dist_coeffs.flatten().tolist()}")
        print(f"  Image size              : {image_size[0]}×{image_size[1]} px")

        if rms < 0.5:
            quality = "EXCELLENT"
        elif rms < 1.0:
            quality = "ACCEPTABLE"
        else:
            quality = "POOR — consider recapturing with better coverage"
        print(f"  Quality assessment      : {quality}  (target < 0.5 px)")
        print(f"  ────────────────────────────────────────────────────────")

        # ------------------------------------------------------------------
        # Store and build maps
        # ------------------------------------------------------------------
        self.camera_matrix = camera_matrix
        self.dist_coeffs   = dist_coeffs
        self._rms          = rms
        self._compute_undistort_maps(image_size)

        # ------------------------------------------------------------------
        # Save
        # ------------------------------------------------------------------
        self._save()
        print(f"\n  ✓  Intrinsics saved to {self._npz_path()} and cameras.json\n")

        return rms < 1.0

    # ------------------------------------------------------------------
    # Calibration — straight-line fallback (no chessboard)
    # ------------------------------------------------------------------

    def calibrate_from_lines(self, source: str) -> bool:
        """
        Estimate radial lens distortion using straight environmental lines.

        USE CASE
        --------
        When a chessboard is unavailable, factory walls, floor markings, and
        conveyor belt edges serve as "known straight lines".  Barrel distortion
        makes these appear curved.  This method fits a single radial distortion
        coefficient k1 by minimising the curvature of detected Hough lines.

        ACCURACY
        --------
        Significantly less accurate than chessboard (≈2–4× worse RMS), but
        useful as a starting point for environments with long straight features.
        Only k1 is estimated; k2, p1, p2, k3 are set to 0.

        PROCEDURE
        ---------
        1. Collect :data:`_LINES_FRAMES` frames from *source*.
        2. For each frame, run ``cv2.HoughLinesP`` to detect long line segments.
        3. For each long segment, sample interior points (endpoints + midpoint).
        4. A straight line through undistorted points should have zero curvature.
           Curvature error for one triplet (A, B, C) is the perpendicular
           distance of B from the line A→C after undistortion.
        5. Use ``scipy.optimize.minimize_scalar`` to find k1 minimising total
           curvature error.
        6. Compute optimal camera matrix and remap maps.
        7. Save results.

        Parameters
        ----------
        source : str
            RTSP URL, video file path, or webcam index string.

        Returns
        -------
        bool
            True if at least 50 line triplets were found and optimisation converged.
        """
        try:
            from scipy.optimize import minimize_scalar
        except ImportError:
            logger.error("scipy is required for line-based calibration.  pip install scipy")
            return False

        print(f"\n[Intrinsic / lines] Camera: {self.camera_id}")

        triplets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        image_size: Optional[tuple[int, int]] = None
        collected = 0

        def _extract_triplets_from_frame(frame: np.ndarray) -> None:
            nonlocal image_size, collected
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if image_size is None:
                image_size = (gray.shape[1], gray.shape[0])
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=80,
                minLineLength=max(image_size[0], image_size[1]) // 4,
                maxLineGap=20,
            )
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0].astype(float)
                    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    triplets.append((
                        np.array([x1, y1]),
                        np.array([mx, my]),
                        np.array([x2, y2]),
                    ))
            collected += 1

        # ── Still image source ────────────────────────────────────────────────
        if _is_image_source(source):
            frame = _imread_source(source)
            if frame is None:
                return False
            print(f"  Processing image for line detection (single frame) …")
            _extract_triplets_from_frame(frame)

        # ── Video / RTSP source ───────────────────────────────────────────────
        else:
            source_cv = int(source) if source.isdigit() else source
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
            cap = (
                cv2.VideoCapture(source_cv)
                if isinstance(source_cv, int)
                else cv2.VideoCapture(source_cv, cv2.CAP_FFMPEG)
            )
            if not cap.isOpened():
                logger.error("[%s] Cannot open source: %s", self.camera_id, source)
                return False

            print(f"  Collecting {_LINES_FRAMES} frames for line detection …")
            try:
                while collected < _LINES_FRAMES:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if collected % 5 != 0:   # sample every 5th frame
                        collected += 1
                        continue
                    _extract_triplets_from_frame(frame)
            finally:
                cap.release()

        if len(triplets) < 50:
            print(f"  ✗  Only {len(triplets)} line triplets found (need ≥50).")
            print("     Try in a scene with more straight lines (walls, floor markings).")
            return False

        print(f"  Found {len(triplets)} line triplets from {collected} frames.")
        print("  Optimising k1 distortion coefficient …")

        if image_size is None:
            return False

        w, h = image_size
        cx, cy = w / 2.0, h / 2.0
        # Rough focal length estimate from image width
        f_est = float(max(w, h) * 0.8)

        # Build approximate camera matrix for undistortPoints
        K_approx = np.array([
            [f_est, 0,    cx],
            [0,    f_est, cy],
            [0,    0,    1.0],
        ], dtype=np.float64)

        def curvature_cost(k1: float) -> float:
            """Sum of midpoint deviations after undistortion with coefficient k1."""
            dist = np.array([[k1, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
            total_error = 0.0
            for A_raw, B_raw, C_raw in triplets:
                pts = np.array([[A_raw], [B_raw], [C_raw]], dtype=np.float64)
                undist = cv2.undistortPoints(pts, K_approx, dist, P=K_approx)
                A = undist[0, 0]
                B = undist[1, 0]
                C = undist[2, 0]
                # Perpendicular distance of B from line A→C
                AC = C - A
                AC_len = np.linalg.norm(AC)
                if AC_len < 1e-6:
                    continue
                AB = B - A
                cross = abs(AB[0] * AC[1] - AB[1] * AC[0])
                total_error += cross / AC_len
            return total_error

        result = minimize_scalar(
            curvature_cost,
            bounds=(-0.8, 0.8),
            method="bounded",
            options={"xatol": 1e-5, "maxiter": 500},
        )

        if not result.success and not np.isfinite(result.fun):
            print("  ✗  Optimisation did not converge.")
            return False

        k1_opt = float(result.x)
        print(f"  Optimised k1 = {k1_opt:.6f}  (cost={result.fun:.2f} px)")

        if abs(k1_opt) < 1e-5:
            print("  ⚠  k1 ≈ 0 — very little distortion detected.  Result may be unreliable.")

        # Build final matrices
        self.camera_matrix = K_approx.copy()
        self.dist_coeffs   = np.array([[k1_opt, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        self._rms          = float(result.fun / max(len(triplets), 1))
        self._compute_undistort_maps(image_size)
        self._save()

        print(f"\n  ✓  Line-based intrinsics saved  (k1={k1_opt:.6f})")
        print("  ⚠  Accuracy is lower than chessboard.  Use for rough correction only.\n")
        return True

    # ------------------------------------------------------------------
    # Calibration — circle / dot grid  (printed alternative to chessboard)
    # ------------------------------------------------------------------

    def calibrate_from_circle_grid(
        self,
        source: str,
        grid_size: tuple[int, int] = (4, 11),
        grid_spacing_mm: float = 25.0,
        num_frames: int = _DEFAULT_FRAMES,
        symmetric: bool = True,
    ) -> bool:
        """
        Calibrate using a printed circle/dot grid — the easiest chessboard
        alternative.

        WHY CIRCLES INSTEAD OF CHESSBOARD
        -----------------------------------
        A circle grid is simpler to make: print a regular grid of dots, or
        stick round stickers on a piece of cardboard.  OpenCV's
        ``findCirclesGrid`` locates the centres of filled circles with
        sub-pixel accuracy comparable to ``findChessboardCorners``.

        HOW TO CREATE THE PATTERN
        --------------------------
        Option A — print: search "OpenCV circle grid generator" or use the
        script printed at the end of this docstring.
        Option B — stickers: place identical round stickers (any size) in a
        regular W×H grid on a rigid flat surface at uniform spacing.
        Option C — floor markers: tape circular markers on the factory floor
        and photograph from the side at multiple angles (not overhead).

        SYMMETRIC vs ASYMMETRIC
        -----------------------
        ``symmetric=True``  — standard rectangular grid, e.g. 4×11 dots.
        ``symmetric=False`` — rows are offset (like asymmetric circle grid),
                              gives better calibration coverage; default
                              spacing applies differently.

        PROCEDURE
        ---------
        Same as ``calibrate_from_chessboard``:
        1. Open source (RTSP live or video file).
        2. Collect *num_frames* successful detections.
        3. ``cv2.calibrateCamera`` → intrinsics + maps → save.

        GENERATE A PATTERN (run once)
        ------------------------------
        .. code-block:: python

            import cv2
            img = cv2.imencode('.png',
                cv2.drawChessboardCorners(
                    np.ones((700, 500, 3), dtype=np.uint8) * 255,
                    (4, 11),
                    np.zeros((44, 1, 2), np.float32),  # placeholder
                    True))[1]

        Or simply search for "OpenCV asymmetric circle grid PDF" — many free
        printable versions are available online.

        Parameters
        ----------
        source : str
            RTSP URL, video file, or webcam index string.
        grid_size : (cols, rows)
            Number of circles across width and height.
        grid_spacing_mm : float
            Centre-to-centre distance between adjacent circles in mm.
        num_frames : int
            Minimum successful detections before calibrating.
        symmetric : bool
            True = symmetric grid, False = asymmetric (staggered rows).

        Returns
        -------
        bool
            True if calibration succeeded with RMS < 1.0 px.
        """
        # ── Image file: not useful for circle grid (needs 20+ different views) ─
        if _is_image_source(source):
            print(f"\n  ✗  --method circles requires multiple frames from a video "
                  f"or live feed — a single image is not sufficient.")
            print(f"     Use --method lines instead:  "
                  f"python main.py --intrinsic {self.camera_id} "
                  f"--method lines --source {source}")
            return False

        is_live   = _is_live_source(source)
        source_cv = int(source) if source.isdigit() else source
        flags     = cv2.CALIB_CB_SYMMETRIC_GRID if symmetric else cv2.CALIB_CB_ASYMMETRIC_GRID

        # Force RTSP-over-TCP for Hikvision / Dahua cameras (no-op for local files)
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        cap = (
            cv2.VideoCapture(source_cv)
            if isinstance(source_cv, int)
            else cv2.VideoCapture(source_cv, cv2.CAP_FFMPEG)
        )
        if not cap.isOpened():
            logger.error("[%s] Cannot open source: %s", self.camera_id, source)
            return False

        spacing_m = grid_spacing_mm / 1000.0
        cols, rows = grid_size
        n_pts = cols * rows

        # Build object points for one view
        objp = np.zeros((n_pts, 3), dtype=np.float32)
        if symmetric:
            objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * spacing_m
        else:
            # Asymmetric: odd rows are shifted by half a spacing
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    x = (2 * c + r % 2) * spacing_m
                    y = r * spacing_m
                    objp[idx] = [x, y, 0]
                    idx += 1

        obj_points: list[np.ndarray] = []
        img_points: list[np.ndarray] = []
        image_size: Optional[tuple[int, int]] = None

        grid_label = "symmetric" if symmetric else "asymmetric"
        window = f"Circle Grid Calibration — {self.camera_id}"

        print(f"\n{'='*60}")
        print(f"  Circle Grid Calibration — {self.camera_id}")
        print(f"{'='*60}")
        print(f"  Grid    : {cols}×{rows} circles ({grid_label})")
        print(f"  Spacing : {grid_spacing_mm:.0f} mm centre-to-centre")
        print(f"  Target  : {num_frames} valid captures")
        if is_live:
            print("  Mode    : LIVE — press  SPACE  to capture,  q  to quit")
        else:
            print(f"  Mode    : VIDEO — auto-capture every {_VIDEO_INTERVAL} frames")
        print()
        print("  TIP: tilt/rotate the pattern to different angles between captures")
        print("       for best coverage of the camera's field of view.\n")

        frame_idx    = 0
        _disp_scale  = None    # computed on first frame
        _win_created = False   # WINDOW_AUTOSIZE: create window on first frame
        try:
            while len(obj_points) < num_frames:
                ret, frame = cap.read()
                if not ret:
                    if not is_live:
                        print("  End of video reached.")
                        break
                    continue

                frame_idx += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if image_size is None:
                    image_size = (gray.shape[1], gray.shape[0])

                display  = frame.copy()
                n_so_far = len(obj_points)

                # Attempt detection
                attempt = (not is_live and frame_idx % _VIDEO_INTERVAL == 0)
                found, centres = False, None

                if attempt or is_live:
                    found, centres = cv2.findCirclesGrid(gray, grid_size, flags=flags)

                if found and centres is not None:
                    cv2.drawChessboardCorners(display, grid_size, centres, found)
                    status_txt = f"FOUND  ({n_so_far+1}/{num_frames})"
                    status_col = (0, 255, 0)
                else:
                    status_txt = f"Searching … ({n_so_far}/{num_frames})"
                    status_col = (0, 120, 255)

                cv2.putText(display, status_txt, (12, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_col, 2)
                if is_live:
                    cv2.putText(display, "SPACE=capture  q=quit",
                                (12, display.shape[0] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                # Scale display for WINDOW_AUTOSIZE
                if _disp_scale is None:
                    _disp_scale = min(
                        _MAX_WIN_W / max(display.shape[1], 1),
                        _MAX_WIN_H / max(display.shape[0], 1),
                        1.0,
                    )
                if _disp_scale < 1.0:
                    dw = max(1, int(display.shape[1] * _disp_scale))
                    dh = max(1, int(display.shape[0] * _disp_scale))
                    display = cv2.resize(display, (dw, dh), interpolation=cv2.INTER_AREA)
                if not _win_created:
                    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
                    _win_created = True
                cv2.imshow(window, display)
                key = cv2.waitKey(1 if not is_live else 30) & 0xFF

                if key == ord("q"):
                    print("  Quit by user.")
                    break

                if is_live and key == ord(" "):
                    attempt = True
                    if not found:
                        found, centres = cv2.findCirclesGrid(gray, grid_size, flags=flags)

                if attempt and found and centres is not None:
                    obj_points.append(objp.copy())
                    img_points.append(centres)
                    print(f"  ✓  Sample {len(obj_points)}/{num_frames} captured")
                elif attempt and not found:
                    print("  ✗  Grid not detected — check lighting and that all circles are visible")

        finally:
            cap.release()
            cv2.destroyWindow(window)

        return self._run_calibration(obj_points, img_points, image_size, source)

    # ------------------------------------------------------------------
    # Calibration — existing floor / wall marking points (no printing)
    # ------------------------------------------------------------------

    def calibrate_from_floor_markers(
        self,
        source: str,
        grid_cols: int = 4,
        grid_rows: int = 3,
        grid_spacing_mm: float = 500.0,
        num_views: int = 5,
    ) -> bool:
        """
        Calibrate using existing factory floor or wall markings — no printed
        pattern required.

        WHEN TO USE THIS
        ----------------
        Use when you cannot print or hold a calibration board in front of the
        camera, but can identify a set of **regularly-spaced physical marks**
        already present in the scene:

          • Bolt-hole grid on a machine bed
          • Painted crosses or dots on the factory floor (e.g. 500 mm grid)
          • Tile-corner intersections on a tiled floor
          • Screws / rivets on a conveyor frame at known spacing
          • Tape crosses placed temporarily on a flat surface

        REQUIREMENTS
        ------------
        • The marks must lie on a **single flat plane** (floor or wall).
        • They must form a **regular rectangular grid** with uniform spacing.
        • You must know (or measure) the spacing between them.
        • For full calibration: you need **multiple views** from different
          angles/distances.  Have a helper tilt a rigid board with the markers
          in front of the camera, or physically move the camera slightly
          between sessions.
        • Minimum viable: 1 view → only k1 distortion estimated (rough).
          5+ views from different angles → full k1,k2,p1,p2 calibration.

        PROCEDURE
        ---------
        For each view you will see a live frame.  Click the grid markers
        **row by row, left to right** starting from the top-left corner.
        Press ``d`` when all markers in the current view are clicked.
        Move the camera / pattern to a new angle and press ``n`` for the
        next view, or ``c`` to compute calibration immediately.

        Controls
        --------
        Left-click   — mark a point
        d            — done with this view (accept current clicks)
        u            — undo last click
        r            — reset clicks for current view
        n            — next view (skip this one)
        c            — compute calibration now (with views collected so far)
        q            — quit without calibrating

        Parameters
        ----------
        source : str
            RTSP URL, video file path, or webcam index.
        grid_cols, grid_rows : int
            Number of markers across width and height of the grid.
        grid_spacing_mm : float
            Spacing between adjacent markers in millimetres.
        num_views : int
            Target number of views to collect (more → more accurate).

        Returns
        -------
        bool
            True if calibration succeeded.
        """
        # ── Load the first frame (image file OR video/RTSP) ──────────────────
        cap = None
        if _is_image_source(source):
            first_frame = _imread_source(source)
            if first_frame is None:
                return False
        else:
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
            source_cv = int(source) if source.isdigit() else source
            cap = (
                cv2.VideoCapture(source_cv)
                if isinstance(source_cv, int)
                else cv2.VideoCapture(source_cv, cv2.CAP_FFMPEG)
            )
            if not cap.isOpened():
                print(f"  ERROR: Cannot open source: {source}")
                logger.error("[%s] Cannot open source: %s", self.camera_id, source)
                return False
            ret, first_frame = cap.read()
            if not ret or first_frame is None:
                print(f"  ERROR: Cannot read frame from: {source}")
                logger.error("[%s] Cannot read frame from: %s", self.camera_id, source)
                cap.release()
                return False

        image_size: tuple[int, int] = (first_frame.shape[1], first_frame.shape[0])

        spacing_m = grid_spacing_mm / 1000.0
        n_pts = grid_cols * grid_rows

        # Build the ideal 3-D object points for one view (flat plane, Z=0)
        objp = np.zeros((n_pts, 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0:grid_cols, 0:grid_rows].T.reshape(-1, 2) * spacing_m

        obj_points: list[np.ndarray] = []
        img_points: list[np.ndarray] = []

        print(f"\n{'='*60}")
        print(f"  Floor Marker Calibration - {self.camera_id}")
        print(f"{'='*60}")
        print(f"  Grid          : {grid_cols}×{grid_rows} markers")
        print(f"  Spacing       : {grid_spacing_mm:.0f} mm")
        print(f"  Points/view   : {n_pts}")
        print(f"  Target views  : {num_views}")
        print()
        print("  CLICK ORDER: row by row, left→right, top→bottom")
        _print_marker_grid_diagram(grid_cols, grid_rows)
        print()
        print("  Controls:")
        print("    Left-click → mark point  |  u → undo  |  r → reset view")
        print("    d → accept view          |  n → skip view")
        print("    c → compute now          |  q → quit\n")

        view_num   = 0
        clicks: list[tuple[int, int]] = []
        last_frame = first_frame.copy()   # seed so the loop never stalls on None

        # ── Window creation (WINDOW_AUTOSIZE — most reliable on Qt5/GTK) ───────
        # Scale the image to fit the screen so Qt opens it at the correct size
        # immediately, without any resizeWindow timing issues.
        window = f"Floor Marker Calibration - {self.camera_id}"
        _fh, _fw = first_frame.shape[:2]
        _disp_scale = min(_MAX_WIN_W / max(_fw, 1), _MAX_WIN_H / max(_fh, 1), 1.0)

        def _scale_display(img: np.ndarray) -> np.ndarray:
            if _disp_scale >= 1.0:
                return img
            dw = max(1, int(img.shape[1] * _disp_scale))
            dh = max(1, int(img.shape[0] * _disp_scale))
            return cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)

        # Mouse callback — unscale display coordinates → original frame coords
        _click_state: dict = {"pending": None}

        def _on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                _click_state["pending"] = (
                    int(round(x / _disp_scale)),
                    int(round(y / _disp_scale)),
                )

        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window, _scale_display(first_frame))
        # Pump Qt5 event loop until the native window handle is registered
        for _ in range(5):
            cv2.waitKey(20)
        cv2.setMouseCallback(window, _on_mouse)

        accepted_views = 0
        try:
            while accepted_views < num_views:
                if cap is not None:
                    ret, frame = cap.read()
                    if not ret:
                        # For video files: keep showing last frame; for live: retry
                        frame = last_frame.copy()
                    else:
                        last_frame = frame.copy()
                else:
                    # Image source — always show the same static frame
                    frame = last_frame.copy()

                # Consume pending click
                if _click_state["pending"] is not None and len(clicks) < n_pts:
                    clicks.append(_click_state["pending"])
                    _click_state["pending"] = None

                # Build display
                display = frame.copy()
                self._draw_marker_overlay(display, clicks, grid_cols, grid_rows)

                # Status bar
                progress = f"View {accepted_views+1}/{num_views}  |  Clicks: {len(clicks)}/{n_pts}"
                cv2.putText(display, progress, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

                hint = "d=accept  u=undo  r=reset  n=next  c=compute  q=quit"
                cv2.putText(display, hint, (10, display.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                # Show the ideal grid order overlay guide
                if len(clicks) < n_pts:
                    next_num = len(clicks) + 1
                    next_row = (next_num - 1) // grid_cols + 1
                    next_col = (next_num - 1) %  grid_cols + 1
                    guide = f"Next: point #{next_num}  (row {next_row}, col {next_col})"
                    cv2.putText(display, guide, (10, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)

                cv2.imshow(window, _scale_display(display))
                key = cv2.waitKey(30) & 0xFF

                if key == ord("q"):
                    print("  Quit by user.")
                    break

                elif key == ord("u"):  # undo
                    if clicks:
                        clicks.pop()
                        print(f"  Undo. {len(clicks)}/{n_pts} clicks remain.")

                elif key == ord("r"):  # reset
                    clicks.clear()
                    print("  View reset.")

                elif key == ord("n"):  # skip view
                    clicks.clear()
                    print(f"  View {accepted_views+1} skipped.")

                elif key == ord("c"):  # compute now
                    if obj_points:
                        print(f"\n  Computing calibration from {len(obj_points)} view(s) …")
                        break
                    else:
                        print("  No complete views yet — accept at least one view first.")

                elif key == ord("d"):  # accept view
                    if len(clicks) != n_pts:
                        print(f"  Need exactly {n_pts} clicks (have {len(clicks)}).  "
                              f"Click remaining points or press r to reset.")
                    else:
                        obj_points.append(objp.copy())
                        img_points.append(
                            np.array(clicks, dtype=np.float32).reshape(-1, 1, 2)
                        )
                        accepted_views += 1
                        print(f"  ✓  View {accepted_views}/{num_views} accepted  "
                              f"({n_pts} points)")
                        clicks = []
                        if accepted_views < num_views:
                            print(f"  Reposition the camera/pattern, then click {n_pts} "
                                  f"markers for view {accepted_views+1}.")

        finally:
            if cap is not None:
                cap.release()
            cv2.destroyWindow(window)

        n_views_collected = len(obj_points)
        if n_views_collected == 0:
            print("  ✗  No complete views collected.  Calibration aborted.")
            return False

        if n_views_collected == 1:
            # Single-view: run optimisation-based k1 estimate
            print(f"\n  Only 1 view collected — using single-view k1 optimisation …")
            print("  ⚠  For better accuracy, collect 5+ views from different angles.")
            return self._single_view_marker_calibration(
                img_points[0].reshape(-1, 2), objp, image_size
            )

        # Full calibration from multiple views
        print(f"\n  Running full calibration from {n_views_collected} views …")
        return self._run_calibration(obj_points, img_points, image_size, source)

    # ------------------------------------------------------------------
    # Single-view k1 estimation (fallback for floor marker method)
    # ------------------------------------------------------------------

    def _single_view_marker_calibration(
        self,
        img_pts: np.ndarray,
        obj_pts: np.ndarray,
        image_size: tuple[int, int],
    ) -> bool:
        """
        When only one view is available, estimate k1 by finding the value
        that best maps clicked image points to the known flat-grid pattern
        via a homography.

        Algorithm
        ---------
        1. Compute a rough homography from raw image points → grid XY.
        2. For candidate k1 values, undistort image points and recompute H.
        3. Minimise the sum of reprojection residuals.
        4. Report k1 and save results.
        """
        try:
            from scipy.optimize import minimize_scalar
        except ImportError:
            logger.error("scipy required.  pip install scipy")
            return False

        floor_pts_2d = obj_pts[:, :2].astype(np.float64)
        w, h         = image_size
        cx, cy       = w / 2.0, h / 2.0
        f_est        = float(max(w, h) * 0.85)

        K = np.array([[f_est, 0, cx],
                      [0, f_est, cy],
                      [0,  0,   1.0]], dtype=np.float64)

        # Scale floor coords to image-pixel scale for numerics
        floor_scale = float(max(w, h))
        floor_norm  = floor_pts_2d / floor_scale

        def reprojection_cost(k1: float) -> float:
            dist = np.array([[k1, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
            pts  = img_pts.astype(np.float64).reshape(-1, 1, 2)
            undist = cv2.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)

            H, mask = cv2.findHomography(
                undist.astype(np.float32),
                (floor_norm * floor_scale).astype(np.float32),
                cv2.RANSAC, 8.0,
            )
            if H is None:
                return 1e9

            # Project undistorted pts through H and compare to floor grid
            ones  = np.ones((len(undist), 1), dtype=np.float64)
            hpts  = np.hstack([undist, ones])  # (N, 3)
            proj  = (H @ hpts.T).T             # (N, 3)
            proj  = proj[:, :2] / proj[:, 2:3]

            target = (floor_norm * floor_scale).astype(np.float64)
            err    = np.linalg.norm(proj - target, axis=1)
            return float(err.mean())

        print("  Optimising k1 …")
        result = minimize_scalar(
            reprojection_cost,
            bounds=(-0.8, 0.8),
            method="bounded",
            options={"xatol": 1e-6, "maxiter": 500},
        )

        k1_opt = float(result.x)
        cost   = float(result.fun)
        print(f"  k1 = {k1_opt:.6f}   mean reprojection residual = {cost:.2f} px")

        if cost > 20:
            print("  ⚠  High residual — check that clicks are in correct order "
                  "(row by row, left to right).")

        self.camera_matrix = K.copy()
        self.dist_coeffs   = np.array([[k1_opt, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        self._rms          = cost
        self._compute_undistort_maps(image_size)
        self._save()

        print(f"\n  ✓  Single-view marker calibration saved  (k1={k1_opt:.6f})")
        print("  ⚠  Only k1 estimated.  Collect 5+ views for full accuracy.\n")
        return True

    # ------------------------------------------------------------------
    # Marker overlay helper
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_marker_overlay(
        frame: np.ndarray,
        clicks: list[tuple[int, int]],
        grid_cols: int,
        grid_rows: int,
    ) -> None:
        """
        Draw accepted click dots with index labels, and a faint expected-grid
        wireframe so the user can see if clicks are drifting off-pattern.
        """
        n_pts = grid_cols * grid_rows

        for i, (cx, cy) in enumerate(clicks):
            row = i // grid_cols
            col = i %  grid_cols
            colour = (0, 255, 0) if i < n_pts - 1 else (0, 200, 255)
            cv2.circle(frame, (cx, cy), 7, colour, -1)
            cv2.circle(frame, (cx, cy), 9, (255, 255, 255), 1)
            cv2.putText(frame, f"{i+1}",
                        (cx + 10, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            # Draw connecting lines along rows
            if col > 0:
                prev = clicks[i - 1]
                cv2.line(frame, prev, (cx, cy), (0, 200, 100), 1)
            # Draw connecting lines along columns (link to point in previous row)
            if row > 0:
                prev_row_idx = (row - 1) * grid_cols + col
                if prev_row_idx < len(clicks):
                    prev_r = clicks[prev_row_idx]
                    cv2.line(frame, prev_r, (cx, cy), (0, 150, 200), 1)

    # ------------------------------------------------------------------
    # Runtime undistortion (hot path)
    # ------------------------------------------------------------------

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Remove lens distortion from a frame.

        FAST — uses cv2.remap() with precomputed CV_16SC2 maps (~3× faster
        than cv2.undistort which recomputes the maps each call).

        Must be called BEFORE any homography mapping.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (OpenCV format).

        Returns
        -------
        np.ndarray
            Undistorted frame (same size as input).
            Returns *frame* unchanged if no intrinsic calibration is loaded.
        """
        if self.map1 is None or self.map2 is None:
            # Deferred map build: happens when loaded from cameras.json without size
            if self.camera_matrix is not None and self.dist_coeffs is not None:
                h, w = frame.shape[:2]
                if self._image_size is None or self._image_size != (w, h):
                    self._compute_undistort_maps((w, h))
            if self.map1 is None:
                return frame   # passthrough — not calibrated

        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

    def undistort_point(self, u: float, v: float) -> tuple[float, float]:
        """
        Undistort a single pixel coordinate.

        Used for undistorting individual detection foot-points without
        remapping an entire frame.  Useful when only sparse points need
        correction (e.g. projected foot positions in the detector).

        Parameters
        ----------
        u, v : float
            Distorted pixel coordinates.

        Returns
        -------
        (u', v') : tuple[float, float]
            Undistorted pixel coordinates.
        """
        if not self.is_calibrated:
            return (u, v)

        P = self.new_camera_matrix if self.new_camera_matrix is not None else self.camera_matrix
        pt = np.array([[[u, v]]], dtype=np.float64)
        result = cv2.undistortPoints(
            pt, self.camera_matrix, self.dist_coeffs, P=P
        )
        return (float(result[0, 0, 0]), float(result[0, 0, 1]))

    def undistort_points_batch(self, points: np.ndarray) -> np.ndarray:
        """
        Undistort multiple pixel coordinates in one vectorised call.

        Parameters
        ----------
        points : np.ndarray, shape (N, 2), dtype float32 or float64
            Each row is (u, v).

        Returns
        -------
        np.ndarray, shape (N, 2)
            Undistorted pixel coordinates.
            Returns *points* unchanged if not calibrated.
        """
        if not self.is_calibrated:
            return points

        P = self.new_camera_matrix if self.new_camera_matrix is not None else self.camera_matrix
        pts = points.astype(np.float64).reshape(-1, 1, 2)
        result = cv2.undistortPoints(
            pts, self.camera_matrix, self.dist_coeffs, P=P
        )
        return result.reshape(-1, 2)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_reprojection_error(self) -> float:
        """
        Return the mean reprojection error in pixels from the last calibration.

        Returns
        -------
        float
            RMS reprojection error (px).  0.0 if not calibrated.
        """
        return self._rms

    @property
    def is_calibrated(self) -> bool:
        """True if camera_matrix and dist_coeffs are loaded."""
        return self.camera_matrix is not None and self.dist_coeffs is not None

    def show_undistortion_comparison(self, frame: np.ndarray) -> None:
        """
        Display a side-by-side comparison of the original distorted frame and
        the undistorted result.

        Shows in an OpenCV window; press any key to close.  Also draws a grid
        of reference lines on both panels so distortion curvature is visible.

        Parameters
        ----------
        frame : np.ndarray
            A representative BGR frame from the camera.
        """
        composite = self._render_comparison(frame)

        win_cmp = f"Undistortion Comparison — {self.camera_id}"
        cv2.namedWindow(win_cmp, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(win_cmp, composite)
        cv2.waitKey(1)   # flush Qt event queue so window appears
        print("  Press any key to close the comparison window …")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_undistortion_comparison(self, frame: np.ndarray, output_path: Optional[str] = None) -> str:
        """
        Render a side-by-side comparison and save it to a file.

        Parameters
        ----------
        frame : np.ndarray
        output_path : str, optional
            Where to save.  Defaults to output/intrinsic_check_{camera_id}.jpg

        Returns
        -------
        str
            The actual save path.
        """
        if output_path is None:
            # Try to find 'output/' relative to current working dir, or use config_path/../output
            out_dir = Path("output")
            if not out_dir.exists():
                out_dir = Path(self.config_path).parent / "output"
            out_dir.mkdir(exist_ok=True, parents=True)
            output_path = str(out_dir / f"intrinsic_check_{self.camera_id}.jpg")

        composite = self._render_comparison(frame)
        cv2.imwrite(output_path, composite)
        logger.info("[%s] Saved undistortion comparison to %s", self.camera_id, output_path)
        print(f"  ✓  Undistortion comparison saved to {output_path}")
        return output_path

    def _render_comparison(self, frame: np.ndarray) -> np.ndarray:
        """Helper to create the side-by-side comparison image."""
        undistorted = self.undistort_frame(frame)

        # Draw reference grid on both panels
        distorted_grid   = _draw_reference_grid(frame.copy())
        undistorted_grid = _draw_reference_grid(undistorted.copy())

        # Resize panels to same height if needed
        h1, w1 = distorted_grid.shape[:2]
        h2, w2 = undistorted_grid.shape[:2]
        if h1 != h2:
            undistorted_grid = cv2.resize(undistorted_grid, (w1, h1))

        # Add title banners
        banner_h = 36
        banner_d = np.zeros((banner_h, w1, 3), dtype=np.uint8)
        banner_u = np.zeros((banner_h, w1, 3), dtype=np.uint8)
        cv2.putText(banner_d, "ORIGINAL (distorted)",   (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 140, 255), 2)
        cv2.putText(banner_u, "UNDISTORTED",             (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 140), 2)

        panel_d = np.vstack([banner_d, distorted_grid])
        panel_u = np.vstack([banner_u, undistorted_grid])
        composite = np.hstack([panel_d, panel_u])

        # Add metric info footer
        if self.is_calibrated:
            info = (
                f"RMS={self._rms:.3f}px  "
                f"k1={self.dist_coeffs[0,0]:.4f}  "
                f"k2={self.dist_coeffs[0,1]:.4f}"
            )
            cv2.putText(composite, info, (8, composite.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Optional: scale down if HUGE (for UI viewability)
        comp_h, comp_w = composite.shape[:2]
        if comp_w > 2000:
            scale = 2000 / comp_w
            composite = cv2.resize(composite, (0, 0), fx=scale, fy=scale)

        return composite

    def print_summary(self) -> None:
        """Print a human-readable summary of the current calibration state."""
        print(f"\n[LensCorrector — {self.camera_id}]")
        if not self.is_calibrated:
            print("  Status : NOT calibrated")
            return
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        k  = self.dist_coeffs.flatten()
        print(f"  Status    : calibrated  (RMS={self._rms:.4f} px)")
        print(f"  Image size: {self._image_size}")
        print(f"  fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")
        print(f"  k1={k[0]:.5f}  k2={k[1]:.5f}  p1={k[2]:.5f}  p2={k[3]:.5f}", end="")
        if len(k) > 4:
            print(f"  k3={k[4]:.5f}")
        else:
            print()
        print(f"  Remap maps: {'ready' if self.map1 is not None else 'not built'}")


# ===========================================================================
# Module-level helpers
# ===========================================================================

def _print_marker_grid_diagram(cols: int, rows: int) -> None:
    """Print an ASCII diagram showing the expected click order."""
    print("  Expected click order:")
    n = 1
    for r in range(rows):
        row_str = "    "
        for c in range(cols):
            row_str += f"[{n:2d}]"
            n += 1
        print(row_str)
    print("  (top-left = #1, row by row, left to right)")


def _build_object_points(cols: int, rows: int, square_m: float) -> np.ndarray:
    """Return (cols*rows, 3) float32 object-space coordinates for one board view."""
    objp = np.zeros((cols * rows, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_m
    return objp


def _is_live_source(source: str) -> bool:
    """Return True if the source looks like a live feed (not a local file)."""
    if source.isdigit():
        return True
    lower = source.lower()
    if lower.startswith(("rtsp://", "rtmp://", "http://", "https://")):
        return True
    # If the file exists on disk → treat as recorded video
    if os.path.exists(source):
        return False
    # Unknown: treat as live
    return True


def _draw_reference_grid(
    frame: np.ndarray,
    n_cols: int = 8,
    n_rows: int = 6,
) -> np.ndarray:
    """
    Draw a thin reference grid over *frame* so distortion curvature is visible.
    Returns the modified frame.
    """
    h, w = frame.shape[:2]
    col_step = w // n_cols
    row_step = h // n_rows
    colour = (0, 200, 255)
    thickness = 1

    for x in range(0, w, col_step):
        cv2.line(frame, (x, 0), (x, h), colour, thickness)
    for y in range(0, h, row_step):
        cv2.line(frame, (0, y), (w, y), colour, thickness)

    return frame


# ===========================================================================
# Convenience: load all correctors from cameras.json
# ===========================================================================

def load_all_correctors(config_path: str = "config/") -> dict[str, LensCorrector]:
    """
    Instantiate a LensCorrector for every camera in cameras.json.

    Returns
    -------
    dict[str, LensCorrector]
        Keyed by camera_id.  Each corrector is either loaded (is_calibrated=True)
        or empty (is_calibrated=False) if no .npz or inline data exists.
    """
    json_path = os.path.join(config_path, "cameras.json")
    with open(json_path) as f:
        config = json.load(f)

    return {
        cam["id"]: LensCorrector(cam["id"], config_path)
        for cam in config["cameras"]
    }
