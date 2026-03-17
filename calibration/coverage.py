"""
calibration/coverage.py — Interactive floor-coverage polygon tool.

Lets you define a camera's floor_coverage_polygon by clicking corners of the
visible floor area directly on the camera feed, instead of editing JSON by hand.

Two modes depending on whether the camera is already homography-calibrated:

  CALIBRATED (--calibrate already done)
      Click a floor corner → floor coordinates appear instantly on screen.
      No typing needed.  Fast, accurate.

  UNCALIBRATED
      Click a floor corner → terminal asks for the real-world metres.
      Useful when setting up coverage before full calibration.

Usage
-----
    python main.py --coverage cam_1
    python main.py --coverage cam_1 --source rtsp://...
    python main.py --coverage cam_1 --source video.mp4
    python main.py --coverage cam_1 --source frame.jpg
    python main.py --coverage cam_1 --source frame.png

Keys during the session
-----------------------
    Left-click   Add a corner point
    U            Undo last point
    C            Clear all points (start over)
    V            Verify — show saved polygon on the floor renderer
    Enter / S    Save polygon to cameras.json and exit
    Q / Esc      Cancel (no changes saved)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Image file extensions treated as static frames (not video)
_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# Maximum display window size (proportional fit, never upscale)
_MAX_WIN_W = 1400
_MAX_WIN_H = 900


def _win_size(frame_w: int, frame_h: int) -> tuple[int, int]:
    """Return (w, h) that fits frame on-screen without distorting content."""
    if frame_w <= 0 or frame_h <= 0:
        return _MAX_WIN_W, _MAX_WIN_H
    scale = min(_MAX_WIN_W / frame_w, _MAX_WIN_H / frame_h, 1.0)
    return max(1, int(frame_w * scale)), max(1, int(frame_h * scale))

# ── Project-root relative imports ─────────────────────────────────────────────
_HERE       = Path(__file__).parent
_PROJ_ROOT  = _HERE.parent
_CONFIG_DIR = _PROJ_ROOT / "config"


class CoverageMapper:
    """
    Interactive tool to define ``floor_coverage_polygon`` for one camera.

    Parameters
    ----------
    camera_id   : str   — must match an entry in cameras.json
    source      : str   — video file, RTSP URL, or None (read from cameras.json)
    config_path : str   — directory containing cameras.json
    """

    # ── Overlay colours (BGR) ─────────────────────────────────────────────────
    _COL_POINT   = (0,   215, 255)   # yellow — clicked corners
    _COL_LINE    = (0,   215, 255)   # yellow — polygon edges
    _COL_FILL    = (0,   200, 240)   # semi-transparent fill
    _COL_LABEL   = (255, 255, 255)   # white — coordinate labels
    _COL_SHADOW  = (0,   0,   0)     # black text shadow
    _COL_CURSOR  = (80,  255, 80)    # green — cursor crosshair
    _COL_INSTRUCT= (220, 220, 220)   # light gray instruction text
    _COL_OK      = (80,  220, 80)    # green status
    _COL_WARN    = (0,   165, 255)   # orange warning

    def __init__(
        self,
        camera_id:   str,
        source:      Optional[str] = None,
        config_path: str           = "config/",
    ) -> None:
        self.camera_id   = camera_id
        self.config_path = Path(config_path)
        self.cameras_cfg = self.config_path / "cameras.json"

        # Resolve source from cameras.json if not supplied
        if source:
            self.source = source
        else:
            with open(self.cameras_cfg) as f:
                cfg = json.load(f)
            entry = next((c for c in cfg["cameras"] if c["id"] == camera_id), None)
            if entry is None:
                print(f"  ✗  Camera '{camera_id}' not found in cameras.json")
                sys.exit(1)
            self.source = entry["source"]

        # ── Try to load existing homography for auto-coordinate conversion ───
        self._homography = None
        try:
            from calibration.homography import HomographyMapper
            hm = HomographyMapper(camera_id, config_path=str(config_path))
            if hm.is_calibrated:
                self._homography = hm
                print(f"  ✓  Homography loaded for {camera_id} — "
                      f"clicks will auto-convert to floor metres.")
            else:
                print(f"  ℹ  No homography for {camera_id} — "
                      f"you will type coordinates for each point.")
        except Exception:
            print(f"  ℹ  Could not load homography — "
                  f"you will type coordinates for each point.")

        # ── State ─────────────────────────────────────────────────────────────
        self._pixel_pts:  list[tuple[int, int]]     = []   # clicked image pixels
        self._floor_pts:  list[tuple[float, float]] = []   # real-world metres
        self._cursor:     tuple[int, int]            = (0, 0)
        self._base_frame: Optional[np.ndarray]      = None  # original frame
        self._disp_scale: float                     = 1.0  # display→image scale factor

    # ══════════════════════════════════════════════════════════════════════════
    #  Public entry point
    # ══════════════════════════════════════════════════════════════════════════

    def run(self) -> bool:
        """
        Run the interactive session.

        Returns
        -------
        bool  True if polygon was saved successfully.
        """
        # ── Load one frame (image file OR video/RTSP) ─────────────────────────
        frame = self._read_source_frame()
        if frame is None:
            return False

        # Store the original frame; compute a display scale so the image fits
        # within _MAX_WIN bounds without any resizeWindow calls (WINDOW_AUTOSIZE
        # is the most reliable approach on Qt5 / GTK backends).
        h, w = frame.shape[:2]
        self._disp_scale = min(_MAX_WIN_W / max(w, 1), _MAX_WIN_H / max(h, 1), 1.0)
        self._base_frame = frame.copy()   # always stored at original resolution

        self._print_instructions()

        # ── Window setup: WINDOW_AUTOSIZE — window matches image exactly ──────
        # ASCII-only name: Qt5 may silently fail to register unicode window names.
        win = f"Coverage: {self.camera_id}"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(win, self._redraw())   # show scaled frame immediately
        # Pump event loop several times — Qt5 needs more than 1 ms to register
        # the native window handle before setMouseCallback can use it.
        for _ in range(5):
            cv2.waitKey(20)
        cv2.setMouseCallback(win, self._on_mouse)

        # ── Interaction loop ──────────────────────────────────────────────────
        saved = False
        while True:
            canvas = self._redraw()
            cv2.imshow(win, canvas)
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), ord("Q"), 27):         # Q / Esc — cancel
                print("\n  ✗  Cancelled — no changes saved.")
                break

            elif key in (ord("u"), ord("U")):           # U — undo
                if self._pixel_pts:
                    self._pixel_pts.pop()
                    self._floor_pts.pop()
                    print(f"  Undo — {len(self._pixel_pts)} point(s) remaining.")

            elif key in (ord("c"), ord("C")):           # C — clear
                self._pixel_pts.clear()
                self._floor_pts.clear()
                print("  Cleared all points.")

            elif key in (13, ord("s"), ord("S")):       # Enter / S — save
                if len(self._floor_pts) < 3:
                    print(f"  ⚠  Need at least 3 points (have {len(self._floor_pts)}).")
                    continue
                saved = self._save()
                if saved:
                    break

            elif key in (ord("v"), ord("V")):           # V — verify on floor map
                self._verify_on_floor_map()

        cv2.destroyWindow(win)
        return saved

    # ══════════════════════════════════════════════════════════════════════════
    #  Frame loading (image file or video/RTSP)
    # ══════════════════════════════════════════════════════════════════════════

    def _read_source_frame(self) -> Optional[np.ndarray]:
        """
        Return a single BGR frame from *self.source*.

        Supports:
          • Still image files  (.jpg .jpeg .png .bmp .tiff .webp)
          • Video files        (.mp4 .avi .mkv …)
          • RTSP streams       (rtsp://…)
        """
        src = str(self.source)

        # ── Still image ───────────────────────────────────────────────────
        if Path(src).suffix.lower() in _IMAGE_EXTS:
            frame = cv2.imread(src)
            if frame is None:
                print(f"  ✗  Cannot read image file: {src}")
                return None
            print(f"  ✓  Loaded image: {src}  "
                  f"({frame.shape[1]}×{frame.shape[0]})")
            return frame

        # ── Video / RTSP ──────────────────────────────────────────────────
        if src.lower().startswith("rtsp"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            print(f"  ✗  Cannot open source: {src}")
            return None

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            print(f"  ✗  Cannot read frame from: {src}")
            return None
        return frame

    # ══════════════════════════════════════════════════════════════════════════
    #  Mouse callback
    # ══════════════════════════════════════════════════════════════════════════

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        # x, y are in display (scaled) space — unscale to original frame space
        s = self._disp_scale
        orig_x = int(round(x / s))
        orig_y = int(round(y / s))
        self._cursor = (orig_x, orig_y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self._add_point(orig_x, orig_y)

    # ══════════════════════════════════════════════════════════════════════════
    #  Point addition
    # ══════════════════════════════════════════════════════════════════════════

    def _add_point(self, px: int, py: int) -> None:
        """Add a corner point — auto-convert or prompt for floor coords."""
        n = len(self._pixel_pts) + 1

        if self._homography is not None:
            # ── Auto mode: homography does the conversion ─────────────────────
            fx, fy = self._homography.pixel_to_floor(px, py)
            if fx is None or fy is None:
                print(f"  ⚠  Point {n} ({px},{py}) could not be mapped — "
                      f"try a different location.")
                return
            self._pixel_pts.append((px, py))
            self._floor_pts.append((round(float(fx), 2), round(float(fy), 2)))
            print(f"  Point {n:2d} → pixel ({px:4d},{py:4d})  "
                  f"→ floor ({fx:6.2f}, {fy:6.2f}) m  [auto]")

        else:
            # ── Manual mode: user types real-world metres ─────────────────────
            print(f"\n  Point {n} — pixel ({px}, {py})")
            try:
                fx_s = input("    Floor X (m) → ").strip()
                fy_s = input("    Floor Y (m) → ").strip()
                if not fx_s or not fy_s:
                    print("  Skipped (empty input).")
                    return
                fx, fy = float(fx_s), float(fy_s)
            except ValueError:
                print("  ✗  Invalid number — point skipped.")
                return
            self._pixel_pts.append((px, py))
            self._floor_pts.append((round(fx, 2), round(fy, 2)))
            print(f"  Point {n:2d} → pixel ({px:4d},{py:4d})  "
                  f"→ floor ({fx:6.2f}, {fy:6.2f}) m  [manual]")

    # ══════════════════════════════════════════════════════════════════════════
    #  Drawing
    # ══════════════════════════════════════════════════════════════════════════

    def _redraw(self) -> np.ndarray:
        canvas = self._base_frame.copy()
        pts    = self._pixel_pts
        n      = len(pts)

        # ── Semi-transparent polygon fill (3+ points) ─────────────────────────
        if n >= 3:
            overlay = canvas.copy()
            poly    = np.array(pts, dtype=np.int32)
            cv2.fillPoly(overlay, [poly], self._COL_FILL)
            cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0, canvas)

        # ── Polygon edges ─────────────────────────────────────────────────────
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n] if i < n - 1 else None
            if p2:
                cv2.line(canvas, p1, p2, self._COL_LINE, 2, cv2.LINE_AA)

        # Closing edge preview (dashed) from last point to cursor
        if n >= 1:
            self._draw_dashed_line(canvas, pts[-1], self._cursor,
                                   (180, 180, 60), 1)
        # Closing edge preview (dashed) from cursor to first point
        if n >= 2:
            self._draw_dashed_line(canvas, self._cursor, pts[0],
                                   (100, 100, 40), 1)

        # ── Corner markers + coordinate labels ────────────────────────────────
        for i, (ppx, ppy) in enumerate(pts):
            fx, fy = self._floor_pts[i]

            # Outer ring
            cv2.circle(canvas, (ppx, ppy), 10, (0, 0, 0),     2, cv2.LINE_AA)
            cv2.circle(canvas, (ppx, ppy), 10, self._COL_POINT,2, cv2.LINE_AA)
            # Filled dot
            cv2.circle(canvas, (ppx, ppy),  5, self._COL_POINT,-1, cv2.LINE_AA)
            # Point index
            cv2.putText(canvas, str(i + 1), (ppx + 13, ppy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0),       2, cv2.LINE_AA)
            cv2.putText(canvas, str(i + 1), (ppx + 13, ppy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, self._COL_POINT,  1, cv2.LINE_AA)
            # Coordinate label
            coord_lbl = f"({fx:.1f}, {fy:.1f}) m"
            cv2.putText(canvas, coord_lbl, (ppx + 13, ppy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, self._COL_SHADOW, 2, cv2.LINE_AA)
            cv2.putText(canvas, coord_lbl, (ppx + 13, ppy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, self._COL_LABEL,  1, cv2.LINE_AA)

        # ── Cursor crosshair ──────────────────────────────────────────────────
        cx, cy = self._cursor
        cv2.line(canvas, (cx - 12, cy), (cx + 12, cy),
                 self._COL_CURSOR, 1, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy - 12), (cx, cy + 12),
                 self._COL_CURSOR, 1, cv2.LINE_AA)

        # ── Instruction overlay (bottom strip) ───────────────────────────────
        self._draw_instructions(canvas, n)

        # ── Scale to fit screen (WINDOW_AUTOSIZE approach) ────────────────────
        if self._disp_scale < 1.0:
            dh_orig, dw_orig = canvas.shape[:2]
            dw = max(1, int(dw_orig * self._disp_scale))
            dh = max(1, int(dh_orig * self._disp_scale))
            canvas = cv2.resize(canvas, (dw, dh), interpolation=cv2.INTER_AREA)

        return canvas

    def _draw_instructions(self, canvas: np.ndarray, n_pts: int) -> None:
        h, w = canvas.shape[:2]
        bar_h = 52
        # Dark bar at bottom
        cv2.rectangle(canvas, (0, h - bar_h), (w, h), (30, 30, 28), -1)
        cv2.line(canvas, (0, h - bar_h), (w, h - bar_h), (70, 70, 68), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        y1   = h - bar_h + 18
        y2   = h - 8

        # Status
        mode_str = "AUTO (homography)" if self._homography else "MANUAL (type coords)"
        status   = (f"  {self.camera_id}  |  "
                    f"Mode: {mode_str}  |  "
                    f"Points: {n_pts}")
        col_s = self._COL_OK if n_pts >= 3 else self._COL_WARN
        cv2.putText(canvas, status, (8, y1), font, 0.43, col_s, 1, cv2.LINE_AA)

        # Keys
        keys = ("[Click] Add point    "
                "[U] Undo    [C] Clear    "
                "[V] Verify on map    "
                "[Enter/S] Save    [Q] Cancel")
        cv2.putText(canvas, keys, (8, y2), font, 0.38,
                    self._COL_INSTRUCT, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_dashed_line(
        canvas: np.ndarray,
        p1: tuple[int, int],
        p2: tuple[int, int],
        color: tuple,
        thickness: int = 1,
        dash: int = 8,
        gap: int  = 5,
    ) -> None:
        dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < 1:
            return
        ux, uy = dx / dist, dy / dist
        d, drawing = 0.0, True
        while d < dist:
            seg = dash if drawing else gap
            d2  = min(d + seg, dist)
            if drawing:
                x1 = int(round(p1[0] + ux * d));  y1 = int(round(p1[1] + uy * d))
                x2 = int(round(p1[0] + ux * d2)); y2 = int(round(p1[1] + uy * d2))
                cv2.line(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
            d = d2; drawing = not drawing

    # ══════════════════════════════════════════════════════════════════════════
    #  Verify on floor renderer
    # ══════════════════════════════════════════════════════════════════════════

    def _verify_on_floor_map(self) -> None:
        """Show the polygon on the top-down floor renderer."""
        if len(self._floor_pts) < 3:
            print("  ⚠  Need at least 3 points to verify.")
            return

        try:
            from visualization.floor_renderer import load_renderer
        except ImportError:
            print("  ⚠  Could not import FloorRenderer — skipping verify.")
            return

        # Temporarily patch polygon into a renderer
        try:
            renderer = load_renderer(str(self.config_path))
            # Override this camera's polygon with the new one
            if self.camera_id in renderer._cam_data:
                renderer._cam_data[self.camera_id]["polygon"] = list(self._floor_pts)

            preview = renderer.render([])
            win = f"Verify Coverage — {self.camera_id}  (any key to close)"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, renderer.canvas_w, renderer.canvas_h)
            cv2.imshow(win, preview)
            print("  ▷  Floor map preview — press any key to close.")
            cv2.waitKey(0)
            cv2.destroyWindow(win)
        except Exception as e:
            print(f"  ⚠  Verify failed: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    #  Save
    # ══════════════════════════════════════════════════════════════════════════

    def _save(self) -> bool:
        """Write polygon to cameras.json under this camera's entry."""
        poly = [list(pt) for pt in self._floor_pts]

        try:
            with open(self.cameras_cfg) as f:
                cfg = json.load(f)

            updated = False
            for cam in cfg["cameras"]:
                if cam["id"] == self.camera_id:
                    old = cam.get("floor_coverage_polygon", [])
                    cam["floor_coverage_polygon"] = poly
                    updated = True
                    break

            if not updated:
                print(f"  ✗  Camera '{self.camera_id}' not found in cameras.json")
                return False

            with open(self.cameras_cfg, "w") as f:
                json.dump(cfg, f, indent=2)

            print(f"\n  ✓  Polygon saved to cameras.json")
            print(f"     Camera  : {self.camera_id}")
            print(f"     Points  : {len(poly)}")
            print(f"     Polygon : {poly}")
            if old:
                print(f"     Previous: {old}")
            return True

        except Exception as e:
            print(f"  ✗  Save failed: {e}")
            return False

    # ══════════════════════════════════════════════════════════════════════════
    #  Instructions
    # ══════════════════════════════════════════════════════════════════════════

    def _print_instructions(self) -> None:
        mode = ("AUTO — clicks auto-convert via homography"
                if self._homography
                else "MANUAL — terminal will ask for each coordinate")
        print()
        print("═" * 64)
        print(f"  Coverage Polygon Tool  —  {self.camera_id}")
        print("═" * 64)
        print(f"  Mode    : {mode}")
        print(f"  Source  : {self.source}")
        print()
        print("  HOW TO USE:")
        print("  ─────────────────────────────────────────────────────")
        print("  1. Click the corners of the floor area visible in the")
        print("     camera — go around the boundary (clockwise or CCW).")
        print("  2. The shape does NOT need to be a rectangle.")
        print("     Trapezoid / irregular polygon — all fine.")
        print("  3. Minimum 3 points, recommended 4–6.")
        print()
        print("  Click] Add point     [U] Undo     [C] Clear all")
        print("  [V] Preview on floor map           ")
        print("  [Enter] or [S] Save  [Q] Cancel")
        print("═" * 64)
        print()

