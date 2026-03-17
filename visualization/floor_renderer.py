"""
floor_renderer.py — Customer-Facing Factory Floor Visualisation

Renders a top-down, metric 2-D view of the factory floor with:

  • Warm gray background, white floor rectangle, dark border
  • Minor (1 m) and major (5 m) anti-aliased grid lines
  • Per-camera coverage polygons — semi-transparent, colour-coded
  • Overlap zones     — yellow semi-transparent fill + dashed yellow border
  • Person detections — RED circle (single) or PURPLE diamond (fused)
  • Status bar (top)  — Persons / Fused / FPS
  • Right-side legend — symbols + camera swatches + per-camera counts

Optional composite view: floor map (left) + stacked camera feeds (right).

Coordinate conventions
----------------------
Floor space  : (0, 0) = bottom-left; x = right (m); y = up (m)
Canvas space : (0, 0) = top-left  ; x = right (px); y = down (px)

Transform
---------
    cx = MARGIN_L + fx * px_per_m
    cy = STATUS_H + MARGIN_T + floor_h_px - fy * px_per_m
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from fusion.fuse import FusedDetection
from fusion.overlap import OverlapZone

logger = logging.getLogger(__name__)

# ── Config defaults ────────────────────────────────────────────────────────
_PROJECT_ROOT   = Path(__file__).parent.parent
_FLOOR_CFG      = _PROJECT_ROOT / "config" / "floor_config.json"
_CAMERAS_CFG    = _PROJECT_ROOT / "config" / "cameras.json"
_OVERLAP_CFG    = _PROJECT_ROOT / "config" / "overlap_zones.json"
_EDGES_CFG      = _PROJECT_ROOT / "config" / "edges.json"


# ═══════════════════════════════════════════════════════════════════════════
#  FloorRenderer
# ═══════════════════════════════════════════════════════════════════════════

class FloorRenderer:
    """
    Renders a bird's-eye factory floor view as a BGR ``np.ndarray``.

    Parameters
    ----------
    floor_config   : dict | str | Path  — floor_config.json content or path
    cameras_config : dict | str | Path  — cameras.json content or path
    overlap_config : dict | str | Path  — overlap_zones.json content or path
    window_width   : int                — total canvas pixel width (default 1200)
    """

    # ── Layout constants (pixels) ─────────────────────────────────────────
    _STATUS_H  = 32   # status bar height at top
    _MARGIN_T  = 12   # gap between status bar and floor top edge
    _MARGIN_B  = 38   # gap below floor (X-axis labels)
    _MARGIN_L  = 50   # gap left of floor (Y-axis labels)
    _MARGIN_R  = 170  # right panel width (legend)

    # ── Color palette (all BGR) ───────────────────────────────────────────
    _BG            = (245, 245, 240)   # warm gray canvas background
    _FLOOR_FILL    = (255, 255, 255)   # white floor
    _FLOOR_BORDER  = (55,  55,  55)    # dark floor border
    _GRID_MINOR    = (200, 200, 192)   # 1 m grid line (slightly darker for visibility)
    _GRID_MAJOR    = (165, 165, 158)   # 5 m grid line
    _GRID_LABEL    = (110, 110, 105)   # axis numeric labels
    _STATUS_BG     = (45,  45,  42)    # status bar background
    _STATUS_TEXT   = (235, 235, 230)   # status bar text
    _OVERLAP_COL   = (0,   215, 255)   # yellow  (overlap fill + dashed border)
    _DET_SINGLE    = (30,  30,  210)   # red     (single-camera detection)
    _DET_FUSED     = (165, 25,  155)   # purple  (fused detection)
    _DET_LABEL     = (30,  30,  30)    # confidence label text
    _LEGEND_DIVIDER = (195, 195, 190)  # hairline between floor and legend
    _LEGEND_TEXT   = (50,  50,  50)    # legend text
    _LEGEND_HEAD   = (80,  80,  80)    # legend section headings
    _EDGE_LINE_COL = (30,  145, 200)   # amber/gold — virtual counting-edge lines

    # ── Alpha values ──────────────────────────────────────────────────────
    _CAM_ALPHA     = 0.08
    _OVERLAP_ALPHA = 0.12

    # ──────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        floor_config,
        cameras_config,
        overlap_config,
        window_width: int = 1200,
    ) -> None:
        self._floor_cfg   = self._coerce_cfg(floor_config)
        self._cameras_cfg = self._coerce_cfg(cameras_config)
        self._overlap_cfg = self._coerce_cfg(overlap_config)

        # ── Floor dimensions ──────────────────────────────────────────────
        self.floor_w    = float(self._floor_cfg["floor_width_m"])
        self.floor_h    = float(self._floor_cfg["floor_height_m"])
        self.grid_minor = float(self._floor_cfg.get("grid_cell_size_m",   1.0))
        self.grid_major = float(self._floor_cfg.get("major_grid_every_m", 5.0))

        # ── Floor origin — may be negative when cameras cover area left/below 0
        # auto_configure writes floor_origin_x_m / floor_origin_y_m whenever
        # any camera polygon extends into negative coordinates.
        self.floor_ox = float(self._floor_cfg.get("floor_origin_x_m", 0.0))
        self.floor_oy = float(self._floor_cfg.get("floor_origin_y_m", 0.0))

        # ── Pixel scaling — SNAPPED to integer for pixel-perfect 1 m cells ──
        #
        # If px_per_m is non-integer (e.g. 32.67), different 1 m cells round
        # to 32 or 33 px alternately → visible size jitter across the grid.
        # Fix: floor px_per_m to the nearest whole number so every 1 m step
        # is *exactly* the same number of pixels.
        #
        #   floor_w=30 m, window=1200, available=980 px
        #   → px_per_m = int(980/30) = 32   (not 32.67)
        #   → floor_px_w = 32 × 30 = 960 px (fits inside 980 px)
        #   → leftover 20 px → split evenly as extra left/right padding
        #   → every 1 m cell = exactly 32 × 32 pixels — true square
        #
        avail_px         = window_width - self._MARGIN_L - self._MARGIN_R
        self.px_per_m    = max(4, int(avail_px / self.floor_w))   # integer!
        self._floor_px_w = int(self.floor_w * self.px_per_m)       # exact px width
        self._floor_px_h = int(self.floor_h * self.px_per_m)       # exact px height

        # Centre the floor horizontally inside the available strip
        _extra           = avail_px - self._floor_px_w             # leftover pixels
        self._ox_offset  = _extra // 2                             # shift right by half

        # ── Canvas dimensions ─────────────────────────────────────────────
        self.canvas_w = window_width
        self.canvas_h = (
            self._STATUS_H + self._MARGIN_T
            + self._floor_px_h
            + self._MARGIN_B
        )

        # ── Canvas-coord of floor origin — bottom-left pixel of the floor rect
        # When floor_ox < 0, the canvas (0,0) maps to floor_ox,floor_oy so
        # negative-coordinate detections appear inside the canvas.
        self._ox = self._MARGIN_L + self._ox_offset
        self._oy = self._STATUS_H + self._MARGIN_T + self._floor_px_h

        # ── Camera colour / polygon data ──────────────────────────────────
        self._cam_data: dict[str, dict] = {}
        for cam in self._cameras_cfg.get("cameras", []):
            rgb = cam.get("color", [128, 128, 128])
            bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
            self._cam_data[cam["id"]] = {
                "color":   bgr,
                "polygon": cam.get("floor_coverage_polygon", []),
                "name":    cam.get("name", cam["id"]),
            }

        # ── Overlap zones ─────────────────────────────────────────────────
        self._overlap_zones: list[OverlapZone] = [
            OverlapZone(z)
            for z in self._overlap_cfg.get("overlap_zones", [])
        ]

        # ── Virtual counting edges (from edges.json) ──────────────────────
        self._edge_defs: list[dict] = []
        try:
            if _EDGES_CFG.exists():
                with open(_EDGES_CFG) as _ef:
                    _edata = json.load(_ef)
                self._edge_defs = _edata.get("edges", [])
        except Exception as _exc:
            logger.warning("Could not load edges.json: %s", _exc)

        # ── Toggle flags ──────────────────────────────────────────────────
        self._show_grid         = True
        self._show_cameras      = True
        self._show_overlap      = True
        self._show_cell_labels  = True   # 'L' key — show (col,row) in every cell
        self._show_edges        = True   # 'E' key — virtual counting-edge lines

        # ── FPS rolling average (last 30 frames) ─────────────────────────
        self._fps_times: deque[float] = deque(maxlen=30)

    # ══════════════════════════════════════════════════════════════════════
    #  Public API
    # ══════════════════════════════════════════════════════════════════════

    def render(
        self,
        fused_detections: Optional[list[FusedDetection]] = None,
        stats: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Render the floor map.

        Layers (bottom → top):
          1. Background     warm gray
          2. Status bar     "Persons: N | Fused: F | FPS: X"
          3. Floor rect     white fill, dark border
          4. Camera zones   semi-transparent colour polygons (α 0.08)
          5. Overlap zones  yellow semi-transparent fill (α 0.12) + dashed border
          6. Grid           1 m minor + 5 m major lines drawn ON TOP of zones
                            px_per_m is integer → pixel-perfect 1 m² cells
                            tick marks (+) at every intersection always visible
                            1 m scale bar in bottom-right corner of floor
          7. Detections     RED circle (single) / PURPLE diamond (fused)
          8. Axis labels    0 m … along edges
          9. Legend         right-side panel

        Parameters
        ----------
        fused_detections : list[FusedDetection] | None
        stats            : dict (from DetectionFuser.get_stats()) | None

        Returns
        -------
        np.ndarray  BGR image of shape (canvas_h, canvas_w, 3)
        """
        # ── FPS tracking ──────────────────────────────────────────────────
        now = time.perf_counter()
        self._fps_times.append(now)
        fps = 0.0
        if len(self._fps_times) >= 2:
            elapsed = self._fps_times[-1] - self._fps_times[0]
            if elapsed > 0:
                fps = (len(self._fps_times) - 1) / elapsed

        # ── Derive stats if not provided ──────────────────────────────────
        if stats is None and fused_detections:
            fc = sum(1 for f in fused_detections if f.is_fused)
            tc = len(fused_detections)
            stats = {
                "total_persons": tc,
                "fused_count":   fc,
                "single_count":  tc - fc,
                "by_camera":     {},
            }

        # ── Allocate canvas ───────────────────────────────────────────────
        canvas = np.full(
            (self.canvas_h, self.canvas_w, 3),
            self._BG,
            dtype=np.uint8,
        )

        # ── Layer 1: status bar ───────────────────────────────────────────
        self._draw_status_bar(canvas, stats, fps)

        # ── Layer 2: floor rectangle (white fill) ─────────────────────────
        self._draw_floor_rect(canvas)

        # ── Layer 3: camera zones (semi-transparent fill + outline) ───────
        #    Drawn BEFORE grid so the grid lines are always visible on top.
        if self._show_cameras:
            self._draw_camera_zones(canvas)

        # ── Layer 4: overlap zones (semi-transparent + dashed border) ─────
        #    Also drawn before grid so grid stays sharp across the overlap.
        if self._show_overlap:
            self._draw_overlap_zones(canvas)

        # ── Layer 5: grid — drawn ON TOP of all zone fills ────────────────
        #    px_per_m is an integer → every 1 m cell = exactly px_per_m²
        #    pixels → all cells are identically sized true squares.
        #    Tick marks at every intersection remain visible even when zone
        #    fills change the background colour.
        if self._show_grid:
            self._draw_grid(canvas)

        # ── Layer 5b: virtual counting edges ──────────────────────────────
        #    Amber lines drawn on top of the gray grid — one line per entry
        #    in edges.json.  When step_m=1.0, every 1 m gridline is an edge.
        if self._show_edges and self._edge_defs:
            self._draw_edge_lines(canvas)

        # ── Layer 6a: occupied cell highlights ───────────────────────────
        #    Subtle tint on each 1 m cell that contains at least one person.
        #    Drawn on top of grid lines so the colour is clearly visible.
        if fused_detections:
            self._draw_occupied_cells(canvas, fused_detections)

        # ── Layer 6b: grid cell labels ────────────────────────────────────
        #    When _show_cell_labels=True  → "(col,row)" centred in EVERY cell.
        #    Always                       → "(col,row)" in corner of OCCUPIED cells.
        self._draw_cell_labels(canvas, fused_detections or [])

        # ── Layer 7: detections ───────────────────────────────────────────
        if fused_detections:
            self._draw_detections(canvas, fused_detections)

        # ── Layer 8: axis labels ──────────────────────────────────────────
        self._draw_axis_labels(canvas)

        # ── Layer 9: legend ───────────────────────────────────────────────
        self._draw_legend(canvas, stats)

        return canvas

    def render_with_camera_feeds(
        self,
        camera_frames: dict[str, np.ndarray],
        fused_detections: Optional[list[FusedDetection]] = None,
        stats: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Composite view: floor map (left) + camera feeds stacked (right).

        Parameters
        ----------
        camera_frames     : dict[cam_id → BGR np.ndarray]
        fused_detections  : list[FusedDetection] | None
        stats             : dict | None

        Returns
        -------
        np.ndarray  Wide BGR image (canvas_h × combined_width)
        """
        floor_img = self.render(fused_detections, stats)

        if not camera_frames:
            return floor_img

        n_cams       = len(camera_frames)
        cam_ids      = sorted(camera_frames.keys())
        # Camera panel is as wide as the floor map itself so feeds are large
        cam_panel_w  = self.canvas_w
        header_h     = 32                              # per-camera header

        cam_panel = np.full(
            (self.canvas_h, cam_panel_w, 3),
            (50, 50, 48),
            dtype=np.uint8,
        )

        slot_h_base = self.canvas_h // n_cams

        for i, cam_id in enumerate(cam_ids):
            y_start  = i * slot_h_base
            y_end    = y_start + slot_h_base if i < n_cams - 1 else self.canvas_h
            slot_h   = y_end - y_start

            cam_data  = self._cam_data.get(cam_id, {})
            cam_color = cam_data.get("color", (130, 130, 130))
            cam_name  = cam_data.get("name", cam_id)

            # Coloured header strip
            cv2.rectangle(
                cam_panel,
                (0, y_start), (cam_panel_w, y_start + header_h),
                cam_color, -1,
            )
            cv2.putText(
                cam_panel, cam_name,
                (8, y_start + header_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )

            # Scale frame to fit the slot below the header
            frame    = camera_frames[cam_id]
            avail_h  = slot_h - header_h
            avail_w  = cam_panel_w
            fh, fw   = frame.shape[:2]

            if fh > 0 and fw > 0:
                scale  = min(avail_w / fw, avail_h / fh)
                new_w  = max(1, int(fw * scale))
                new_h  = max(1, int(fh * scale))
                scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Centre in the slot
                x_off = (avail_w - new_w) // 2
                y_off = y_start + header_h + (avail_h - new_h) // 2

                # Safe clamp before copy
                y1 = max(0, y_off);                     y2 = min(cam_panel.shape[0], y_off + new_h)
                x1 = max(0, x_off);                     x2 = min(cam_panel.shape[1], x_off + new_w)
                sh = y2 - y1;                            sw = x2 - x1
                if sh > 0 and sw > 0:
                    cam_panel[y1:y2, x1:x2] = scaled[:sh, :sw]

            # Thin separator line between cameras
            if i < n_cams - 1:
                cv2.line(
                    cam_panel,
                    (0, y_end - 1), (cam_panel_w, y_end - 1),
                    (80, 80, 78), 1,
                )

        # Thin vertical separator between floor map and camera panel
        sep = np.full((self.canvas_h, 2, 3), (90, 90, 88), dtype=np.uint8)
        return np.hstack([floor_img, sep, cam_panel])

    # ── Toggles ───────────────────────────────────────────────────────────

    def toggle_grid(self) -> bool:
        """Toggle grid visibility.  Returns new state."""
        self._show_grid = not self._show_grid
        return self._show_grid

    def toggle_cameras(self) -> bool:
        """Toggle camera-zone overlay visibility.  Returns new state."""
        self._show_cameras = not self._show_cameras
        return self._show_cameras

    def toggle_overlap(self) -> bool:
        """Toggle overlap-zone overlay visibility.  Returns new state."""
        self._show_overlap = not self._show_overlap
        return self._show_overlap

    def toggle_edges(self) -> bool:
        """Toggle virtual counting-edge line overlay.  Returns new state."""
        self._show_edges = not self._show_edges
        return self._show_edges

    # ── Snapshot ──────────────────────────────────────────────────────────

    def save_snapshot(self, img: np.ndarray, filepath: str | Path) -> bool:
        """
        Save a rendered frame to disk.

        Parameters
        ----------
        img      : np.ndarray   — BGR image (e.g. from render())
        filepath : str | Path   — output path (.png / .jpg)

        Returns
        -------
        bool  True on success
        """
        ok = cv2.imwrite(str(filepath), img)
        if ok:
            logger.info("Snapshot saved → %s", filepath)
        else:
            logger.error("Failed to save snapshot to %s", filepath)
        return ok

    # ── Coordinate helpers ────────────────────────────────────────────────

    def floor_to_canvas(self, fx: float, fy: float) -> tuple[int, int]:
        """Floor metres → canvas pixels.  Supports negative floor coordinates."""
        cx = int(round(self._ox + (fx - self.floor_ox) * self.px_per_m))
        cy = int(round(self._oy - (fy - self.floor_oy) * self.px_per_m))
        return cx, cy

    def canvas_to_floor(self, cx: int, cy: int) -> tuple[float, float]:
        """Canvas pixels → floor metres (inverse of floor_to_canvas)."""
        fx = (cx - self._ox) / self.px_per_m + self.floor_ox
        fy = (self._oy - cy) / self.px_per_m + self.floor_oy
        return fx, fy

    # ══════════════════════════════════════════════════════════════════════
    #  Drawing layers (private)
    # ══════════════════════════════════════════════════════════════════════

    def _draw_status_bar(
        self,
        canvas: np.ndarray,
        stats: Optional[dict],
        fps: float,
    ) -> None:
        """Dark strip at top: title  |  Persons/Fused  |  FPS."""
        # Background
        cv2.rectangle(canvas, (0, 0), (self.canvas_w, self._STATUS_H),
                      self._STATUS_BG, -1)
        # Thin bottom edge
        cv2.line(canvas,
                 (0, self._STATUS_H - 1), (self.canvas_w, self._STATUS_H - 1),
                 (70, 70, 68), 1)

        font  = cv2.FONT_HERSHEY_SIMPLEX
        sz    = 0.48
        color = self._STATUS_TEXT
        y_txt = self._STATUS_H - 10

        # Left: title
        cv2.putText(canvas, "FACTORY FLOOR MONITOR",
                    (self._MARGIN_L, y_txt), font, sz, (200, 200, 195), 1, cv2.LINE_AA)

        # Centre: counts
        if stats:
            total  = stats.get("total_persons", 0)
            fused  = stats.get("fused_count",   0)
            single = stats.get("single_count",  0)
            centre_text = (
                f"Persons: {total}   |   "
                f"Fused: {fused}   |   "
                f"Single: {single}"
            )
        else:
            centre_text = "Persons: --  |  Fused: --  |  Single: --"

        (tw, _), _ = cv2.getTextSize(centre_text, font, sz, 1)
        cx_txt = (self.canvas_w - self._MARGIN_R) // 2 - tw // 2
        cv2.putText(canvas, centre_text,
                    (cx_txt, y_txt), font, sz, color, 1, cv2.LINE_AA)

        # Right: FPS
        fps_str = f"FPS: {fps:.1f}"
        (fw, _), _ = cv2.getTextSize(fps_str, font, sz, 1)
        cv2.putText(canvas, fps_str,
                    (self.canvas_w - self._MARGIN_R - fw - 10, y_txt),
                    font, sz, (160, 220, 130), 1, cv2.LINE_AA)

    def _draw_floor_rect(self, canvas: np.ndarray) -> None:
        """White floor rectangle with dark border."""
        tl = self.floor_to_canvas(0,          self.floor_h)
        br = self.floor_to_canvas(self.floor_w, 0)
        cv2.rectangle(canvas, tl, br, self._FLOOR_FILL,  -1)
        cv2.rectangle(canvas, tl, br, self._FLOOR_BORDER, 1, cv2.LINE_AA)

    def _draw_grid(self, canvas: np.ndarray) -> None:
        """
        Draw a pixel-perfect 1 m × 1 m grid on the floor area.

        Because ``px_per_m`` is snapped to an integer in ``__init__``, every
        1 m step is *exactly* ``px_per_m`` pixels.  Grid lines are placed by
        integer arithmetic (``i * px_per_m``) rather than accumulated floats,
        so ALL cells are identically sized regardless of floor dimensions.

        Visual layers inside this method (bottom → top):
          1. Minor lines  — 1 m spacing, very light gray
          2. Major lines  — 5 m spacing, medium gray (thicker)
          3. Tick marks   — small + cross at every 1 m intersection
                           Always visible on any background colour, confirming
                           grid consistency inside the overlap zone too.
          4. Scale bar    — "1 m" reference in the bottom-right corner
        """
        floor_top = self._STATUS_H + self._MARGIN_T
        floor_bot = self._oy
        floor_lft = self._ox
        floor_rgt = self._ox + self._floor_px_w

        # Integer step counts (exact — no off-by-one from float rounding)
        n_x = int(round(self.floor_w / self.grid_minor))   # e.g. 30
        n_y = int(round(self.floor_h / self.grid_minor))   # e.g. 20
        major_step = int(round(self.grid_major / self.grid_minor))  # e.g. 5
        px_step    = int(round(self.px_per_m * self.grid_minor))    # pixels per cell

        # ── 1: Minor grid lines ───────────────────────────────────────────
        for i in range(n_x + 1):
            px = floor_lft + i * px_step
            if i % major_step == 0:
                continue            # painted in pass 2 as major
            cv2.line(canvas, (px, floor_top), (px, floor_bot),
                     self._GRID_MINOR, 1, cv2.LINE_AA)

        for j in range(n_y + 1):
            py = floor_bot - j * px_step
            if j % major_step == 0:
                continue
            cv2.line(canvas, (floor_lft, py), (floor_rgt, py),
                     self._GRID_MINOR, 1, cv2.LINE_AA)

        # ── 2: Major grid lines (every 5 m) — slightly thicker + darker ──
        for i in range(n_x + 1):
            if i % major_step != 0:
                continue
            px = floor_lft + i * px_step
            cv2.line(canvas, (px, floor_top), (px, floor_bot),
                     self._GRID_MAJOR, 1, cv2.LINE_AA)

        for j in range(n_y + 1):
            if j % major_step != 0:
                continue
            py = floor_bot - j * px_step
            cv2.line(canvas, (floor_lft, py), (floor_rgt, py),
                     self._GRID_MAJOR, 1, cv2.LINE_AA)

        # ── 3: Tick marks (+) at every 1 m intersection ───────────────────
        # Small cross drawn on top of zone fills so the grid is always
        # clearly legible inside the overlap zone (yellow background).
        tick = max(3, px_step // 10)   # ~10 % of cell size, min 3 px
        for i in range(n_x + 1):
            for j in range(n_y + 1):
                px = floor_lft + i * px_step
                py = floor_bot  - j * px_step
                cv2.line(canvas, (px - tick, py), (px + tick, py),
                         self._GRID_MAJOR, 1, cv2.LINE_AA)
                cv2.line(canvas, (px, py - tick), (px, py + tick),
                         self._GRID_MAJOR, 1, cv2.LINE_AA)

        # ── 4: 1 m scale bar (bottom-right of floor) ─────────────────────
        self._draw_scale_bar(canvas, px_step)

    def _draw_camera_zones(self, canvas: np.ndarray) -> None:
        """Semi-transparent coloured polygons for each camera's coverage area."""
        overlay = canvas.copy()
        for cam_id, data in self._cam_data.items():
            pts = data["polygon"]
            if not pts:
                continue
            canvas_pts = np.array(
                [self.floor_to_canvas(p[0], p[1]) for p in pts], dtype=np.int32
            )
            cv2.fillPoly(overlay, [canvas_pts], data["color"])

        cv2.addWeighted(overlay, self._CAM_ALPHA, canvas, 1.0 - self._CAM_ALPHA,
                        0, canvas)

        # Polygon outlines + centroid labels (drawn on top, opaque)
        for cam_id, data in self._cam_data.items():
            pts = data["polygon"]
            if not pts:
                continue
            canvas_pts = np.array(
                [self.floor_to_canvas(p[0], p[1]) for p in pts], dtype=np.int32
            )
            cv2.polylines(canvas, [canvas_pts], True, data["color"], 1, cv2.LINE_AA)

            # Centroid label
            cx_m = sum(p[0] for p in pts) / len(pts)
            cy_m = sum(p[1] for p in pts) / len(pts)
            cx_px, cy_px = self.floor_to_canvas(cx_m, cy_m)
            cv2.putText(canvas, cam_id,
                        (cx_px - 18, cy_px),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, data["color"],
                        1, cv2.LINE_AA)

    def _draw_overlap_zones(self, canvas: np.ndarray) -> None:
        """Yellow semi-transparent fill + dashed yellow border for overlap zones."""
        if not self._overlap_zones:
            return

        # ── Semi-transparent fill ─────────────────────────────────────────
        overlay = canvas.copy()
        for zone in self._overlap_zones:
            coords     = list(zone.polygon.exterior.coords)[:-1]  # drop repeated last
            canvas_pts = np.array(
                [self.floor_to_canvas(p[0], p[1]) for p in coords], dtype=np.int32
            )
            cv2.fillPoly(overlay, [canvas_pts], self._OVERLAP_COL)

        cv2.addWeighted(overlay, self._OVERLAP_ALPHA, canvas,
                        1.0 - self._OVERLAP_ALPHA, 0, canvas)

        # ── Dashed border + label ─────────────────────────────────────────
        for zone in self._overlap_zones:
            coords     = list(zone.polygon.exterior.coords)[:-1]
            canvas_pts = np.array(
                [self.floor_to_canvas(p[0], p[1]) for p in coords], dtype=np.int32
            )
            self._draw_dashed_polygon(canvas, canvas_pts, self._OVERLAP_COL,
                                      thickness=2, dash_px=9, gap_px=5)

            # "OVERLAP" label at centroid
            cx_m = sum(p[0] for p in coords) / len(coords)
            cy_m = sum(p[1] for p in coords) / len(coords)
            cx_px, cy_px = self.floor_to_canvas(cx_m, cy_m)
            cv2.putText(canvas, "OVERLAP",
                        (cx_px - 30, cy_px),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, self._OVERLAP_COL,
                        1, cv2.LINE_AA)

    def _draw_edge_lines(self, canvas: np.ndarray) -> None:
        """
        Draw virtual counting-edge lines (from edges.json) in amber/gold.

        Vertical edges   (type='vertical')   → full-height column line at x=value
        Horizontal edges (type='horizontal') → full-width row line at y=value

        Each line is drawn as a thin semi-transparent amber overlay so it's
        clearly distinct from the gray grid while not overwhelming the other
        map elements.  Small edge-ID labels appear at the floor border.
        """
        if not self._edge_defs:
            return

        floor_top = self._STATUS_H + self._MARGIN_T
        floor_bot = self._oy
        floor_lft = self._ox
        floor_rgt = self._ox + self._floor_px_w
        col       = self._EDGE_LINE_COL
        font      = cv2.FONT_HERSHEY_SIMPLEX
        lbl_sz    = 0.26

        # Draw all lines onto a separate overlay for alpha blending
        overlay = canvas.copy()

        for edge in self._edge_defs:
            etype = edge.get("type", "")
            val   = float(edge["value"])
            eid   = str(edge.get("id", ""))

            if etype == "vertical":
                px, _ = self.floor_to_canvas(val, 0)
                if not (floor_lft <= px <= floor_rgt):
                    continue
                # Line through the full floor height
                cv2.line(overlay, (px, floor_top), (px, floor_bot), col, 1, cv2.LINE_AA)
                # Small label just below the floor bottom
                (tw, th), _ = cv2.getTextSize(eid, font, lbl_sz, 1)
                cv2.putText(canvas, eid,
                            (px - tw // 2, floor_bot + th + 2),
                            font, lbl_sz, col, 1, cv2.LINE_AA)

            elif etype == "horizontal":
                _, py = self.floor_to_canvas(0, val)
                if not (floor_top <= py <= floor_bot):
                    continue
                # Line through the full floor width
                cv2.line(overlay, (floor_lft, py), (floor_rgt, py), col, 1, cv2.LINE_AA)
                # Small label just to the left of the floor
                (tw, th), _ = cv2.getTextSize(eid, font, lbl_sz, 1)
                cv2.putText(canvas, eid,
                            (floor_lft - tw - 3, py + th // 2),
                            font, lbl_sz, col, 1, cv2.LINE_AA)

        # Blend: 35 % amber lines over the existing canvas so grid stays legible
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

    def _draw_detections(
        self,
        canvas: np.ndarray,
        detections: list[FusedDetection],
    ) -> None:
        """
        Single-camera : RED filled circle + camera-coloured outer ring.
        Fused         : PURPLE filled diamond + small numeric badge + white outline.
        Both          : confidence % label below the marker.
        """
        for fd in detections:
            cx, cy = self.floor_to_canvas(fd.floor_x, fd.floor_y)

            if fd.is_fused:
                self._draw_fused_marker(canvas, cx, cy, fd)
            else:
                self._draw_single_marker(canvas, cx, cy, fd)

            # Confidence label
            conf_str = f"{int(round(fd.confidence * 100))}%"
            cv2.putText(canvas, conf_str,
                        (cx - 10, cy + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, self._DET_LABEL,
                        1, cv2.LINE_AA)

            # Grid-cell name — always shown under the confidence label
            cell_str = self.get_grid_cell(fd.floor_x, fd.floor_y)
            (tw, _), _ = cv2.getTextSize(
                cell_str, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
            cv2.putText(canvas, cell_str,
                        (cx - tw // 2, cy + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (80, 80, 80),
                        1, cv2.LINE_AA)

    def _draw_single_marker(
        self,
        canvas: np.ndarray,
        cx: int,
        cy: int,
        fd: FusedDetection,
    ) -> None:
        """RED filled circle with camera-coloured outer ring."""
        # Camera-coloured outer ring
        cam_id    = fd.source_cameras[0] if fd.source_cameras else ""
        cam_color = self._cam_data.get(cam_id, {}).get("color", (160, 160, 160))
        cv2.circle(canvas, (cx, cy), 16, cam_color, 2, cv2.LINE_AA)
        # Red filled inner circle
        cv2.circle(canvas, (cx, cy), 10, self._DET_SINGLE, -1, cv2.LINE_AA)
        # White edge for contrast
        cv2.circle(canvas, (cx, cy), 10, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_fused_marker(
        self,
        canvas: np.ndarray,
        cx: int,
        cy: int,
        fd: FusedDetection,
    ) -> None:
        """GREEN filled circle with numeric source-count badge."""
        green_bgr = (40, 220, 60) # A nice bright green BGR
        cv2.circle(canvas, (cx, cy), 12, green_bgr, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 12, (255, 255, 255), 1, cv2.LINE_AA)

        # Badge: small circle top-right with camera count
        n_cams   = len(fd.source_cameras)
        bx, by   = cx + 9, cy - 9
        badge_r  = 8
        cv2.circle(canvas, (bx, by), badge_r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (bx, by), badge_r, self._DET_FUSED, 1, cv2.LINE_AA)
        n_str    = str(n_cams)
        (tw, th), _ = cv2.getTextSize(n_str, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(canvas, n_str,
                    (bx - tw // 2, by + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, self._DET_FUSED,
                    1, cv2.LINE_AA)

    def _draw_axis_labels(self, canvas: np.ndarray) -> None:
        """Numeric labels at major grid positions; axis-name labels."""
        font  = cv2.FONT_HERSHEY_SIMPLEX
        sz    = 0.36
        col   = self._GRID_LABEL

        # X axis — labels below floor
        y_lbl = self._oy + 18
        x = 0.0
        while x <= self.floor_w + 1e-9:
            if abs(x % self.grid_major) < 1e-6 or x == 0:
                px, _ = self.floor_to_canvas(x, 0)
                lbl   = f"{int(x)}m"
                (tw, _), _ = cv2.getTextSize(lbl, font, sz, 1)
                cv2.putText(canvas, lbl, (px - tw // 2, y_lbl),
                            font, sz, col, 1, cv2.LINE_AA)
            x += self.grid_major

        # Y axis — labels to the left of floor
        x_lbl = self._ox - 8
        y = 0.0
        while y <= self.floor_h + 1e-9:
            if abs(y % self.grid_major) < 1e-6 or y == 0:
                _, py = self.floor_to_canvas(0, y)
                lbl   = f"{int(y)}m"
                (tw, th), _ = cv2.getTextSize(lbl, font, sz, 1)
                cv2.putText(canvas, lbl, (x_lbl - tw, py + th // 2),
                            font, sz, col, 1, cv2.LINE_AA)
            y += self.grid_major

        # "X (m)" axis name at bottom-centre of floor
        x_name = "X (m)"
        (tw, _), _ = cv2.getTextSize(x_name, font, sz, 1)
        cv2.putText(canvas, x_name,
                    (self._ox + self._floor_px_w // 2 - tw // 2,
                     self.canvas_h - 6),
                    font, sz, col, 1, cv2.LINE_AA)

        # "Y (m)" axis name — draw sideways using two characters (no rotation needed)
        cv2.putText(canvas, "Y",
                    (6, self._STATUS_H + self._MARGIN_T + self._floor_px_h // 2 - 10),
                    font, sz, col, 1, cv2.LINE_AA)
        cv2.putText(canvas, "(m)",
                    (2, self._STATUS_H + self._MARGIN_T + self._floor_px_h // 2 + 10),
                    font, 0.30, col, 1, cv2.LINE_AA)

    def _draw_legend(
        self,
        canvas: np.ndarray,
        stats: Optional[dict],
    ) -> None:
        """Right-side legend panel: symbols + camera swatches + optional stats."""
        # ── Legend panel background ───────────────────────────────────────
        lx0 = self.canvas_w - self._MARGIN_R + 4
        ly0 = self._STATUS_H
        lx1 = self.canvas_w - 2
        ly1 = self.canvas_h

        cv2.rectangle(canvas, (lx0, ly0), (lx1, ly1), (240, 240, 236), -1)
        # Left border
        cv2.line(canvas, (lx0 - 1, ly0), (lx0 - 1, ly1),
                 self._LEGEND_DIVIDER, 1)

        font  = cv2.FONT_HERSHEY_SIMPLEX
        lx    = lx0 + 10   # text x start
        ly    = ly0 + 18   # current y cursor

        def _head(text: str) -> None:
            nonlocal ly
            cv2.putText(canvas, text, (lx, ly),
                        font, 0.38, self._LEGEND_HEAD, 1, cv2.LINE_AA)
            ly += 4
            cv2.line(canvas, (lx, ly), (lx0 + self._MARGIN_R - 14, ly),
                     self._LEGEND_DIVIDER, 1)
            ly += 14

        def _item(dot_fn, label: str) -> None:
            nonlocal ly
            dot_fn(ly)
            cv2.putText(canvas, label, (lx + 22, ly + 4),
                        font, 0.37, self._LEGEND_TEXT, 1, cv2.LINE_AA)
            ly += 22

        # ── DETECTIONS section ────────────────────────────────────────────
        _head("DETECTIONS")

        def _red_dot(y):
            cv2.circle(canvas, (lx + 8, y), 8,  self._DET_SINGLE, -1, cv2.LINE_AA)
            cv2.circle(canvas, (lx + 8, y), 8,  (255, 255, 255),   1, cv2.LINE_AA)
        _item(_red_dot, "Single camera")

        def _green_circle(y):
            green_bgr = (40, 220, 60)
            cv2.circle(canvas, (lx + 8, y), 9, green_bgr, -1, cv2.LINE_AA)
            cv2.circle(canvas, (lx + 8, y), 9, (255, 255, 255), 1, cv2.LINE_AA)
        _item(_green_circle, "Fused (2 cams)")

        ly += 6

        # ── CAMERAS section ───────────────────────────────────────────────
        _head("CAMERAS")

        for cam_id, data in self._cam_data.items():
            col  = data["color"]
            name = data["name"]

            # Colour swatch
            def _swatch(y, _col=col):
                cv2.rectangle(canvas, (lx + 1, y - 6), (lx + 15, y + 6), _col, -1)
                cv2.rectangle(canvas, (lx + 1, y - 6), (lx + 15, y + 6),
                              (80, 80, 80), 1)

            # Per-camera count from stats
            count_str = ""
            if stats and "by_camera" in stats:
                cnt = stats["by_camera"].get(cam_id)
                if cnt is not None:
                    count_str = f" ({cnt})"

            _item(_swatch, f"{cam_id}{count_str}")

        ly += 6

        # ── ZONES section ─────────────────────────────────────────────────
        _head("ZONES")

        def _dashed_line(y):
            for dx in range(0, 17, 6):
                x1 = lx + dx
                x2 = min(lx + dx + 4, lx + 16)
                cv2.line(canvas, (x1, y), (x2, y), self._OVERLAP_COL, 2, cv2.LINE_AA)
        _item(_dashed_line, "Overlap zone")

        def _edge_line(y):
            cv2.line(canvas, (lx + 1, y), (lx + 15, y), self._EDGE_LINE_COL, 2, cv2.LINE_AA)
            # small tick at center to show it's a counting line
            cv2.line(canvas, (lx + 8, y - 4), (lx + 8, y + 4),
                     self._EDGE_LINE_COL, 1, cv2.LINE_AA)
        _item(_edge_line, "Counting edge")

        # ── CONTROLS hint ─────────────────────────────────────────────────
        ly = min(ly + 16, ly1 - 80)
        cv2.line(canvas, (lx, ly), (lx0 + self._MARGIN_R - 14, ly),
                 self._LEGEND_DIVIDER, 1)
        ly += 12
        cv2.putText(canvas, "KEYS", (lx, ly),
                    font, 0.35, self._LEGEND_HEAD, 1, cv2.LINE_AA)
        ly += 16
        for hint in ["G - grid", "C - cameras", "O - overlap",
                     "E - edges", "V - cam grid",
                     "L - cell labels",
                     "J - JSON report", "S - snapshot"]:
            cv2.putText(canvas, hint, (lx, ly),
                        font, 0.34, self._LEGEND_TEXT, 1, cv2.LINE_AA)
            ly += 15
            if ly > ly1 - 10:
                break

    # ── Cell coordinate API ────────────────────────────────────────────────

    def toggle_cell_labels(self) -> bool:
        """Toggle display of (col,row) labels in every grid cell.  Returns new state."""
        self._show_cell_labels = not self._show_cell_labels
        return self._show_cell_labels

    def get_grid_cell(self, floor_x: float, floor_y: float) -> str:
        """
        Return the grid-cell name for a floor position.

        Naming convention
        -----------------
        ``(col, row)``  where
            col = floor(x_m)  clipped to [0, floor_width_m  - 1]
            row = floor(y_m)  clipped to [0, floor_height_m - 1]

        Examples (floor 30 × 20 m, cell size 1 m):
            (5.7, 3.2) → "(5,3)"   i.e. the square 5–6 m x, 3–4 m y
            (0.0, 0.0) → "(0,0)"   bottom-left corner cell
            (29.9,19.9)→ "(29,19)" top-right corner cell
        """
        import math
        col = int(max(0, min(math.floor(floor_x), int(self.floor_w) - 1)))
        row = int(max(0, min(math.floor(floor_y), int(self.floor_h) - 1)))
        return f"({col},{row})"

    # ── Occupied-cell drawing ──────────────────────────────────────────────

    def _draw_occupied_cells(
        self,
        canvas: np.ndarray,
        detections: list,
    ) -> None:
        """
        Draw a subtle light-green tint on every 1 m cell that contains at
        least one fused detection.  Drawn on top of zone fills and grid lines
        so the occupied area is clearly visible.
        """
        import math
        px   = int(round(self.px_per_m * self.grid_minor))   # pixels per cell
        seen: set[tuple[int, int]] = set()
        overlay = canvas.copy()

        for fd in detections:
            col = int(max(0, min(math.floor(fd.floor_x), int(self.floor_w) - 1)))
            row = int(max(0, min(math.floor(fd.floor_y), int(self.floor_h) - 1)))
            if (col, row) in seen:
                continue
            seen.add((col, row))

            # Canvas coords of cell corners
            x1 = self._ox + col * px
            y2 = self._oy - row * px
            x2 = x1 + px
            y1 = y2 - px

            # Clamp to floor rectangle
            x1 = max(self._ox,  x1);  x2 = min(self._ox + self._floor_px_w, x2)
            y1 = max(self._STATUS_H + self._MARGIN_T, y1);  y2 = min(self._oy, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # Fused → blue-green tint; single → warm-green tint
            tint = (210, 245, 210) if not fd.is_fused else (230, 245, 215)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), tint, -1)

        cv2.addWeighted(overlay, 0.30, canvas, 0.70, 0, canvas)

    def _draw_cell_labels(
        self,
        canvas: np.ndarray,
        detections: list,
    ) -> None:
        """
        Draw grid-cell coordinate labels.

        Behaviour
        ---------
        _show_cell_labels = False (default)
            Only *occupied* cells show a tiny "(col,row)" label in their
            top-left corner.  This confirms which cell each person is in.

        _show_cell_labels = True  (toggle with 'L')
            Every cell shows its "(col,row)" label centred inside the cell.
            Useful for identifying exact positions during commissioning.
        """
        import math
        px    = int(round(self.px_per_m * self.grid_minor))
        font  = cv2.FONT_HERSHEY_SIMPLEX

        if self._show_cell_labels:
            # ── All cells — centred label ─────────────────────────────────
            if px < 18:
                return   # cells too small to be readable
            sz    = max(0.18, min(0.30, px / 130.0))   # scale with cell size
            col_c = (160, 160, 155)                    # light gray

            n_x = int(round(self.floor_w / self.grid_minor))
            n_y = int(round(self.floor_h / self.grid_minor))
            for ci in range(n_x):
                for ri in range(n_y):
                    cx_px = self._ox + ci * px + px // 2
                    cy_px = self._oy - ri * px - px // 2
                    lbl   = f"{ci},{ri}"
                    (tw, th), _ = cv2.getTextSize(lbl, font, sz, 1)
                    cv2.putText(canvas, lbl,
                                (cx_px - tw // 2, cy_px + th // 2),
                                font, sz, col_c, 1, cv2.LINE_AA)
        else:
            # ── Occupied cells only — corner label ────────────────────────
            sz    = 0.28
            col_c = (80, 100, 80)   # dark green-gray, visible on tinted cell
            seen: set[tuple[int, int]] = set()

            for fd in detections:
                col = int(max(0, min(math.floor(fd.floor_x), int(self.floor_w) - 1)))
                row = int(max(0, min(math.floor(fd.floor_y), int(self.floor_h) - 1)))
                if (col, row) in seen:
                    continue
                seen.add((col, row))

                # Top-left of cell + 2 px padding
                tx = self._ox + col * px + 2
                ty = self._oy - row * px - px + 9
                lbl = f"{col},{row}"
                cv2.putText(canvas, lbl, (tx, ty),
                            font, sz, col_c, 1, cv2.LINE_AA)

    # ── Grid report (JSON output) ──────────────────────────────────────────

    def get_grid_report(
        self,
        fused_detections: list,
        stats: Optional[dict] = None,
        timestamp: Optional[str] = None,
    ) -> dict:
        """
        Build a JSON-serialisable report mapping every person to a grid cell.

        Schema
        ------
        {
          "timestamp":              "2026-03-01T12:34:56",
          "floor_width_m":          30.0,
          "floor_height_m":         20.0,
          "grid_cell_size_m":       1.0,
          "cell_naming_convention": "(col,row) where col=floor(x_m), row=floor(y_m)",
          "total_persons":          12,
          "fused_count":            6,
          "single_count":           6,
          "occupied_cell_count":    12,
          "occupied_cells":         ["(8,3)", "(14,8)", ...],
          "cells_occupied":         { "(8,3)": [1], "(14,8)": [2,3] },
          "persons": [
            {
              "id":                 1,
              "cell_id":            "(8,3)",
              "grid_col":           8,
              "grid_row":           3,
              "floor_x_m":          8.74,
              "floor_y_m":          3.21,
              "confidence":         0.87,
              "is_fused":           true,
              "source_cameras":     ["cam_1","cam_2"],
              "fusion_distance_m":  0.23
            }, ...
          ],
          "by_camera":  { "cam_1": 9, "cam_2": 9 }   (if stats provided)
        }

        Parameters
        ----------
        fused_detections : list[FusedDetection]
        stats            : dict from DetectionFuser.get_stats()  (optional)
        timestamp        : ISO-8601 string; defaults to now
        """
        import math
        from datetime import datetime

        ts = timestamp or datetime.now().isoformat(timespec="seconds")

        persons:        list[dict]             = []
        occupied_cells: set[str]               = set()
        cells_occupied: dict[str, list[int]]   = {}

        for idx, fd in enumerate(fused_detections):
            col = int(max(0, min(math.floor(fd.floor_x), int(self.floor_w) - 1)))
            row = int(max(0, min(math.floor(fd.floor_y), int(self.floor_h) - 1)))
            cell_id = f"({col},{row})"
            occupied_cells.add(cell_id)
            cells_occupied.setdefault(cell_id, []).append(idx + 1)

            entry: dict = {
                "id":               idx + 1,
                "cell_id":          cell_id,
                "grid_col":         col,
                "grid_row":         row,
                "floor_x_m":        round(float(fd.floor_x), 3),
                "floor_y_m":        round(float(fd.floor_y), 3),
                "confidence":       round(float(fd.confidence), 3),
                "is_fused":         bool(fd.is_fused),
                "source_cameras":   list(fd.source_cameras),
                "fusion_distance_m": round(float(getattr(fd, "fusion_distance", 0.0)), 3),
            }
            persons.append(entry)

        # Sort by cell position for deterministic / readable output
        persons.sort(key=lambda p: (p["grid_col"], p["grid_row"], p["id"]))

        total   = len(persons)
        fused_c = sum(1 for p in persons if p["is_fused"])

        report: dict = {
            "timestamp":              ts,
            "floor_width_m":          float(self.floor_w),
            "floor_height_m":         float(self.floor_h),
            "grid_cell_size_m":       float(self.grid_minor),
            "cell_naming_convention": (
                "(col,row)  col=floor(x_m) clipped to 0..floor_w-1, "
                "row=floor(y_m) clipped to 0..floor_h-1"
            ),
            "total_persons":          total,
            "fused_count":            fused_c,
            "single_count":           total - fused_c,
            "occupied_cell_count":    len(occupied_cells),
            "occupied_cells":         sorted(occupied_cells),
            "cells_occupied":         {
                k: v for k, v in sorted(cells_occupied.items())
            },
            "persons":                persons,
        }

        if stats and "by_camera" in stats:
            report["by_camera"] = stats["by_camera"]

        return report

    def _draw_scale_bar(self, canvas: np.ndarray, px_step: int) -> None:
        """
        Draw a "1 m" scale bar in the bottom-right of the floor area.

        Visually confirms the grid cell size at a glance.
        Layout:  "|——— 1 m ———|"  horizontal bar, px_step pixels wide.
        """
        # Position: 10 px from right edge, 14 px above the floor bottom
        bar_x2 = self._ox + self._floor_px_w - 10
        bar_x1 = bar_x2 - px_step
        bar_y  = self._oy - 14

        col   = self._GRID_LABEL
        font  = cv2.FONT_HERSHEY_SIMPLEX
        sz    = 0.32
        thick = 1

        # Horizontal line
        cv2.line(canvas, (bar_x1, bar_y), (bar_x2, bar_y), col, thick, cv2.LINE_AA)

        # End ticks (vertical serifs)
        tick_h = 5
        cv2.line(canvas, (bar_x1, bar_y - tick_h), (bar_x1, bar_y + tick_h),
                 col, thick, cv2.LINE_AA)
        cv2.line(canvas, (bar_x2, bar_y - tick_h), (bar_x2, bar_y + tick_h),
                 col, thick, cv2.LINE_AA)

        # "1 m" label centred above the bar
        lbl = "1 m"
        (tw, th), _ = cv2.getTextSize(lbl, font, sz, thick)
        mid_x = (bar_x1 + bar_x2) // 2
        cv2.putText(canvas, lbl,
                    (mid_x - tw // 2, bar_y - tick_h - 3),
                    font, sz, col, thick, cv2.LINE_AA)

    # ══════════════════════════════════════════════════════════════════════
    #  Low-level drawing utilities
    # ══════════════════════════════════════════════════════════════════════

    def _draw_dashed_polygon(
        self,
        canvas: np.ndarray,
        pts: np.ndarray,
        color: tuple,
        thickness: int = 2,
        dash_px: int = 9,
        gap_px: int = 5,
    ) -> None:
        """Draw a closed dashed polyline through *pts* (array of [x, y])."""
        n = len(pts)
        for i in range(n):
            self._draw_dashed_line(
                canvas,
                tuple(pts[i]),
                tuple(pts[(i + 1) % n]),
                color, thickness, dash_px, gap_px,
            )

    @staticmethod
    def _draw_dashed_line(
        canvas: np.ndarray,
        p1: tuple,
        p2: tuple,
        color: tuple,
        thickness: int = 2,
        dash_px: int = 9,
        gap_px: int = 5,
    ) -> None:
        """Draw a dashed line from *p1* to *p2*."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1:
            return
        ux, uy  = dx / length, dy / length
        d       = 0.0
        drawing = True
        while d < length:
            seg = dash_px if drawing else gap_px
            d2  = min(d + seg, length)
            if drawing:
                x1 = int(round(p1[0] + ux * d))
                y1 = int(round(p1[1] + uy * d))
                x2 = int(round(p1[0] + ux * d2))
                y2 = int(round(p1[1] + uy * d2))
                cv2.line(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
            d       = d2
            drawing = not drawing

    # ══════════════════════════════════════════════════════════════════════
    #  Config helpers
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _coerce_cfg(source) -> dict:
        """Accept a dict, Path, or string path; always return a dict."""
        if isinstance(source, dict):
            return source
        with open(source, "r") as f:
            return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
#  Factory function
# ═══════════════════════════════════════════════════════════════════════════

def load_renderer(
    config_path: str | Path = "config/",
    window_width: int = 1200,
) -> FloorRenderer:
    """
    Convenience factory: load all three JSON configs from *config_path*
    and return a ready ``FloorRenderer``.

    Parameters
    ----------
    config_path  : directory containing floor_config.json, cameras.json,
                   and overlap_zones.json  (default: "config/")
    window_width : total canvas width in pixels  (default: 1200)

    Returns
    -------
    FloorRenderer
    """
    base = Path(config_path)
    return FloorRenderer(
        floor_config   = base / "floor_config.json",
        cameras_config = base / "cameras.json",
        overlap_config = base / "overlap_zones.json",
        window_width   = window_width,
    )
