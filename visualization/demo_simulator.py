"""
demo_simulator.py — Customer-Facing Demo Mode  (python main.py --demo)

Runs a fully animated factory-floor simulation without cameras, RTSP streams,
or a YOLO model.  Every component of the real pipeline is exercised:

    DemoSimulator.step()          — generate synthetic per-camera detections
    DetectionFuser.fuse()         — Hungarian overlap deduplication
    FloorRenderer.render()        — draw the floor map with all layers

Simulation details
------------------
• 12 virtual persons execute a *correlated random walk* (smooth curved paths).
• Speed: 0.05 – 0.25 m/frame (randomised per person).
• Direction changes slowly each frame (Gaussian angular noise ≈ 14 °/frame).
• Walls cause angle-reflection (elastic bounce).
• Confidence jitters ±0.02 per frame, clamped to [0.55, 0.99].
• 3 persons are seeded inside the cam_1/cam_2 overlap zone at startup.
• Per-camera Gaussian noise (σ_cam1=0.15 m, σ_cam2=0.18 m, σ_cam3=0.12 m)
  simulates real homography imperfection:  different cameras give slightly
  different floor readings for the same physical position.

Controls
--------
  Q / Esc   Quit
  Space     Pause / resume
  G         Toggle grid
  C         Toggle camera zones
  O         Toggle overlap zone
  S         Save snapshot
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np

from fusion.fuse import DetectionFuser, FusedDetection
from fusion.overlap import load_overlap_zones
from visualization.floor_renderer import load_renderer

_PROJECT_ROOT = Path(__file__).parent.parent
_OUTPUT_DIR   = _PROJECT_ROOT / "output"
_OUTPUT_DIR.mkdir(exist_ok=True)
_CONFIG_DIR   = _PROJECT_ROOT / "config"
_FLOOR_CFG    = _CONFIG_DIR / "floor_config.json"
_CAMERAS_CFG  = _CONFIG_DIR / "cameras.json"
_OVERLAP_CFG  = _CONFIG_DIR / "overlap_zones.json"

# Per-camera Gaussian noise stddev (metres) — simulates homography imperfection
# Different values model each camera having its own calibration quality
_CAM_NOISE: dict[str, float] = {
    "cam_1": 0.15,
    "cam_2": 0.18,
}
_DEFAULT_NOISE = 0.15


# ═══════════════════════════════════════════════════════════════════════════
#  SimulatedPerson
# ═══════════════════════════════════════════════════════════════════════════

class SimulatedPerson:
    """
    One simulated person executing a correlated random walk.

    Motion model
    ------------
    • Speed (m/frame): assigned at init from Uniform(0.05, 0.25); kept constant.
    • Direction angle (rad): updated each frame by Gaussian noise ≈ ±0.25 rad
      (≈ ±14°) so paths curve smoothly rather than jittering.
    • Wall bounce: when the person would leave the floor boundary their angle is
      reflected (elastic collision).
    • Confidence: jitters ±0.02 per frame (Gaussian σ=0.02), clamped [0.55, 0.99].
    """

    __slots__ = [
        "person_id", "x", "y", "floor_w", "floor_h",
        "speed", "angle", "confidence",
    ]

    def __init__(
        self,
        person_id: int,
        x: float,
        y: float,
        floor_w: float = 50.0,
        floor_h: float = 30.0,
    ) -> None:
        self.person_id  = person_id
        self.x          = float(x)
        self.y          = float(y)
        self.floor_w    = float(floor_w)
        self.floor_h    = float(floor_h)
        self.speed      = random.uniform(0.05, 0.25)          # m/frame
        self.angle      = random.uniform(0.0, 2.0 * math.pi) # radians
        self.confidence = random.uniform(0.72, 0.96)

    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance position by one frame and jitter confidence."""
        # ── Correlated direction change (~±14° per frame) ──────────────
        self.angle += random.gauss(0.0, 0.25)

        nx = self.x + math.cos(self.angle) * self.speed
        ny = self.y + math.sin(self.angle) * self.speed

        # ── Wall bounce (elastic reflection) ──────────────────────────
        if nx < 0.1:
            nx = 0.1
            self.angle = math.pi - self.angle          # reflect off left wall
        elif nx > self.floor_w - 0.1:
            nx = self.floor_w - 0.1
            self.angle = math.pi - self.angle          # reflect off right wall

        if ny < 0.1:
            ny = 0.1
            self.angle = -self.angle                   # reflect off bottom wall
        elif ny > self.floor_h - 0.1:
            ny = self.floor_h - 0.1
            self.angle = -self.angle                   # reflect off top wall

        self.x = nx
        self.y = ny

        # ── Confidence jitter ±0.02 ────────────────────────────────────
        self.confidence = max(0.55, min(0.99,
                                        self.confidence + random.gauss(0.0, 0.02)))

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"SimulatedPerson(id={self.person_id}  "
                f"pos=({self.x:.2f},{self.y:.2f})  "
                f"spd={self.speed:.3f}  conf={self.confidence:.2f})")


# ═══════════════════════════════════════════════════════════════════════════
#  _SimDetection  —  minimal object accepted by DetectionFuser
# ═══════════════════════════════════════════════════════════════════════════

class _SimDetection:
    """
    Lightweight detection object compatible with ``DetectionFuser.fuse()``.

    The fuser inspects ``floor_x``, ``floor_y``, and ``confidence``;
    ``camera_id`` is taken from the dict key but is also stored here for
    debugging.  ``pixel_bbox`` / ``pixel_foot`` are set to ``None`` (not
    needed for floor-only visualisation).
    """

    __slots__ = ["camera_id", "floor_x", "floor_y", "confidence",
                 "person_id", "pixel_bbox", "pixel_foot"]

    def __init__(
        self,
        camera_id:  str,
        floor_x:    float,
        floor_y:    float,
        confidence: float,
        person_id:  int = 0,
    ) -> None:
        self.camera_id  = camera_id
        self.floor_x    = float(floor_x)
        self.floor_y    = float(floor_y)
        self.confidence = float(confidence)
        self.person_id  = person_id
        self.pixel_bbox = None
        self.pixel_foot = None

    def __repr__(self) -> str:
        return (f"_SimDetection(cam={self.camera_id} "
                f"x={self.floor_x:.2f} y={self.floor_y:.2f} "
                f"conf={self.confidence:.2f} pid={self.person_id})")


# ═══════════════════════════════════════════════════════════════════════════
#  DemoSimulator
# ═══════════════════════════════════════════════════════════════════════════

class DemoSimulator:
    """
    Manages 12 ``SimulatedPerson`` agents and produces per-camera detection
    lists on each call to ``step()``.

    2-Camera layout  (floor 30 × 20 m)
    ------------------------------------
    cam_1 covers x = 0–22 m   (left + centre-left)
    cam_2 covers x = 8–30 m   (centre-right + right)
    Overlap zone: x = 8–22 m  (14 m wide strip — both cameras see it)

    ┌────────────────────────┬───────┬──────────────────────────────┐
    │ Zone                   │ Count │ Approx. start positions (m)  │
    ├────────────────────────┼───────┼──────────────────────────────┤
    │ cam_1 only  (x < 8)    │   3   │ left strip                   │
    │ Overlap cam_1 ∩ cam_2  │   6   │ x ∈ [8,22] — large centre    │
    │ cam_2 only  (x > 22)   │   3   │ right strip                  │
    └────────────────────────┴───────┴──────────────────────────────┘

    step() → dict[str, list[_SimDetection]]
        Each dict key is a camera ID.  The values contain noisy observations
        for persons visible in that camera's coverage polygon.  Different
        noise seeds per camera simulate distinct homography errors.
    """

    NUM_PERSONS = 12

    # Deterministic start positions: (x, y) in metres
    # Floor: 30 × 20 m   cam_1: x 0–22   cam_2: x 8–30   overlap: x 8–22
    _STARTS: list[tuple[float, float]] = [
        # cam_1 only  (x < 8 m)
        ( 2.0,  5.0),
        ( 4.5, 15.0),
        ( 6.5,  9.0),
        # Overlap zone  (x ∈ [8,22])  ← 6 persons seen by BOTH cameras
        ( 9.5,  4.0),
        (12.0, 17.0),
        (14.5,  8.5),
        (17.0,  3.0),
        (19.5, 14.0),
        (21.0,  9.5),
        # cam_2 only  (x > 22 m)
        (24.0,  6.0),
        (27.0, 14.0),
        (29.0,  2.5),
    ]

    def __init__(self, config_path: str | Path = None) -> None:
        cfg_dir = Path(config_path) if config_path else _CONFIG_DIR

        # ── Floor dimensions ──────────────────────────────────────────────
        with open(cfg_dir / "floor_config.json") as f:
            fc = json.load(f)
        self.floor_w = float(fc["floor_width_m"])
        self.floor_h = float(fc["floor_height_m"])

        # ── Camera coverage polygons ──────────────────────────────────────
        with open(cfg_dir / "cameras.json") as f:
            cam_cfg = json.load(f)
        self._cam_polygons: dict[str, list] = {
            c["id"]: c.get("floor_coverage_polygon", [])
            for c in cam_cfg["cameras"]
        }
        self._cam_ids: list[str] = [c["id"] for c in cam_cfg["cameras"]]

        # ── Per-camera noise (metres) ─────────────────────────────────────
        self._cam_noise: dict[str, float] = {
            cam_id: _CAM_NOISE.get(cam_id, _DEFAULT_NOISE)
            for cam_id in self._cam_ids
        }

        # ── Seed persons ──────────────────────────────────────────────────
        assert len(self._STARTS) == self.NUM_PERSONS, \
            f"_STARTS has {len(self._STARTS)} entries, expected {self.NUM_PERSONS}"

        random.seed(42)  # deterministic startup; randomness diverges after frame 1
        self.persons: list[SimulatedPerson] = []
        for i, (x, y) in enumerate(self._STARTS):
            p = SimulatedPerson(
                person_id = i + 1,
                x = x, y = y,
                floor_w = self.floor_w,
                floor_h = self.floor_h,
            )
            # Spread initial angles evenly so persons don't all move in lockstep
            p.angle = i * (2.0 * math.pi / self.NUM_PERSONS)
            self.persons.append(p)

    # ------------------------------------------------------------------

    def step(self) -> dict[str, list]:
        """
        Advance simulation by one frame and generate per-camera detections.

        Algorithm per person
        --------------------
        1. Move one step  (``SimulatedPerson.step()``).
        2. Check which camera coverage polygons contain this person's position.
        3. For each visible camera, emit one ``_SimDetection`` with independent
           Gaussian noise (σ varies per camera) applied to floor_x / floor_y.
           This models real homography imperfection: the same physical person
           appears at a slightly different floor coordinate in each camera.

        Returns
        -------
        dict[str, list[_SimDetection]]
            ``{"cam_1": [...], "cam_2": [...], "cam_3": [...]}``
            Ready to pass directly to ``DetectionFuser.fuse()``.
        """
        result: dict[str, list] = {cam_id: [] for cam_id in self._cam_ids}

        for person in self.persons:
            person.step()

            for cam_id in self._cam_ids:
                poly = self._cam_polygons.get(cam_id, [])
                if not poly:
                    continue
                if not _point_in_poly(person.x, person.y, poly):
                    continue

                # Per-camera independent noise
                sigma   = self._cam_noise[cam_id]
                noisy_x = person.x + random.gauss(0.0, sigma)
                noisy_y = person.y + random.gauss(0.0, sigma)

                result[cam_id].append(_SimDetection(
                    camera_id  = cam_id,
                    floor_x    = noisy_x,
                    floor_y    = noisy_y,
                    confidence = person.confidence,
                    person_id  = person.person_id,
                ))

        return result

    # ------------------------------------------------------------------

    def person_count(self) -> int:
        """Return total number of simulated persons."""
        return len(self.persons)

    def __repr__(self) -> str:
        return f"DemoSimulator(persons={self.NUM_PERSONS}  floor={self.floor_w}×{self.floor_h} m)"


# ═══════════════════════════════════════════════════════════════════════════
#  Geometry helper
# ═══════════════════════════════════════════════════════════════════════════

def _point_in_poly(x: float, y: float, poly: list) -> bool:
    """Ray-casting point-in-polygon test (floor metres)."""
    n      = len(poly)
    inside = False
    j      = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


# ═══════════════════════════════════════════════════════════════════════════
#  run_demo()  —  called by:  python main.py --demo
# ═══════════════════════════════════════════════════════════════════════════

_DEMO_BANNER = (
    "DEMO MODE — 12 simulated persons  |  "
    "[Q] Quit  [G] Grid  [C] Cameras  [O] Overlap  "
    "[L] Cell Labels  [J] JSON Report  [S] Snapshot"
)


def run_demo(
    debug:         bool  = False,
    save_snapshot: bool  = False,
    target_fps:    float = 12.0,
    window_width:  int   = 1200,
) -> None:
    """
    Main demo animation loop.

    Pipeline per frame::

        DemoSimulator.step()   →   DetectionFuser.fuse()   →   FloorRenderer.render()

    Parameters
    ----------
    debug         : reserved (no-op in this version)
    save_snapshot : if True, save the last rendered frame to disk on exit
    target_fps    : animation speed, default 12 FPS
    window_width  : canvas width in pixels, default 1200
    """
    print("\n" + "=" * 68)
    print("  " + _DEMO_BANNER)
    print("=" * 68)
    print(f"\n  Persons   : {DemoSimulator.NUM_PERSONS}")
    print(f"  FPS target: {target_fps:.0f}")
    print(f"  Noise     : cam_1={_CAM_NOISE['cam_1']} m  cam_2={_CAM_NOISE['cam_2']} m")
    print(f"\n  Space = pause / resume\n")

    # ── Init pipeline ─────────────────────────────────────────────────────
    simulator  = DemoSimulator()
    zones      = load_overlap_zones(_OVERLAP_CFG)
    fuser      = DetectionFuser(zones)
    renderer   = load_renderer(str(_CONFIG_DIR), window_width=window_width)

    WIN_NAME      = "Factory Floor Monitor — DEMO"
    TARGET_MS     = int(1000 / target_fps)
    frame_count   = 0
    last_canvas   = None
    last_fused    = []       # kept for JSON report
    last_stats    = {}       # kept for JSON report
    paused        = False
    snapshot_idx  = 0
    json_idx      = 0

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, window_width, renderer.canvas_h)

    # ── Main loop ─────────────────────────────────────────────────────────
    try:
        while True:
            t0 = time.perf_counter()

            if not paused:
                # ── 1. Step simulation ─────────────────────────────────────
                dets_by_cam = simulator.step()

                # ── 2. Fuse (Hungarian matching in overlap zone) ───────────
                fused       = fuser.fuse(dets_by_cam)
                stats       = fuser.get_stats(fused)

                # ── 3. Render ──────────────────────────────────────────────
                canvas      = renderer.render(fused, stats)
                last_canvas = canvas
                last_fused  = fused
                last_stats  = stats
                frame_count += 1

            else:
                # Show last frame with paused overlay
                canvas = (last_canvas.copy()
                          if last_canvas is not None
                          else renderer.render([], None))
                _draw_paused_overlay(canvas, renderer)

            cv2.imshow(WIN_NAME, canvas)

            # ── Frame-rate control ─────────────────────────────────────────
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            wait_ms    = max(1, TARGET_MS - elapsed_ms)
            key        = cv2.waitKey(wait_ms) & 0xFF

            # ── Keyboard ──────────────────────────────────────────────────
            if key in (ord("q"), ord("Q"), 27):               # Q / Esc — quit
                print("\n[Demo] Quit by user.")
                break

            elif key == ord(" "):                              # Space — pause
                paused = not paused
                print(f"[Demo] {'Paused' if paused else 'Resumed'}")

            elif key in (ord("g"), ord("G")):                  # G — grid
                state = renderer.toggle_grid()
                print(f"[Demo] Grid: {'ON' if state else 'OFF'}")

            elif key in (ord("c"), ord("C")):                  # C — cameras
                state = renderer.toggle_cameras()
                print(f"[Demo] Camera zones: {'ON' if state else 'OFF'}")

            elif key in (ord("o"), ord("O")):                  # O — overlap
                state = renderer.toggle_overlap()
                print(f"[Demo] Overlap zone: {'ON' if state else 'OFF'}")

            elif key in (ord("l"), ord("L")):                  # L — cell labels
                state = renderer.toggle_cell_labels()
                print(f"[Demo] Cell labels: {'ON' if state else 'OFF'}")

            elif key in (ord("j"), ord("J")):                  # J — JSON report
                if last_fused is not None:
                    import json as _json
                    from datetime import datetime
                    json_idx += 1
                    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = _OUTPUT_DIR / f"grid_report_{ts}_{json_idx:03d}.json"
                    report = renderer.get_grid_report(last_fused, last_stats)
                    with open(path, "w") as _fj:
                        _json.dump(report, _fj, indent=2)
                    print(f"[Demo] JSON report saved: output/{path.name}")
                    print(f"       Persons: {report['total_persons']}  "
                          f"Fused: {report['fused_count']}  "
                          f"Occupied cells: {report['occupied_cell_count']}")

            elif key in (ord("s"), ord("S")):                  # S — snapshot
                if last_canvas is not None:
                    snapshot_idx += 1
                    from datetime import datetime
                    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = _OUTPUT_DIR / f"demo_snapshot_{ts}_{snapshot_idx:03d}.png"
                    ok   = renderer.save_snapshot(last_canvas, path)
                    if ok:
                        print(f"[Demo] Snapshot saved: output/{path.name}")

    except KeyboardInterrupt:
        print("\n[Demo] Interrupted.")

    finally:
        import json as _json
        from datetime import datetime as _dt
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")

        # ── Exit snapshot ──────────────────────────────────────────────────
        if save_snapshot and last_canvas is not None:
            path = _OUTPUT_DIR / f"demo_snapshot_{ts}_final.png"
            renderer.save_snapshot(last_canvas, path)
            print(f"[Demo] Exit snapshot saved: output/{path.name}")

        # ── Auto-save final JSON report ────────────────────────────────────
        if last_fused is not None:
            try:
                report = renderer.get_grid_report(last_fused, last_stats)
                path   = _OUTPUT_DIR / f"grid_report_{ts}_final.json"
                with open(path, "w") as _fj:
                    _json.dump(report, _fj, indent=2)
                print(f"\n[Demo] Auto-saved final report → output/{path.name}")
                print(f"  Persons      : {report['total_persons']}")
                print(f"  Fused        : {report['fused_count']}")
                print(f"  Single       : {report['single_count']}")
                print(f"  Cells in use : {report['occupied_cell_count']}  "
                      f"-> {report['occupied_cells']}")
                print("  Cell details:")
                for p in report["persons"]:
                    fuse_tag = " [FUSED]" if p["is_fused"] else ""
                    print(f"    Person {p['id']:2d} -> cell {p['cell_id']}  "
                          f"({p['floor_x_m']:.2f}, {p['floor_y_m']:.2f}) m  "
                          f"conf={p['confidence']:.2f}{fuse_tag}")
            except Exception as _e:
                print(f"[Demo] Could not save report: {_e}")

        cv2.destroyAllWindows()
        print(f"\n[Demo] Finished — {frame_count} frames rendered.")


# ─────────────────────────────────────────────────────────────────────────
#  Drawing helpers
# ─────────────────────────────────────────────────────────────────────────

def _draw_paused_overlay(canvas: np.ndarray, renderer) -> None:
    """Render a small PAUSED notification on top of the floor map."""
    # Anchor to floor top-left corner
    x0 = renderer._ox + 6
    y0 = renderer._oy - renderer._floor_px_h + 6
    x1, y1 = x0 + 310, y0 + 32
    # Semi-opaque dark background
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (40, 40, 38), -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (80, 80, 78), 1)
    cv2.putText(
        canvas, "PAUSED  —  press Space to resume",
        (x0 + 8, y0 + 21),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 215, 255), 1, cv2.LINE_AA,
    )
