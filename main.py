"""
main.py — Factory Floor Visualisation — Production Entry Point

Complete CLI
------------
# Lens calibration (Step 2 — preserved):
  python main.py --intrinsic cam_1
  python main.py --intrinsic cam_1 --source video.mp4
  python main.py --intrinsic cam_1 --method circles
  python main.py --intrinsic cam_1 --method circles --asymmetric
  python main.py --intrinsic cam_1 --method markers --grid-cols 5 --grid-rows 4
  python main.py --intrinsic cam_1 --method lines

# Floor-point homography calibration (Step 3):
  python main.py --calibrate cam_1
  python main.py --calibrate cam_1 --source rtsp://...

# Development phase tests:
  python main.py --phase 1                       # validate all configs
  python main.py --phase 2                       # homography quality report
  python main.py --phase 3                       # YOLO detection — 1 frame/camera
  python main.py --phase 3 --source ./videos     # from local video files
  python main.py --phase 4                       # full pipeline — 1 frame
  python main.py --phase 4 --source ./videos

# Production mode:
  python main.py --run                           # live multi-camera pipeline
  python main.py --run --debug                   # + per-camera feed panel
  python main.py --run --source ./videos         # video files instead of RTSP
  python main.py --run --source ./videos --debug

# Demo mode (no cameras required):
  python main.py --demo
  python main.py --demo --debug
  python main.py --demo --snapshot               # save final frame on exit
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path — allows running from any CWD
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)   # create on first run, never fails

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------------
CONFIG_DIR  = PROJECT_ROOT / "config"
FLOOR_CFG   = CONFIG_DIR / "floor_config.json"
CAMERAS_CFG = CONFIG_DIR / "cameras.json"
OVERLAP_CFG = CONFIG_DIR / "overlap_zones.json"
EDGES_CFG   = CONFIG_DIR / "edges.json"
FUSION_CFG  = CONFIG_DIR / "fusion_config.json"

# ---------------------------------------------------------------------------
# Qt5-safe display helper
# ---------------------------------------------------------------------------
_MAX_DISP_W = 1400
_MAX_DISP_H =  900

def _cv_show(title: str, img) -> None:
    """
    Show an OpenCV image reliably on Qt5.

    Three Qt5 pitfalls fixed here:
      1. cv2.namedWindow must be called before cv2.imshow (or the window
         is created at a default Qt size and may appear as a tiny square).
      2. Window names with non-ASCII characters (e.g. em-dash) can silently
         prevent the native window handle from being registered, causing
         setMouseCallback / further imshow calls to crash.
      3. The display image is scaled down to fit the screen so WINDOW_AUTOSIZE
         opens at the correct size immediately — no resizeWindow timing issues.
    """
    import cv2, numpy as np

    # ── ASCII-safe title ──────────────────────────────────────────────────
    safe_title = title.encode("ascii", errors="replace").decode("ascii")

    # ── Scale image to fit screen ─────────────────────────────────────────
    h, w = img.shape[:2]
    scale = min(_MAX_DISP_W / max(w, 1), _MAX_DISP_H / max(h, 1), 1.0)
    if scale < 1.0:
        disp = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_AREA)
    else:
        disp = img

    # ── Open window then show ─────────────────────────────────────────────
    cv2.namedWindow(safe_title, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(safe_title, disp)


# ═══════════════════════════════════════════════════════════════════════════
#  Auto-configure — floor size + overlap zones from coverage polygons
# ═══════════════════════════════════════════════════════════════════════════

def auto_configure(silent: bool = False) -> bool:
    """
    Auto-compute floor_config.json dimensions and overlap_zones.json
    directly from the floor_coverage_polygon values in cameras.json.

    Floor size
    ----------
    Bounding box of the *union* of all coverage polygons, padded by 1 m on
    each side and rounded up to the nearest 0.5 m.  Always starts at (0, 0).

    Overlap zones
    -------------
    For every pair of cameras whose coverage polygons *intersect* (area > 0.5 m²),
    an overlap zone is created from the Shapely intersection polygon.
    Existing overlap-zone thresholds are preserved if a zone with the same id
    already exists.

    Returns True if at least one coverage polygon was found.
    """
    import math
    try:
        from shapely.geometry import Polygon
    except ImportError:
        print("  ✗  shapely is required.  pip install shapely")
        return False

    with open(CAMERAS_CFG) as f:
        cameras_cfg = json.load(f)

    # ── Build per-camera Shapely polygons ─────────────────────────────────
    cam_polygons: dict[str, "Polygon"] = {}
    for cam in cameras_cfg["cameras"]:
        pts = cam.get("floor_coverage_polygon", [])
        if len(pts) >= 3:
            poly = Polygon(pts)
            if poly.is_valid and not poly.is_empty:
                cam_polygons[cam["id"]] = poly

    if not cam_polygons:
        if not silent:
            print("  ✗  No cameras have a floor_coverage_polygon yet.")
            print("     Run: python main.py --coverage cam_1 --source <image/video>")
        return False

    # ── Floor dimensions: full bounding box of union, supporting negatives ─
    all_coords: list[tuple[float, float]] = []
    for poly in cam_polygons.values():
        all_coords.extend(poly.exterior.coords)

    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    PAD   = 1.0                              # 1 m padding on all sides

    x_min = min(xs) - PAD
    y_min = min(ys) - PAD
    x_max = max(xs) + PAD
    y_max = max(ys) + PAD

    # Snap origin to nearest 0.5 m floor, snap dimensions up to nearest 0.5 m
    origin_x = math.floor(x_min * 2) / 2   # round down (more negative)
    origin_y = math.floor(y_min * 2) / 2

    floor_w = max(5.0, math.ceil((x_max - origin_x) * 2) / 2)
    floor_h = max(5.0, math.ceil((y_max - origin_y) * 2) / 2)

    # Read & update floor_config.json (preserve other keys)
    with open(FLOOR_CFG) as f:
        floor_cfg = json.load(f)
    old_w = floor_cfg.get("floor_width_m")
    old_h = floor_cfg.get("floor_height_m")
    floor_cfg["floor_width_m"]   = floor_w
    floor_cfg["floor_height_m"]  = floor_h
    floor_cfg["floor_origin_x_m"] = origin_x   # may be negative
    floor_cfg["floor_origin_y_m"] = origin_y   # may be negative
    with open(FLOOR_CFG, "w") as f:
        json.dump(floor_cfg, f, indent=2)

    # ── Pairwise overlap zones from polygon intersection ──────────────────
    # Load existing zones to preserve their thresholds
    existing: dict[str, dict] = {}
    if OVERLAP_CFG.exists():
        with open(OVERLAP_CFG) as f:
            _oc = json.load(f)
        for z in _oc.get("overlap_zones", []):
            existing[z["id"]] = z

    cam_ids = sorted(cam_polygons.keys())
    new_zones: list[dict] = []

    for i in range(len(cam_ids)):
        for j in range(i + 1, len(cam_ids)):
            id_a, id_b  = cam_ids[i], cam_ids[j]
            intersection = cam_polygons[id_a].intersection(cam_polygons[id_b])

            if intersection.is_empty or intersection.area < 0.5:
                continue   # not enough overlap to matter

            # Extract polygon coords (convex hull for simplicity)
            hull   = intersection.convex_hull
            # Use intersection bounds to compute conservative distance threshold (30% of shortest side)
            minx, miny, maxx, maxy = intersection.bounds
            z_w = maxx - minx
            z_h = maxy - miny
            auto_thresh = round(min(z_w, z_h) * 0.3, 2)
            
            coords = [
                [round(float(x), 2), round(float(y), 2)]
                for x, y in list(hull.exterior.coords)[:-1]   # drop closing repeat
            ]

            zone_id = f"overlap_{id_a}_{id_b}"
            # Preserve existing threshold settings if they exist, else use auto_thresh
            base = existing.get(zone_id, {})
            new_zones.append({
                "id":                   zone_id,
                "cameras":              [id_a, id_b],
                "floor_polygon":        coords,
                "distance_threshold_m": base.get("distance_threshold_m", auto_thresh),
                "buffer_margin_m":      base.get("buffer_margin_m", 0.5),
                "fusion_strategy":      base.get("fusion_strategy", "weighted_average"),
            })

    with open(OVERLAP_CFG, "w") as f:
        json.dump({"overlap_zones": new_zones}, f, indent=2)

    if not silent:
        print(f"\n  ✓  floor_config.json updated")
        if old_w != floor_w or old_h != floor_h:
            print(f"     Floor: {old_w} m × {old_h} m  →  {floor_w} m × {floor_h} m")
        else:
            print(f"     Floor: {floor_w} m × {floor_h} m  (unchanged)")

        print(f"\n  ✓  overlap_zones.json updated  ({len(new_zones)} zone(s))")
        for z in new_zones:
            area = Polygon(z["floor_polygon"]).area
            print(f"     [{z['id']}]  {z['cameras']}  "
                  f"area={area:.1f} m²  polygon={z['floor_polygon']}")
        if not new_zones:
            print("     (no camera pairs with overlapping coverage)")

    # ── Write default fusion_config.json if missing ──────────────────────
    if not FUSION_CFG.exists():
        default_fusion = {
            "timestamp_tolerance_s": 1.0,
            "default_distance_threshold_m": 1.5,
            "output_dir": "output/",
            "csv_format": {
                "columns": [
                    "timestamp", "track_id", "class_name", "edge_id",
                    "direction", "crossing_x", "crossing_y", "camera_id"
                ],
                "timestamp_format": "%Y-%m-%d %H:%M:%S",
                "float_precision": 2
            }
        }
        with open(FUSION_CFG, "w") as f:
            json.dump(default_fusion, f, indent=4)
        if not silent:
            print(f"\n  ✓  fusion_config.json created with defaults")

    return True


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 1 — Config loading & summary
# ═══════════════════════════════════════════════════════════════════════════

def phase_1() -> None:
    """Load all three config files and print a human-readable summary."""
    print("\n" + "=" * 60)
    print("  PHASE 1 — Configuration Loading")
    print("=" * 60)

    with open(FLOOR_CFG) as f:
        floor_cfg = json.load(f)

    print(f"\n[floor_config.json]")
    print(f"  Floor dimensions     : {floor_cfg['floor_width_m']} m × {floor_cfg['floor_height_m']} m")
    print(f"  Grid (minor / major) : {floor_cfg['grid_cell_size_m']} m / {floor_cfg['major_grid_every_m']} m")
    print(f"  Origin               : {floor_cfg['origin']}")
    print(f"  Unit                 : {floor_cfg['unit']}")

    with open(CAMERAS_CFG) as f:
        cameras_cfg = json.load(f)

    print(f"\n[cameras.json]  ({len(cameras_cfg['cameras'])} cameras)")
    for cam in cameras_cfg["cameras"]:
        calibrated = cam["intrinsics"].get("calibrated", False)
        n_pts  = len(cam["calibration_points"]["image_points"])
        has_H  = "homography_matrix" in cam
        has_ocr = "ocr_region" in cam
        poly   = cam.get("floor_coverage_polygon", [])
        poly_str = f"{len(poly)}-point polygon" if poly else "not set"
        parts  = []
        if calibrated:
            rms = cam["intrinsics"].get("rms_px", "?")
            parts.append(f"intrinsics ✓ (RMS {rms} px)")
        else:
            parts.append("intrinsics ✗")
        parts.append("homography ✓" if has_H else f"homography ✗ ({n_pts} cal pts)")
        parts.append("OCR ✓" if has_ocr else "OCR ✗")
        
        print(f"  [{cam['id']}] {cam['name']}")
        print(f"         source   : {cam['source']}")
        print(f"         coverage : {poly_str}")
        print(f"         status   : {'  |  '.join(parts)}")

    with open(OVERLAP_CFG) as f:
        overlap_cfg = json.load(f)

    print(f"\n[overlap_zones.json]  ({len(overlap_cfg['overlap_zones'])} zones)")
    for zone in overlap_cfg["overlap_zones"]:
        cams = " ↔ ".join(zone["cameras"])
        print(f"  [{zone['id']}]  cameras: {cams}")
        print(f"         polygon         : {zone['floor_polygon']}")
        print(f"         distance thresh : {zone['distance_threshold_m']} m")
        print(f"         buffer margin   : {zone['buffer_margin_m']} m")
        print(f"         fusion strategy : {zone['fusion_strategy']}")

    print("\n✓ All configs loaded successfully.\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 2 — Homography quality report
# ═══════════════════════════════════════════════════════════════════════════

def phase_2() -> None:
    """Load homographies for all cameras and print a detailed quality report."""
    print("\n" + "=" * 60)
    print("  PHASE 2 — Homography Quality Report")
    print("=" * 60)
    print(f"  Config : {CONFIG_DIR}\n")

    from calibration.homography import load_all_homographies

    mappers        = load_all_homographies(CONFIG_DIR)
    any_calibrated = False
    all_valid      = True

    for cam_id, mapper in mappers.items():
        print(f"  ┌─ [{cam_id}] {'─'*44}")

        lc = mapper.lens_corrector
        if lc.is_calibrated:
            lc_rms = lc.get_reprojection_error()
            print(f"  │  Lens calibration : ✓  RMS={lc_rms:.4f} px")
        else:
            print(f"  │  Lens calibration : ✗  (run: python main.py --intrinsic {cam_id})")

        if not mapper.is_calibrated:
            all_valid = False
            with open(CAMERAS_CFG) as f:
                _cfg = json.load(f)
            _entry = next((c for c in _cfg["cameras"] if c["id"] == cam_id), None)
            pts_in_cfg = len(_entry["calibration_points"]["image_points"]) if _entry else 0

            print(f"  │  Homography       : ✗  ({pts_in_cfg} calibration pts in config)")
            if pts_in_cfg == 0:
                print(f"  │  → Run: python main.py --calibrate {cam_id}")
            elif pts_in_cfg < 4:
                print(f"  │  → Need ≥4 point pairs (have {pts_in_cfg})")
                print(f"  │  → Run: python main.py --calibrate {cam_id}")
            else:
                print(f"  │  → {pts_in_cfg} pts found but homography failed to compute")
                print(f"  │  → Check floor_points in cameras.json")
            print(f"  └{'─'*50}")
            continue

        any_calibrated = True
        report = mapper.get_reprojection_error()

        if report:
            mean_m  = report["mean_error_m"]
            max_m   = report["max_error_m"]
            n_in    = report["inlier_count"]
            n_tot   = report["total_points"]
            quality = report["quality"]
            method  = report["method"]
            worst   = report["worst_point_idx"]
            per_pt  = report["per_point_errors_m"]
            lc_used = report["lens_corrected"]

            q_icon = {"EXCELLENT": "✓✓", "GOOD": "✓", "ACCEPTABLE": "⚠", "POOR": "✗"}.get(quality, "?")
            print(f"  │  Homography       : ✓")
            print(f"  │  Method           : {method}")
            print(f"  │  Lens corrected   : {'YES ✓' if lc_used else 'NO  (run --intrinsic for better accuracy)'}")
            print(f"  │  Inliers          : {n_in}/{n_tot}")
            print(f"  │  Mean error       : {mean_m*100:.2f} cm  →  {q_icon} {quality}")
            print(f"  │  Max error        : {max_m*100:.2f} cm  (point #{worst+1})")
            print(f"  │  Per-point (cm)   : {[round(e*100, 2) for e in per_pt]}")
            if quality == "POOR":
                all_valid = False
                print(f"  │  ⚠  Add more spread pts / run --intrinsic / verify floor_points")
            elif not lc_used:
                print(f"  │  ℹ  Run --intrinsic {cam_id} to reduce edge errors by 5–20×")
        else:
            print(f"  │  Homography       : ✓  (error report unavailable)")

        if mapper.floor_points is not None and len(mapper.floor_points) > 0:
            fp = mapper.floor_points
            spread_x = fp[:, 0].max() - fp[:, 0].min()
            spread_y = fp[:, 1].max() - fp[:, 1].min()
            warn = "  ⚠ small spread" if (spread_x < 3 or spread_y < 3) else ""
            print(f"  │  Floor coverage   : "
                  f"X={fp[:,0].min():.1f}–{fp[:,0].max():.1f} m  "
                  f"Y={fp[:,1].min():.1f}–{fp[:,1].max():.1f} m{warn}")

        print(f"  └{'─'*50}")

    print()
    if not any_calibrated:
        print("  ✗  No cameras calibrated yet.")
        print("  Step 1: python main.py --intrinsic <cam_id>  (optional, improves accuracy)")
        print("  Step 2: python main.py --calibrate <cam_id>  (required per camera)")
    elif not all_valid:
        print("  ⚠  Some cameras have poor accuracy or are uncalibrated.")
    else:
        print("  ✓  Phase 2 complete — all cameras calibrated.\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 3 — YOLO detection test (one frame per camera)
# ═══════════════════════════════════════════════════════════════════════════

def phase_3(source_dir: str = None, track_point: str = "bottom", model_name: str = "yolov8m.pt") -> None:
    """
    Undistort → detect → floor-project one frame per camera.
    Prints floor coordinates and opens annotated OpenCV windows.
    """
    print("\n" + "=" * 60)
    print("  PHASE 3 — YOLO Person Detection Test (1 frame/camera)")
    print("=" * 60)

    import cv2
    from detection.detector import PersonDetector, CameraProcessor, draw_detections, FloorDetection
    from calibration.homography import load_all_homographies

    mappers  = load_all_homographies(CONFIG_DIR)
    detector = PersonDetector(model_name=model_name, confidence=0.45)

    with open(CAMERAS_CFG) as f:
        cameras_cfg = json.load(f)

    windows_opened: list[str] = []
    total_floor_dets = 0

    for cam in cameras_cfg["cameras"]:
        cam_id  = cam["id"]
        cam_cfg = dict(cam)

        if source_dir:
            source_path = Path(source_dir)
            candidates  = list(source_path.glob(f"*{cam_id}*"))
            if candidates:
                cam_cfg["source"] = str(candidates[0])
                print(f"\n  [{cam_id}] Using file: {cam_cfg['source']}")
            else:
                print(f"\n  [{cam_id}] No matching file in {source_dir} — skipping")
                continue

        mapper = mappers.get(cam_id)
        if mapper is None:
            print(f"\n  [{cam_id}] ✗  No HomographyMapper — run: python main.py --calibrate {cam_id}")
            continue

        _cam_tp = cam_cfg.get("track_point", track_point)  # per-camera override
        processor = CameraProcessor(camera_config=cam_cfg, homography=mapper, detector=detector, track_point=_cam_tp)

        if not processor.is_active():
            print(f"\n  [{cam_id}] ✗  Cannot open source: {cam_cfg['source']}")
            continue

        floor_dets, frame = processor.process_frame()
        processor.release()

        if frame is None:
            print(f"\n  [{cam_id}] ✗  No frame received")
            continue

        lens_status = (
            f"lens ✓  (RMS={mapper.lens_corrector.get_reprojection_error():.3f} px)"
            if mapper.lens_corrector.is_calibrated
            else "lens ✗  (run --intrinsic)"
        )
        homo_status = "homography ✓" if mapper.is_calibrated else "homography ✗  (run --calibrate)"

        print(f"\n  ┌─ [{cam_id}] ──── {lens_status}  |  {homo_status}")

        if floor_dets:
            total_floor_dets += len(floor_dets)
            print(f"  │  Persons detected : {len(floor_dets)}")
            for fd in floor_dets:
                fx, fy = fd.pixel_foot
                print(
                    f"  │    conf={fd.confidence:.2f}  "
                    f"foot_px=({fx:.0f},{fy:.0f})  "
                    f"floor=({fd.floor_x:.2f},{fd.floor_y:.2f}) m"
                )
        else:
            raw_dets = getattr(processor, '_last_detections', [])
            if raw_dets:
                print(f"  │  Persons (px only) : {len(raw_dets)}  (no floor mapping)")
                for d in raw_dets:
                    fx, fy = d.foot_point
                    print(f"  │    conf={d.confidence:.2f}  foot_px=({fx:.0f},{fy:.0f})")
            else:
                print("  │  No persons detected in this frame.")

        print("  └" + "─" * 50)

        if floor_dets:
            annotated = draw_detections(frame, floor_dets)
        else:
            annotated = draw_detections(frame, getattr(processor, '_last_detections', []))

        h, w = annotated.shape[:2]
        for i, line in enumerate([cam_id, lens_status, homo_status,
                                   f"persons: {len(floor_dets) or len(getattr(processor, '_last_detections', []))}"]):
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            y = 20 + i * (th + 8)
            cv2.rectangle(annotated, (w - tw - 12, y - th - 3), (w - 4, y + 3), (30, 30, 30), -1)
            cv2.putText(annotated, line, (w - tw - 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1, cv2.LINE_AA)

        win = f"Phase 3: {cam_id}"
        _cv_show(win, annotated)
        windows_opened.append(win)

    print(f"\n  Total floor-mapped detections : {total_floor_dets}")
    if windows_opened:
        print(f"  {len(windows_opened)} window(s) open — press any key to close …\n")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("  No windows to show — check sources or run --calibrate.\n")
    print("✓ Phase 3 complete.\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 4 — Full pipeline test (one frame, fusion, floor render)
# ═══════════════════════════════════════════════════════════════════════════

def phase_4(source_dir: str = None, track_point: str = "bottom", model_name: str = "yolov8m.pt") -> None:
    """
    Full pipeline on a single set of frames:
    capture → undistort → detect → map → fuse → render.
    """
    print("\n" + "=" * 60)
    print("  PHASE 4 — Full Pipeline Test (1 frame)")
    print("=" * 60)

    import cv2
    from detection.detector import PersonDetector, CameraProcessor
    from calibration.homography import load_all_homographies
    from fusion.fuse import DetectionFuser
    from fusion.overlap import load_overlap_zones
    from visualization.floor_renderer import FloorRenderer

    # ── Setup ─────────────────────────────────────────────────────────────
    with open(CAMERAS_CFG) as f:
        cameras_cfg = json.load(f)

    mappers  = load_all_homographies(CONFIG_DIR)
    detector = PersonDetector(model_name=model_name, confidence=0.45)
    zones    = load_overlap_zones(OVERLAP_CFG)
    fuser    = DetectionFuser(zones)
    renderer = FloorRenderer(FLOOR_CFG, CAMERAS_CFG, OVERLAP_CFG)

    # ── One frame per camera ──────────────────────────────────────────────
    processors: list[tuple[str, CameraProcessor]] = []
    for cam in cameras_cfg["cameras"]:
        cam_id  = cam["id"]
        cam_cfg = dict(cam)

        if source_dir:
            source_path = Path(source_dir)
            candidates  = list(source_path.glob(f"*{cam_id}*"))
            if candidates:
                cam_cfg["source"] = str(candidates[0])
            else:
                print(f"  [{cam_id}] No file in {source_dir} — skipping")
                continue

        mapper = mappers.get(cam_id)
        if mapper is None:
            print(f"  [{cam_id}] ✗  Not calibrated — skipping")
            continue

        _cam_tp = cam_cfg.get("track_point", track_point)  # per-camera override
        proc = CameraProcessor(camera_config=cam_cfg, homography=mapper, detector=detector, track_point=_cam_tp)
        if not proc.is_active():
            print(f"  [{cam_id}] ✗  Cannot open source — skipping")
            proc.release()
            continue
        processors.append((cam_id, proc))

    if not processors:
        print("  No cameras available — check sources or run --calibrate.\n")
        return

    detections_by_camera: dict[str, list] = {}
    camera_frames: dict[str, object] = {}

    for cam_id, proc in processors:
        floor_dets, frame = proc.process_frame()
        proc.release()
        detections_by_camera[cam_id] = floor_dets or []
        if frame is not None:
            camera_frames[cam_id] = proc.get_annotated_frame(frame)

    # ── Fuse ──────────────────────────────────────────────────────────────
    fused = fuser.fuse(detections_by_camera)
    stats = fuser.get_stats(fused)

    # ── Report ────────────────────────────────────────────────────────────
    print(f"\n  Cameras processed    : {list(camera_frames.keys())}")
    print(f"  Detections by camera : {  {k: len(v) for k, v in detections_by_camera.items()}  }")
    print(f"  Fused persons        : {stats['total_persons']}  "
          f"(fused={stats['fused_count']}, single={stats['single_count']})")
    for fd in fused:
        print(f"    {fd}")

    # ── Render composite view ─────────────────────────────────────────────
    if camera_frames:
        canvas = renderer.render_with_camera_feeds(camera_frames, fused, stats)
    else:
        canvas = renderer.render(fused, stats)

    _cv_show("Phase 4: Full Pipeline", canvas)
    print("\n  Press any key to close …")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("✓ Phase 4 complete.\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Production Mode  (--run)
# ═══════════════════════════════════════════════════════════════════════════

_CONTROLS_BANNER = """
=================================================================
  FACTORY FLOOR MONITOR  —  Production Mode
=================================================================
  [Q] / Esc   Quit
  [G]         Toggle grid lines
  [C]         Toggle camera-zone overlays
  [O]         Toggle overlap-zone overlay
  [E]         Toggle counting-edge lines (1 m grid amber overlay)
  [V]         Toggle floor grid projected onto camera feeds (debug mode)
  [L]         Toggle cell (col,row) labels in every grid cell
  [J]         Save grid-occupancy JSON report  → output/
  [S]         Save PNG snapshot                → output/
  [D]         Toggle debug panel  (adds camera feeds on right)
  [R]         Reset / reconnect all camera captures
=================================================================""" 


def run_live(source_dir: str = None, debug: bool = False,
             record: str = None, track_point: str = "bottom", model_name: str = "yolov8m.pt") -> None:
    """
    Production live multi-camera pipeline.

    1. Load configs + homographies (LensCorrector loaded automatically inside each mapper).
    2. Shared PersonDetector — one YOLO model for all cameras.
    3. CameraProcessor per camera — capture → undistort → detect → map-to-floor.
    4. DetectionFuser(load_overlap_zones()) — Hungarian matching per overlap zone.
    5. FloorRenderer — render floor map (or composite with camera feeds in debug mode).

    Keyboard controls (printed on startup):
      Q / Esc  quit
      G        toggle grid
      C        toggle camera zones
      O        toggle overlap zones
      S        save snapshot
      D        toggle debug (camera-feeds panel)
      R        reset all captures

    Target ~15 FPS.  Actual FPS depends on YOLO inference hardware.
    """
    import os
    import time
    from datetime import datetime

    import cv2
    from detection.detector import PersonDetector, CameraProcessor
    from calibration.homography import load_all_homographies, HomographyMapper
    from calibration.ocr_timestamp import TimestampExtractor
    from fusion.fuse import DetectionFuser
    from fusion.overlap import load_overlap_zones
    from fusion.crossing import LineCrossingDetector, generate_edges
    from visualization.floor_renderer import FloorRenderer

    # Force TCP transport for RTSP streams (Hikvision / Axis cameras)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    print(_CONTROLS_BANNER)

    # ── 1. Configs ────────────────────────────────────────────────────────
    with open(CAMERAS_CFG) as f:
        cameras_cfg = json.load(f)

    # ── 2. Homographies ───────────────────────────────────────────────────
    mappers = load_all_homographies(CONFIG_DIR)

    # ── 3. Shared detector ────────────────────────────────────────────────
    logger.info("[Live] Loading YOLO model …")
    detector = PersonDetector(model_name=model_name, confidence=0.45)
    logger.info("[Live] YOLO ready.")

    # ── 4. CameraProcessors ───────────────────────────────────────────────
    processors: list[tuple[str, CameraProcessor, dict]] = []
    for cam in cameras_cfg["cameras"]:
        cam_id  = cam["id"]
        cam_cfg = dict(cam)

        if source_dir:
            source_path = Path(source_dir)
            candidates  = list(source_path.glob(f"*{cam_id}*"))
            if candidates:
                cam_cfg["source"] = str(candidates[0])
                logger.info("[Live] %s → file: %s", cam_id, cam_cfg["source"])
            else:
                logger.warning("[Live] %s — no file found in %s; skipping", cam_id, source_dir)
                continue

        mapper = mappers.get(cam_id)
        if mapper is None:
            # Create an uncalibrated mapper so the processor still opens the capture
            mapper = HomographyMapper(cam_id, config_path=str(CONFIG_DIR) + "/")

        _cam_tp = cam_cfg.get("track_point", track_point)  # per-camera override
        proc = CameraProcessor(camera_config=cam_cfg, homography=mapper, detector=detector, track_point=_cam_tp)

        if proc.is_active():
            lens_tag = "lens ✓" if mapper.lens_corrector.is_calibrated else "lens ✗"
            homo_tag = "H ✓"    if mapper.is_calibrated                 else "H ✗ (no floor mapping)"
            logger.info("[Live] %-8s opened  %s  %s", cam_id, lens_tag, homo_tag)
            processors.append((cam_id, proc, cam_cfg))
        else:
            logger.warning("[Live] %s — cannot open source %s", cam_id, cam_cfg["source"])
            proc.release()

    if not processors:
        logger.error("[Live] No cameras available — check sources / calibration and retry.")
        return

    active_ids = [p[0] for p in processors]
    print(f"\n  Active cameras : {active_ids}")
    print(f"  Floor          : {cameras_cfg['cameras'][0].get('floor_coverage_polygon', 'N/A')}")
    with open(FLOOR_CFG) as f:
        _fc = json.load(f)
    print(f"  Floor size     : {_fc['floor_width_m']} m × {_fc['floor_height_m']} m")
    print()

    # ── 5. Fuser + Renderer ───────────────────────────────────────────────
    zones    = load_overlap_zones(OVERLAP_CFG)
    fuser    = DetectionFuser(zones)
    renderer = FloorRenderer(FLOOR_CFG, CAMERAS_CFG, OVERLAP_CFG)

    # ── 6. Line-crossing detectors (CSV log per camera) ────────────────────
    # Dynamically build a 1 m × 1 m edge grid from the floor dimensions and
    # write it back to edges.json so the user can inspect / override it.
    with open(FLOOR_CFG) as _fcf:
        _fc = json.load(_fcf)
    _edges = generate_edges(
        floor_width_m  = _fc["floor_width_m"],
        floor_height_m = _fc["floor_height_m"],
        step_m         = 1.0,
        save_path      = EDGES_CFG,
    )
    
    # We will instantiate crossing detectors per active camera below
    crossing_detectors: dict[str, LineCrossingDetector] = {}
    
    print(f"[Live] Crossing edges: {len(_edges)} lines  (1 m grid, saved → {EDGES_CFG.name})")

    TARGET_FPS      = 15
    TARGET_MS       = int(1000 / TARGET_FPS)   # ~66 ms per frame
    WIN_NAME        = "Factory Floor Monitor"  # ASCII-only — required for Qt5
    debug_mode      = debug
    snapshot_count  = 0
    json_count      = 0
    frame_count     = 0
    _cam_grid       = True   # 'V' key — projected floor grid on camera feeds
    # Floor dimensions used by the camera grid overlay
    _floor_w        = float(_fc["floor_width_m"])
    _floor_h        = float(_fc["floor_height_m"])
    _floor_step     = float(_fc.get("grid_cell_size_m", 1.0))
    _floor_major    = float(_fc.get("major_grid_every_m", 5.0))
    _session_start_ns = time.perf_counter_ns()   # for timestamp_ms in crossing log
    last_fused      = []      # kept so J key can report at any time
    last_stats: dict = {}
    _win_created    = False   # open WINDOW_AUTOSIZE on first rendered frame
    _live_scale     = 1.0    # display scale; recomputed whenever canvas size changes
    _last_shape     = (0, 0) # (h, w) of previous canvas — detect size changes
    detections_by_camera: dict[str, list] = {}
    camera_frames: dict[str, object] = {}

    # ── VideoWriter for --record ───────────────────────────────────────────
    _vwriter: cv2.VideoWriter | None = None
    if record:
        _rec_path = Path(record)
        if _rec_path.is_dir() or record.endswith("/"):
            ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
            _rec_path = Path(record) / f"recording_{ts}.mp4"
        _rec_path.parent.mkdir(parents=True, exist_ok=True)
        # Writer is created lazily on the first frame so we know the canvas size.
        print(f"[Live] Recording enabled → {_rec_path}")

    # ── Main loop ─────────────────────────────────────────────────────────
    try:
        while True:
            t0 = time.perf_counter()

            # Setup OCR extrators and crossing detectors per camera if missing
            camera_timestamps: dict[str, datetime] = {}
            for cam_id, proc, _ in processors:
                if cam_id not in crossing_detectors:
                    crossing_detectors[cam_id] = LineCrossingDetector(str(EDGES_CFG), cam_id, str(OUTPUT_DIR))
                
                # Assign extractor if not existing (storing via proc attributes or a separate dict)
                # For simplicity, stored on proc as `_ocr_extractor`
                extractor = getattr(proc, "_ocr_extractor", None)
                if extractor is None:
                    extractor = TimestampExtractor(cam_id, str(CONFIG_DIR) + "/")
                    setattr(proc, "_ocr_extractor", extractor)
                    setattr(proc, "_frame_idx", 0)

            for cam_id, proc, _ in processors:
                floor_dets, frame = proc.process_frame()
                detections_by_camera[cam_id] = floor_dets or []
                
                # Extract OCR Timestamp
                extractor = getattr(proc, "_ocr_extractor")
                fidx = getattr(proc, "_frame_idx")
                setattr(proc, "_frame_idx", fidx + 1)
                
                ts_dt = None
                if frame is not None:
                    ts_dt = extractor.extract(frame)
                    
                if ts_dt is None:
                    # Fallback
                    # Use a rough fps estimate like TARGET_FPS
                    ts_dt = extractor.get_fps_adjusted_timestamp(fidx, float(TARGET_FPS))
                    if ts_dt is None:
                        # Final fallback if base time never initialized
                        ts_dt = datetime.now()
                        
                camera_timestamps[cam_id] = ts_dt

                if debug_mode and frame is not None:
                    ann = proc.get_annotated_frame(frame)
                    # ── Projected floor grid on camera feed ───────────────
                    if _cam_grid:
                        ann = proc.draw_grid_overlay(
                            ann, _floor_w, _floor_h,
                            step_m    = _floor_step,
                            major_every = _floor_major,
                        )
                    # ── Highlight persons inside overlap zones ─────────────
                    # Draw a bright orange box + "OVERLAP" badge so overlap-
                    # zone persons are immediately obvious in the camera feed.
                    for fd in (floor_dets or []):
                        in_overlap = any(
                            zone.contains_point(fd.floor_x, fd.floor_y,
                                                use_buffer=False)
                            for zone in zones
                        )
                        if in_overlap:
                            bx1 = int(fd.pixel_bbox[0])
                            by1 = int(fd.pixel_bbox[1])
                            bx2 = int(fd.pixel_bbox[2])
                            by2 = int(fd.pixel_bbox[3])
                            # Thick orange rectangle
                            cv2.rectangle(ann, (bx1, by1), (bx2, by2),
                                          (0, 140, 255), 3)
                            # "OVERLAP" badge above the box
                            badge   = "OVERLAP"
                            bfont   = cv2.FONT_HERSHEY_SIMPLEX
                            (bw, bh), _ = cv2.getTextSize(badge, bfont, 0.5, 1)
                            badge_y = max(by1 - 6, bh + 8)
                            cv2.rectangle(ann,
                                          (bx1, badge_y - bh - 5),
                                          (bx1 + bw + 8, badge_y + 1),
                                          (0, 140, 255), -1)
                            cv2.putText(ann, badge,
                                        (bx1 + 4, badge_y - 3),
                                        bfont, 0.5, (255, 255, 255),
                                        1, cv2.LINE_AA)
                    camera_frames[cam_id] = ann

            # Fuse
            fused      = fuser.fuse(detections_by_camera)
            stats      = fuser.get_stats(fused)
            last_fused = fused
            last_stats = stats

            # ── Crossing detection → CSV per camera ───────────────────────────────────
            # Because fusing mixes camera sources, but we need per-camera CSV logs matching
            # the source camera ID (as specified in prompt), we iterate over fused tracks
            # and push updates to the detector of the camera(s) that provided the detection.
            for fd in fused:
                cam_ids = fd.source_cameras if fd.source_cameras else [list(camera_timestamps.keys())[0]] if camera_timestamps else ["unknown"]
                # For crossing detection, update once per involved camera (or just the primary)
                # Often it's best to run crossing on the primary fused coordinate for each camera involved.
                for cid in cam_ids:
                    if cid in crossing_detectors:
                        ts = camera_timestamps.get(cid, datetime.now())
                        crossing_detectors[cid].update(
                            track_id=fd.track_id, 
                            class_name="person", 
                            floor_x=fd.floor_x, 
                            floor_y=fd.floor_y, 
                            timestamp=ts
                        )

            # Render
            if debug_mode and camera_frames:
                canvas = renderer.render_with_camera_feeds(camera_frames, fused, stats)
            else:
                canvas = renderer.render(fused, stats)

            # ── Scale canvas to fit screen ────────────────────────────────────
            # Recompute scale whenever canvas dimensions change (e.g. D toggle).
            # Use 2× width budget for composite (floor + camera feeds) mode.
            ch, cw = canvas.shape[:2]
            if (ch, cw) != _last_shape:
                budget_w = _MAX_DISP_W * 2 if debug_mode else _MAX_DISP_W
                _live_scale = min(budget_w / max(cw, 1),
                                  _MAX_DISP_H / max(ch, 1), 1.0)
                _last_shape = (ch, cw)

            if _live_scale < 1.0:
                disp = cv2.resize(
                    canvas,
                    (max(1, int(cw * _live_scale)), max(1, int(ch * _live_scale))),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                disp = canvas

            if not _win_created:
                cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
                _win_created = True
            cv2.imshow(WIN_NAME, disp)

            # ── Record frame ──────────────────────────────────────────────────
            if record:
                if _vwriter is None:
                    fh, fw = canvas.shape[:2]
                    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
                    _vwriter = cv2.VideoWriter(
                        str(_rec_path), fourcc, TARGET_FPS, (fw, fh))
                    if not _vwriter.isOpened():
                        print("[Live] WARNING: VideoWriter failed to open — "
                              "recording disabled.")
                        _vwriter = None
                if _vwriter is not None:
                    _vwriter.write(canvas)

            # ── Frame-rate control ─────────────────────────────────────────
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            wait_ms    = max(1, TARGET_MS - elapsed_ms)
            key        = cv2.waitKey(wait_ms) & 0xFF

            # ── Keyboard handling ──────────────────────────────────────────
            if key in (ord("q"), ord("Q"), 27):                   # Q / Esc
                print("\n[Live] Quit by user.")
                break

            elif key in (ord("g"), ord("G")):                     # G — grid
                state = renderer.toggle_grid()
                print(f"[Live] Grid: {'ON' if state else 'OFF'}")

            elif key in (ord("c"), ord("C")):                     # C — cameras
                state = renderer.toggle_cameras()
                print(f"[Live] Camera zones: {'ON' if state else 'OFF'}")

            elif key in (ord("o"), ord("O")):                     # O — overlap
                state = renderer.toggle_overlap()
                print(f"[Live] Overlap zones: {'ON' if state else 'OFF'}")

            elif key in (ord("e"), ord("E")):                     # E — counting edges
                state = renderer.toggle_edges()
                print(f"[Live] Counting edges: {'ON' if state else 'OFF'}")

            elif key in (ord("v"), ord("V")):                     # V — video grid
                _cam_grid = not _cam_grid
                print(f"[Live] Camera grid overlay: {'ON' if _cam_grid else 'OFF'}")

            elif key in (ord("l"), ord("L")):                     # L — cell labels
                state = renderer.toggle_cell_labels()
                print(f"[Live] Cell labels: {'ON' if state else 'OFF'}")

            elif key in (ord("j"), ord("J")):                     # J — JSON report
                if last_fused:
                    json_count += 1
                    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path   = OUTPUT_DIR / f"grid_report_{ts}_{json_count:03d}.json"
                    report = renderer.get_grid_report(last_fused, last_stats)
                    with open(path, "w") as _fj:
                        json.dump(report, _fj, indent=2)
                    print(f"[Live] JSON report saved: output/{path.name}")
                    print(f"       Persons: {report['total_persons']}  "
                          f"Fused: {report['fused_count']}  "
                          f"Occupied cells: {report['occupied_cell_count']}")
                else:
                    print("[Live] No detections yet — run for a few frames first.")

            elif key in (ord("s"), ord("S")):                     # S — snapshot
                snapshot_count += 1
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = OUTPUT_DIR / f"snapshot_{ts}_{snapshot_count:03d}.png"
                ok   = renderer.save_snapshot(canvas, path)
                if ok:
                    print(f"[Live] Snapshot saved: output/{path.name}")

            elif key in (ord("d"), ord("D")):                     # D — debug toggle
                debug_mode = not debug_mode
                print(f"[Live] Debug mode: {'ON (camera feeds)' if debug_mode else 'OFF'}")

            elif key in (ord("r"), ord("R")):                     # R — reset
                print("[Live] Resetting all camera captures …")
                for _, proc, _ in processors:
                    proc.reset()
                fuser.reset_tracks()          # clear EMA tracks so they reseed cleanly
                for cd in crossing_detectors.values():
                    cd.reset()     # clear previous positions so no phantom crossings
                print("[Live] Reset complete.")

            frame_count += 1

    except KeyboardInterrupt:
        print("\n[Live] Interrupted.")

    finally:
        for _, proc, _ in processors:
            proc.release()
        if record and _vwriter is not None:
            _vwriter.release()
            print(f"[Live] Recording saved → {_rec_path}")
            
        for cd in crossing_detectors.values():
            cd.close()
            
        print(f"[Live] Crossing CSV logs saved → {OUTPUT_DIR}/")
        cv2.destroyAllWindows()
        logger.info("[Live] Stopped after %d frames.", frame_count)

        # ── Auto-save final JSON report on every exit ──────────────────────
        if frame_count > 0:
            try:
                ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
                path   = OUTPUT_DIR / f"grid_report_{ts}_final.json"
                report = renderer.get_grid_report(last_fused, last_stats)
                with open(path, "w") as _fj:
                    json.dump(report, _fj, indent=2)
                print(f"[Live] Auto-saved final report → output/{path.name}")
                print(f"       Persons: {report['total_persons']}  "
                      f"Fused: {report['fused_count']}  "
                      f"Occupied cells: {report['occupied_cell_count']}")
            except Exception as _e:
                logger.warning("[Live] Could not auto-save report: %s", _e)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="factory-floor-viz",
        description="Multi-camera factory floor person-tracking visualisation.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples
--------
# Lens distortion calibration (Step 2):
  python main.py --intrinsic cam_1                                   # chessboard, live RTSP
  python main.py --intrinsic cam_1 --source video.mp4               # from video file
  python main.py --intrinsic cam_1 --method circles                 # dot grid
  python main.py --intrinsic cam_1 --method circles --grid-cols 4 --grid-rows 11
  python main.py --intrinsic cam_1 --method circles --asymmetric    # asymmetric grid
  python main.py --intrinsic cam_1 --method markers --grid-cols 5 --grid-rows 4
  python main.py --intrinsic cam_1 --method markers --grid-spacing 500
  python main.py --intrinsic cam_1 --method lines                   # automatic, no pattern

# Floor coverage polygon (click corners on camera feed):
  python main.py --coverage cam_1                                    # auto coords (if calibrated)
  python main.py --coverage cam_1 --source video.mp4                # from video file
  python main.py --coverage cam_1 --source frame.jpg                # from still image

# Floor-point homography calibration:
  python main.py --calibrate cam_1                                   # uses RTSP from config
  python main.py --calibrate cam_1 --source video.mp4               # from video file
  python main.py --calibrate cam_1 --source frame.jpg               # from still image

# Development phase tests:
  python main.py --phase 1                                           # validate all configs
  python main.py --phase 2                                           # homography quality report
  python main.py --phase 3                                           # YOLO test, 1 frame/camera
  python main.py --phase 3 --source ./test_videos                   # from local videos
  python main.py --phase 4                                           # full pipeline, 1 frame
  python main.py --phase 4 --source ./test_videos

# Production live mode:
  python main.py --run                                               # live pipeline
  python main.py --run --debug                                       # + camera-feeds panel
  python main.py --run --source ./test_videos                        # use local video files
  python main.py --run --source ./test_videos --debug

# Demo mode (no cameras / YOLO required):
  python main.py --demo                                              # animated floor simulation
  python main.py --demo --debug                                      # reserved for future use
  python main.py --demo --snapshot                                   # save final frame on exit
""",
    )

    # ── Mutually exclusive modes ──────────────────────────────────────────
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--phase", type=int, choices=[1, 2, 3, 4], metavar="N",
        help="Development phase (1=configs, 2=homography, 3=detection, 4=full)",
    )
    mode.add_argument(
        "--intrinsic", metavar="CAM_ID",
        help="Run lens/intrinsic calibration for CAM_ID",
    )
    mode.add_argument(
        "--calibrate", metavar="CAM_ID",
        help="Interactive floor-point homography calibration for CAM_ID",
    )
    mode.add_argument(
        "--coverage", metavar="CAM_ID",
        help=(
            "Interactively define floor_coverage_polygon for CAM_ID by clicking\n"
            "corners on the camera feed.  If homography is already calibrated,\n"
            "pixel clicks auto-convert to floor metres.  Otherwise the terminal\n"
            "prompts for each coordinate."
        ),
    )
    mode.add_argument(
        "--run", action="store_true",
        help="Start live production pipeline",
    )
    mode.add_argument(
        "--demo", action="store_true",
        help="Synthetic demo simulation (no cameras required)",
    )
    mode.add_argument(
        "--auto-config", action="store_true", dest="auto_config",
        help=(
            "Auto-compute floor_config.json dimensions and overlap_zones.json\n"
            "from the floor_coverage_polygon values already saved in cameras.json.\n"
            "Run this after --coverage to regenerate floor size and overlap zones."
        ),
    )
    mode.add_argument(
        "--ocr-region", metavar="CAM_ID",
        help="Interactively select and save the timestamp OCR region for CAM_ID",
    )
    mode.add_argument(
        "--ocr-test", metavar="CAM_ID",
        help="Test OCR timestamp extraction on 10 frames from CAM_ID",
    )
    mode.add_argument(
        "--process", action="store_true",
        help="Run the full offline crossing pipeline (per-camera YOLO + multi-camera fusion)",
    )
    mode.add_argument(
        "--process-camera", metavar="CAM_ID",
        help="Run the per-camera offline YOLO pipeline for a single CAM_ID",
    )
    mode.add_argument(
        "--fuse-only", action="store_true",
        help="Run only the multi-camera fusion step on existing output/*_crossings.csv files",
    )
    mode.add_argument(
        "--visualize", metavar="CSV_FILE", type=str,
        help="Visualise offline processing results directly from a generated CSV file",
    )
    parser.add_argument(
        "--headless-mp4", metavar="OUTPUT.mp4", type=str,
        help="Used with --visualize to render headlessly to an MP4 video instead of displaying a window.",
    )

    # ── Shared options ────────────────────────────────────────────────────
    parser.add_argument(
        "--timestamp-tolerance", type=float, metavar="SECONDS",
        help="Override fusion_config.json timestamp_tolerance_s for fusion",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing per-camera CSVs (crossings + tracks) instead of replacing",
    )
    parser.add_argument(
        "--playback-speed", type=float, default=1.0,
        help="Speed multiplier for offline CSV visualization (e.g., 2.0)",
    )
    parser.add_argument(
        "--source", metavar="PATH",
        help=(
            "For --calibrate/--coverage  : video file, RTSP URL, "
            "or still image (.jpg .png .bmp .tiff)\n"
            "For --intrinsic             : video file or RTSP URL\n"
            "For --phase 3/4 and --run   : directory of video files"
        ),
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug panel (camera feeds alongside floor map)",
    )
    parser.add_argument(
        "--snapshot", action="store_true",
        help="(With --demo) save a snapshot image on exit",
    )
    parser.add_argument(
        "--record", metavar="PATH",
        help=(
            "(With --run) record the rendered visualization to a video file.\n"
            "Example: --record output/run.mp4\n"
            "If no path is given the file is auto-named in output/."
        ),
    )
    parser.add_argument(
        "--headless", "--no-gui", action="store_true",
        help="Skip interactive windows and user prompts (saves validation checks to file)",
    )
    parser.add_argument(
        "--track-point",
        choices=["bottom", "center", "top"],
        default="bottom",
        help=(
            "Which point on the bounding box is projected onto the floor plane "
            "for tracking and homography.\n"
            "  bottom — feet / bottom-centre  [default, best for floor-plane]\n"
            "  center — vertical mid-body\n"
            "  top    — head / top-centre"
        ),
    )
    parser.add_argument(
        "--model", type=str, default="yolov8m.pt",
        help="YOLO model weights file to use (e.g. yolov8n.pt, yolov8m.pt, yolov8s.pt)",
    )
    parser.add_argument(
        "--classes", type=str, default=None, metavar="CLASSES",
        help=(
            "Comma-separated YOLO classes to detect (default: person,car,motorcycle,truck).\n"
            "Use names:  'person,truck'   or COCO IDs: '0,2,3,7'\n"
            "Use 'all'   to detect every COCO class.\n"
            "Use 'person' to detect persons only (original behaviour)."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=4, metavar="N",
        help=(
            "Number of cameras to process in parallel (default 4).\n"
            "Each worker loads its own YOLO model on the GPU.\n"
            "A10G (23 GB VRAM): up to 9 workers with yolov8m (~1.5 GB each).\n"
            "Use --workers 1 for fully sequential processing."
        ),
    )
    parser.add_argument(
        "--frame-stride", type=int, default=1, dest="frame_stride", metavar="N",
        help=(
            "Process every Nth video frame (default 1 = every frame).\n"
            "Set to 2 to halve YOLO calls with minimal tracking impact.\n"
            "Frame counter always advances so timestamps stay accurate."
        ),
    )
    parser.add_argument(
        "--ocr-interval", type=int, default=0, dest="ocr_interval", metavar="N",
        help=(
            "Run EasyOCR every N frames (default 0 = auto = once per second).\n"
            "EasyOCR is the slowest step; this gives 10-15× speedup for long\n"
            "recordings. FPS-interpolation fills in between OCR reads."
        ),
    )

    # ── Intrinsic calibration options ─────────────────────────────────────
    parser.add_argument(
        "--method",
        choices=["chessboard", "circles", "markers", "lines"],
        default="chessboard",
        help=(
            "(--intrinsic) calibration method:\n"
            "  chessboard — printed 9×6 board        [default, most accurate]\n"
            "  circles    — printed dot/circle grid   [easy to make]\n"
            "  markers    — click existing floor marks [no printing needed]\n"
            "  lines      — straight lines in scene    [automatic, rough]"
        ),
    )
    parser.add_argument("--board-w",     type=int,   default=9,
                        help="Chessboard inner corners width  (default 9)")
    parser.add_argument("--board-h",     type=int,   default=6,
                        help="Chessboard inner corners height (default 6)")
    parser.add_argument("--square-size", type=float, default=0.025,
                        help="Chessboard square side in metres  (default 0.025 = 25 mm)")
    parser.add_argument("--min-samples", type=int,   default=20,
                        help="Min valid captures before calibrating  (default 20)")
    parser.add_argument("--grid-cols",   type=int,   default=4,
                        help="Grid columns for --method circles/markers  (default 4)")
    parser.add_argument("--grid-rows",   type=int,   default=11,
                        help="Grid rows    for --method circles/markers  (default 11)")
    parser.add_argument("--grid-spacing", type=float, default=25.0,
                        help="Spacing between grid points in mm  (default 25)")
    parser.add_argument("--grid-views",  type=int,   default=5,
                        help="Target number of views for --method markers  (default 5)")
    parser.add_argument("--asymmetric",  action="store_true",
                        help="Use asymmetric circle grid  (with --method circles)")

    return parser


def _resolve_calib_source(source_path: str) -> str:
    """If the source is a directory, return the first video file within it for calibration."""
    p = Path(source_path)
    if p.is_dir():
        vids = sorted(
            [f for f in p.iterdir() if f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]]
        )
        if vids:
            return str(vids[0])
    return source_path


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # No mode → print help
    if not any([args.phase, args.intrinsic, args.calibrate,
                args.coverage, args.ocr_region, args.ocr_test,
                args.process, args.process_camera, args.fuse_only,
                args.visualize, args.run, args.demo, args.auto_config]):
        parser.print_help()
        sys.exit(0)

    # ------------------------------------------------------------------
    if args.auto_config:
        print("\n[Auto-config] Computing floor dimensions and overlap zones …\n")
        ok = auto_configure()
        sys.exit(0 if ok else 1)

    # ------------------------------------------------------------------
    if args.phase == 1:
        phase_1()

    elif args.phase == 2:
        phase_2()

    elif args.phase == 3:
        phase_3(source_dir=args.source, track_point=args.track_point, model_name=args.model)

    elif args.phase == 4:
        phase_4(source_dir=args.source, track_point=args.track_point, model_name=args.model)

    # ------------------------------------------------------------------
    elif args.intrinsic:
        from calibration.lens_correction import LensCorrector

        cam_id    = args.intrinsic
        corrector = LensCorrector(cam_id, config_path=str(CONFIG_DIR) + "/")

        if args.source:
            source = args.source
        else:
            with open(CAMERAS_CFG) as _f:
                _cfg = json.load(_f)
            _entry = next((c for c in _cfg["cameras"] if c["id"] == cam_id), None)
            if _entry is None:
                print(f"  ✗  Camera '{cam_id}' not found in cameras.json")
                sys.exit(1)
            source = _entry["source"]
            
        source = _resolve_calib_source(source)

        if args.method == "lines":
            ok = corrector.calibrate_from_lines(source=source)

        elif args.method == "circles":
            ok = corrector.calibrate_from_circle_grid(
                source=source,
                grid_size=(args.grid_cols, args.grid_rows),
                grid_spacing_mm=args.grid_spacing,
                num_frames=args.min_samples,
                symmetric=not args.asymmetric,
            )

        elif args.method == "markers":
            ok = corrector.calibrate_from_floor_markers(
                source=source,
                grid_cols=args.grid_cols,
                grid_rows=args.grid_rows,
                grid_spacing_mm=args.grid_spacing,
                num_views=args.grid_views,
            )

        else:   # chessboard (default)
            ok = corrector.calibrate_from_chessboard(
                source=source,
                board_size=(args.board_w, args.board_h),
                square_size_mm=args.square_size * 1000,
                num_frames=args.min_samples,
            )

        corrector.print_summary()

        if ok:
            import os as _os
            if _os.path.exists(source):
                import cv2 as _cv2
                _cap = _cv2.VideoCapture(source)
                _ret, _frame = _cap.read()
                _cap.release()
                if _ret:
                    if args.headless:
                        # Auto-save comparison to file instead of showing
                        corrector.save_undistortion_comparison(_frame)
                    else:
                        show = input("  Show undistortion comparison? [y/N] ").strip().lower()
                        if show == "y":
                            corrector.show_undistortion_comparison(_frame)

        sys.exit(0 if ok else 1)

    # ------------------------------------------------------------------
    elif args.calibrate:
        from calibration.calibrate import CalibrationTool

        cam_id = args.calibrate

        if args.source:
            cam_source = args.source
        else:
            with open(CAMERAS_CFG) as _f:
                _cfg = json.load(_f)
            _entry = next((c for c in _cfg["cameras"] if c["id"] == cam_id), None)
            if _entry is None:
                print(f"  ✗  Camera '{cam_id}' not found in cameras.json")
                sys.exit(1)
            cam_source = _entry["source"]
            
        cam_source = _resolve_calib_source(cam_source)

        tool = CalibrationTool(
            camera_id   = cam_id,
            source      = cam_source,
            config_path = str(CONFIG_DIR) + "/",
        )
        ok = tool.run()
        if ok:
            print("\n[Auto-config] Updating floor size and overlap zones …")
            auto_configure()
        sys.exit(0 if ok else 1)

    # ------------------------------------------------------------------
    elif args.coverage:
        from calibration.coverage import CoverageMapper

        cam_id = args.coverage

        if args.source:
            cam_source = args.source
        else:
            with open(CAMERAS_CFG) as _f:
                _cfg = json.load(_f)
            _entry = next((c for c in _cfg["cameras"] if c["id"] == cam_id), None)
            if _entry is None:
                print(f"  ✗  Camera '{cam_id}' not found in cameras.json")
                sys.exit(1)
            cam_source = _entry["source"]
            
        cam_source = _resolve_calib_source(cam_source)

        tool = CoverageMapper(
            camera_id   = cam_id,
            source      = cam_source,
            config_path = str(CONFIG_DIR) + "/",
        )
        ok = tool.run()
        if ok:
            print("\n[Auto-config] Updating floor size and overlap zones …")
            auto_configure()
        sys.exit(0 if ok else 1)

    # ------------------------------------------------------------------
    elif args.ocr_region:
        from calibration.ocr_region import OCRRegionSelector, save_ocr_region

        cam_id = args.ocr_region

        if args.source:
            cam_source = args.source
        else:
            with open(CAMERAS_CFG) as _f:
                _cfg = json.load(_f)
            _entry = next((c for c in _cfg["cameras"] if c["id"] == cam_id), None)
            if _entry is None:
                print(f"  ✗  Camera '{cam_id}' not found in cameras.json")
                sys.exit(1)
            cam_source = _entry["source"]
            
        cam_source = _resolve_calib_source(cam_source)

        selector = OCRRegionSelector(camera_id=cam_id, source=cam_source)
        roi = selector.select_region()
        
        if roi is not None:
            ok = save_ocr_region(cam_id, roi, config_path=str(CONFIG_DIR) + "/")
            sys.exit(0 if ok else 1)
        else:
            sys.exit(1)

    # ------------------------------------------------------------------
    elif args.ocr_test:
        from calibration.ocr_timestamp import TimestampExtractor
        import cv2, time
        cam_id = args.ocr_test
        
        with open(CAMERAS_CFG) as _f:
            _cfg = json.load(_f)
        _entry = next((c for c in _cfg["cameras"] if c["id"] == cam_id), None)
        if _entry is None:
            print(f"  ✗  Camera '{cam_id}' not found in cameras.json")
            sys.exit(1)
            
        print(f"\n[OCR Test] Starting 10-frame extraction for camera: {cam_id}")
        source = _entry["source"]
        source = _resolve_calib_source(source)
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"  ✗  Could not open video source: {source}")
            sys.exit(1)
            
        extractor = TimestampExtractor(cam_id, config_path=str(CONFIG_DIR) + "/")
        success_count = 0
        
        for i in range(10):
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"  !  Warning: ended early at frame {i+1}")
                break
                
            # Simulate real-time delay or standard read spread if needed, but sequential is fine
            t0 = time.time()
            ts = extractor.extract(frame)
            t1 = time.time()
            ts_str = ts.isoformat() if ts else "FAILED"
            ms = (t1 - t0) * 1000
            
            if ts:
                success_count += 1
                
            print(f"  Frame {i+1:2d} | extracted: {ts_str} | took {ms:.1f} ms")
            
        cap.release()
        pct = (success_count / 10) * 100
        print(f"\n[OCR Test] Result: {success_count}/10 ({pct:.0f}%) parsed successfully.")

    # ------------------------------------------------------------------
    elif args.process_camera:
        from pipeline.per_camera import PerCameraProcessor
        from detection.detector import PersonDetector
        
        cam_id = args.process_camera
        print(f"\n[Pipeline] Running offline processing for camera: {cam_id}")
        
        detector = PersonDetector(model_name=args.model, target_classes=args.classes)
        try:
            proc = PerCameraProcessor(
                camera_id=cam_id,
                config_dir=str(CONFIG_DIR),
                output_dir=str(OUTPUT_DIR),
                model=detector,
                append_output=args.append,
            )
            vinfo = proc.get_video_info()
            if not vinfo:
                print(f"  ✗  Could not initialize video source for camera {cam_id}")
                sys.exit(1)
                
            print(f"    Source Frames: {vinfo['total_frames']} | Resolution: {vinfo['resolution']}")
            
            def _prog(cur, tot):
                if tot > 0 and cur % max(1, tot // 10) == 0:
                    print(f"  [{cam_id}] Progress: {cur} / {tot} ({(cur/tot)*100:.0f}%)")
                    
            csv_out = proc.process_video(
                progress_callback = _prog,
                frame_stride      = args.frame_stride,
                ocr_interval      = args.ocr_interval,
            )
            print(f"\n  ✓  Saved crossings to: {csv_out}")
            
        except Exception as e:
            print(f"  ✗  Error processing camera {cam_id}: {e}")
            sys.exit(1)

    # ------------------------------------------------------------------
    elif args.fuse_only:
        from fusion.multi_camera_fusion import CrossingFuser
        import glob
        
        print(f"\n[Pipeline] Running multi-camera fusion on existing CSVs...")
        csv_files = glob.glob(str(OUTPUT_DIR / "*_crossings.csv"))
        # Exclude the fused output itself if passing multiple times
        csv_files = [f for f in csv_files if "fused_crossings" not in f]
        
        if not csv_files:
            print(f"  ✗  No per-camera CSVs found in {OUTPUT_DIR}/")
            sys.exit(1)
            
        print(f"Found {len(csv_files)} camera CSVs.")
        
        tol = args.timestamp_tolerance
        if tol is None:
            # Fallback to configured
            try:
                with open(FUSION_CFG) as _f:
                    fcfg = json.load(_f)
                tol = fcfg.get("timestamp_tolerance_s", 1.0)
            except Exception:
                tol = 1.0
                
        fuser = CrossingFuser(str(OVERLAP_CFG), timestamp_tolerance_s=tol, config_dir=str(CONFIG_DIR))
        
        fused_df = fuser.fuse(csv_files)
        out_path = str(OUTPUT_DIR / "fused_crossings.csv")
        fuser.save_fused_csv(fused_df, out_path)
        
        print("\n  ✓  Fusion complete")

    # ------------------------------------------------------------------
    elif args.process:
        from pipeline.per_camera import MultiCameraRunner
        from fusion.multi_camera_fusion import CrossingFuser
        
        print(f"\n[Pipeline] Validating cameras for full processing...")
        with open(CAMERAS_CFG) as _f:
            _cfg = json.load(_f)
            
        # Quick validation
        for cam in _cfg.get("cameras", []):
            if "homography_matrix" not in cam:
                print(f"  ✗  Camera {cam['id']} is missing homography. Run --calibrate.")
                sys.exit(1)
            if "ocr_region" not in cam:
                print(f"  ✗  Camera {cam['id']} is missing OCR region. Run --ocr-region.")
                sys.exit(1)
                
        print("  ✓  All configs validated. Starting YOLO extraction...")
        
        # 1. Pipeline Runner
        runner = MultiCameraRunner(
            config_dir     = str(CONFIG_DIR),
            output_dir     = str(OUTPUT_DIR),
            model_path     = args.model,
            append_output  = args.append,
            target_classes = args.classes,
        )
        csv_paths = runner.run_all(
            sequential    = False,
            max_workers   = args.workers,
            frame_stride  = args.frame_stride,
            ocr_interval  = args.ocr_interval,
        )
        
        # 2. Fusion
        print(f"\n[Pipeline] Running multi-camera fusion...")
        tol = args.timestamp_tolerance
        if tol is None:
            try:
                with open(FUSION_CFG) as _f:
                    fcfg = json.load(_f)
                tol = fcfg.get("timestamp_tolerance_s", 1.0)
            except Exception:
                tol = 1.0
                
        fuser = CrossingFuser(str(OVERLAP_CFG), timestamp_tolerance_s=tol, config_dir=str(CONFIG_DIR))
        fused_df = fuser.fuse(csv_paths)
        
        out_path = str(OUTPUT_DIR / "fused_crossings.csv")
        fuser.save_fused_csv(fused_df, out_path)
        
        try:
            import pandas as pd
            orig_dfs = [pd.read_csv(p) for p in csv_paths]
            summary = fuser.get_summary(orig_dfs, fused_df)
            print("\n[Summary]")
            for k, v in summary.items():
                print(f"  {k:25s} : {v}")
        except Exception as e:
            logger.warning("Could not generate text summary report: %s", e)
            
        print("\n  ✓  Full offline pipeline complete.")

    # ------------------------------------------------------------------
    elif args.visualize:
        import os
        from visualization.offline_renderer import CsvVisualizer
        
        csv_path = args.visualize
        if not os.path.exists(csv_path):
            print(f"  ✗  CSV file '{csv_path}' not found.")
            sys.exit(1)
            
        print(f"\n[Pipeline] Starting offline visualization: {csv_path} at {args.playback_speed}x...")
        
        with open(FLOOR_CFG) as f:
            floor_config = json.load(f)
        with open(CAMERAS_CFG) as f:
            cameras_config = json.load(f)
        try:
            with open(OVERLAP_CFG) as f:
                overlap_config = json.load(f)
        except Exception:
            overlap_config = {}
            
        visualizer = CsvVisualizer(
            csv_path=csv_path,
            floor_config=floor_config,
            cameras_config=cameras_config,
            overlap_config=overlap_config,
            playback_speed=args.playback_speed,
            persistence_s=1.0,
            output_video=args.headless_mp4
        )
        ok = visualizer.run()
        
        # Post-process with FFmpeg to ensure strict browser compatibility (H.264, yuv420p, moov atom faststart)
        if ok and args.headless_mp4:
            import shutil
            import subprocess
            if shutil.which("ffmpeg"):
                print("\n  [Pipeline] Re-encoding MP4 with FFmpeg for maximum web compatibility...")
                tmp_out = str(args.headless_mp4) + ".tmp.mp4"
                cmd = [
                    "ffmpeg", "-y", "-i", str(args.headless_mp4),
                    "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-loglevel", "error", tmp_out
                ]
                try:
                    subprocess.run(cmd, check=True)
                    os.replace(tmp_out, str(args.headless_mp4))
                    print("  ✓  FFmpeg re-encoding complete.")
                except subprocess.CalledProcessError as e:
                    print(f"  ✗  FFmpeg re-encoding failed: {e}")
                    if os.path.exists(tmp_out):
                        os.unlink(tmp_out)
        
        sys.exit(0 if ok else 1)

    # ------------------------------------------------------------------
    elif args.run:
        run_live(source_dir=args.source, debug=args.debug, record=args.record, track_point=args.track_point, model_name=args.model)

    # ------------------------------------------------------------------
    elif args.demo:
        from visualization.demo_simulator import run_demo
        run_demo(debug=args.debug, save_snapshot=args.snapshot)


if __name__ == "__main__":
    main()
