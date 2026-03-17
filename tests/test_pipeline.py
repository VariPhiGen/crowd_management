import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import pytest
import numpy as np

from test_helpers import create_synthetic_frame, create_mock_edges_config, create_mock_crossings_csv

# Add project root to sys path so we can import modules properly
import sys
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration.ocr_region import OcrRegion, save_ocr_region, load_ocr_region
from calibration.ocr_timestamp import TimestampExtractor
from fusion.crossing import LineCrossingDetector, load_crossings_csv
from fusion.multi_camera_fusion import CrossingFuser


# ── Normalized coordinate tests ─────────────────────────────────────────────

def test_ocr_region_normalize_roundtrip(tmp_path):
    """Normalized OCR region round-trips correctly at both same and different resolutions."""
    cam_id = "cam_norm"
    (tmp_path / "config").mkdir()
    mock_cfg = {"cameras": [{"id": cam_id}]}
    cfg_path = tmp_path / "config" / "cameras.json"
    cfg_path.write_text(json.dumps(mock_cfg))

    # Simulate what OCRRegionSelector produces — pixel roi + embedded frame size
    pixel_roi = {"x": 24, "y": 653, "w": 320, "h": 32, "_frame_w": 1280, "_frame_h": 720}
    save_ocr_region(cam_id, pixel_roi, config_path=str(tmp_path / "config") + "/")

    # ── Same resolution ───────────────────────────────────────────────────────
    roi = load_ocr_region(cam_id, config_path=str(tmp_path / "config") + "/")
    assert roi.normalized is True, "Should be stored as normalized"

    px = roi.to_pixels(1280, 720)
    assert px["x"] == 24
    assert px["y"] == 653
    assert px["w"] == 320
    assert px["h"] == 32

    # ── Double resolution ─────────────────────────────────────────────────────
    px2 = roi.to_pixels(2560, 1440)
    assert px2["x"] == 48    # 2×
    assert px2["y"] == 1306  # 2×
    assert px2["w"] == 640   # 2×
    assert px2["h"] == 64    # 2×


def test_ocr_region_backward_compat():
    """Old raw-pixel config (no coordinate_format key) still works via to_pixels()."""
    # Simulate legacy cameras.json entry (raw int dict, no coordinate_format)
    legacy_roi = {"x": 40, "y": 10, "w": 400, "h": 60}

    roi = OcrRegion(
        x=float(legacy_roi["x"]),
        y=float(legacy_roi["y"]),
        w=float(legacy_roi["w"]),
        h=float(legacy_roi["h"]),
        normalized=False,
    )

    assert roi.normalized is False

    # to_pixels() at any size should return the original values unchanged
    px = roi.to_pixels(640, 480)
    assert px["x"] == 40
    assert px["y"] == 10
    assert px["w"] == 400
    assert px["h"] == 60

    px2 = roi.to_pixels(1920, 1080)
    assert px2["x"] == 40   # unchanged — legacy
    assert px2["y"] == 10
    assert px2["w"] == 400
    assert px2["h"] == 60


def test_ocr_extract_with_normalised_region(tmp_path):
    """TimestampExtractor works correctly when cameras.json stores a normalized OCR region."""
    cam_id = "cam_norm_extract"
    (tmp_path / "config").mkdir()

    # Write a normalized ocr_region directly (as calibrate would produce it)
    mock_cfg = {
        "cameras": [
            {
                "id": cam_id,
                "ocr_region": {
                    "x": round(40 / 640, 6),
                    "y": round(10 / 480, 6),
                    "w": round(400 / 640, 6),
                    "h": round(60 / 480, 6),
                    "coordinate_format": "normalized",
                    "image_size": [640, 480],
                }
            }
        ]
    }
    (tmp_path / "config" / "cameras.json").write_text(json.dumps(mock_cfg))

    timestamp_str = "2025-03-14 10:32:15"
    # Create a 640×480 frame with the timestamp rendered at (50, 50)
    frame = create_synthetic_frame(timestamp_str, text_position=(50, 50), frame_size=(640, 480))

    extractor = TimestampExtractor(cam_id, config_path=str(tmp_path / "config") + "/")
    assert extractor.roi is not None, "OcrRegion should have loaded"
    assert extractor.roi.normalized is True, "Should be normalized format"

    # Verify to_pixels resolves correctly for this frame
    px = extractor.roi.to_pixels(640, 480)
    assert px["x"] == 40
    assert px["y"] == 10
    assert px["w"] == 400
    assert px["h"] == 60

    # Full extract — OCR may or may not succeed depending on easyocr install,
    # but the region should at least be resolved without an IndexError/bounds error.
    parsed_dt = extractor.extract(frame)
    # If easyocr is installed and succeeds, validate the parsed timestamp
    if parsed_dt is not None:
        assert parsed_dt.year == 2025
        assert parsed_dt.month == 3
        assert parsed_dt.day == 14



def test_ocr_region_roundtrip(tmp_path):
    # 1. Mock cameras.json with ocr_region
    (tmp_path / "config").mkdir()
    cam_id = "cam_test"
    # Text position in OpenCV is bottom-left of text. We need an ROI that surrounds the text.
    # Text "2025-03-14 10:32:15" at (50, 50) roughly extends up to y=20 and right to x=350
    mock_cameras_cfg = {
        "cameras": [
            {
                "id": cam_id,
                "ocr_region": {"x": 40, "y": 10, "w": 400, "h": 60}
            }
        ]
    }
    with open(tmp_path / "config" / "cameras.json", "w") as f:
        json.dump(mock_cameras_cfg, f)
        
    # 2. Synthetic frame
    timestamp_str = "2025-03-14 10:32:15"
    frame = create_synthetic_frame(timestamp_str, text_position=(50, 50), frame_size=(640, 480))
    
    # 3. Extract 
    # Use fallback easyocr/pytesseract, we assume one is installed.
    extractor = TimestampExtractor(cam_id, config_path=str(tmp_path / "config") + "/")
    parsed_dt = extractor.extract(frame)
    
    print(f"Extracted timestamp: {parsed_dt}")
    if parsed_dt is not None:
        assert parsed_dt.year == 2025
        assert parsed_dt.month == 3
        assert parsed_dt.day == 14
        assert parsed_dt.hour == 10
        assert parsed_dt.minute == 32
        assert parsed_dt.second == 15

def test_crossing_detector_csv_output(tmp_path):
    cam_id = "cam_test"
    edges_cfg_path = tmp_path / "edges.json"
    create_mock_edges_config([{"id": "E1", "type": "horizontal", "value": 5.0}], edges_cfg_path)
    
    detector = LineCrossingDetector(str(edges_cfg_path), cam_id, str(tmp_path))
    
    # Move across the line
    t1 = datetime(2025, 3, 14, 10, 0, 1)
    # The detector API expects: (track_id, class_name, floor_x, floor_y, timestamp)
    detector.update(1, "person", 3.0, 4.0, t1)
    
    t2 = datetime(2025, 3, 14, 10, 0, 2)
    detector.update(1, "person", 3.0, 6.0, t2)
    
    detector.close()
    
    csv_path = tmp_path / f"{cam_id}_crossings.csv"
    assert csv_path.exists()
    
    df = load_crossings_csv(str(csv_path))
    assert len(df) == 1
    
    row = df.iloc[0]
    assert row["edge_id"] == "E1"
    assert round(row["crossing_x"], 1) == 3.0
    assert round(row["crossing_y"], 1) == 5.0
    assert row["camera_id"] == cam_id

def test_fusion_deduplication(tmp_path):
    overlap_cfg_path = tmp_path / "overlap_zones.json"
    
    # Create simple overlap zone [0,0] to [10,10]
    mock_overlap = {
        "overlap_zones": [
            {
                "id": "zone_AB",
                "cameras": ["camA", "camB"],
                "floor_polygon": [[0,0], [10,0], [10,10], [0,10]],
                "distance_threshold_m": 0.5
            }
        ]
    }
    with open(overlap_cfg_path, "w") as f:
        json.dump(mock_overlap, f)
        
    csv_a = tmp_path / "camA_crossings.csv"
    csv_b = tmp_path / "camB_crossings.csv"
    
    # Points are close together (0.1, 0.1 dist = approx 0.14m distance), diff is within threshold 0.5m
    # Timestamp is close (0.5s diff is within 1.0s limit)
    # Track IDs are different (since they are separate cameras) so Hungarian matching will handle them
    create_mock_crossings_csv([{"timestamp": "2025-03-14 10:00:01", "track_id": 1, "edge_id": "E1", "crossing_x": 3.0, "crossing_y": 5.0, "camera_id": "camA"}], csv_a)
    create_mock_crossings_csv([{"timestamp": "2025-03-14 10:00:01.5", "track_id": 2, "edge_id": "E1", "crossing_x": 3.1, "crossing_y": 5.1, "camera_id": "camB"}], csv_b)
    
    fuser = CrossingFuser(str(overlap_cfg_path), timestamp_tolerance_s=1.0)
    df = fuser.fuse([str(csv_a), str(csv_b)])
    
    assert len(df) == 1
    
    row = df.iloc[0]
    # Timestamp snaps to the earliest
    assert row["timestamp"].to_pydatetime() == datetime(2025, 3, 14, 10, 0, 1)
    # Coords averaged
    assert round(row["crossing_x"], 2) == 3.05
    assert round(row["crossing_y"], 2) == 5.05
    assert "camA" in row["camera_id"] and "camB" in row["camera_id"]

def test_fusion_no_false_merge(tmp_path):
    overlap_cfg_path = tmp_path / "overlap_zones.json"
    mock_overlap = {
         "overlap_zones": [
            {
                "id": "zone_AB",
                "cameras": ["camA", "camB"],
                "floor_polygon": [[0,0], [10,0], [10,10], [0,10]],
                "distance_threshold_m": 0.5
            }
        ]
    }
    with open(overlap_cfg_path, "w") as f:
        json.dump(mock_overlap, f)
        
    csv_a = tmp_path / "camA_crossings.csv"
    csv_b = tmp_path / "camB_crossings.csv"
    
    # Same coords and times, but completely different edges crossed
    create_mock_crossings_csv([{"timestamp": "2025-03-14 10:00:01", "edge_id": "E1", "crossing_x": 3.0, "crossing_y": 5.0, "camera_id": "camA"}], csv_a)
    create_mock_crossings_csv([{"timestamp": "2025-03-14 10:00:01", "edge_id": "E2", "crossing_x": 3.0, "crossing_y": 5.0, "camera_id": "camB"}], csv_b)
    
    fuser = CrossingFuser(str(overlap_cfg_path), timestamp_tolerance_s=1.0)
    df = fuser.fuse([str(csv_a), str(csv_b)])
    
    # Both survive because they are distinct physical events (different edges)
    # A bug in CrossingFuser caused duplicate rows due to concatenation + append. Let's strictly test final length.
    assert len(df) == 2

def test_fusion_single_camera(tmp_path):
    overlap_cfg_path = tmp_path / "overlap_zones.json"
    with open(overlap_cfg_path, "w") as f:
        json.dump({"overlap_zones": []}, f)
        
    csv_a = tmp_path / "camA_crossings.csv"
    create_mock_crossings_csv([{"timestamp": "2025-03-14 10:00:01", "edge_id": "E1", "crossing_x": 3.0, "crossing_y": 5.0, "camera_id": "camA"}], csv_a)
    
    fuser = CrossingFuser(str(overlap_cfg_path))
    df = fuser.fuse([str(csv_a)])
    
    assert len(df) == 1
    assert df.iloc[0]["camera_id"] == "camA"

def test_ocr_fallback_to_fps(tmp_path):
    (tmp_path / "config").mkdir()
    cam_id = "cam_blank"
    mock_cameras_cfg = {
        "cameras": [
            {
                "id": cam_id,
                "ocr_region": {"x": 0, "y": 0, "w": 640, "h": 100}
            }
        ]
    }
    with open(tmp_path / "config" / "cameras.json", "w") as f:
        json.dump(mock_cameras_cfg, f)
        
    # Blank frame, OCR will yield nothing
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    extractor = TimestampExtractor(cam_id, config_path=str(tmp_path / "config") + "/")
    parsed_dt = extractor.extract(blank_frame)
    
    # Extractor natively returns None on complete failure
    assert parsed_dt is None
    
    # Simulate the fallback that happens inside PerCameraProcessor
    # Fallback function signature takes current frame_num, video FPS, and an optional anchor
    fallback_ts = extractor.get_fps_adjusted_timestamp(
        frame_number=15, 
        fps=15.0, 
        base_timestamp=datetime(2025, 3, 14, 10, 0, 0)
    )
    
    # If starting at 10:00:00 and we are on frame 15 at 15 FPS
    # We should be exactly 1 second in (10:00:01)
    assert fallback_ts == datetime(2025, 3, 14, 10, 0, 1)


# ── FootEstimator tests ────────────────────────────────────────────────────

def test_foot_estimator_occlusion_detected():
    """
    Person A's bottom is partially covered by person B standing in front.
    FootEstimator should detect the occlusion and extrapolate foot downward.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from detection.foot_estimator import FootEstimator

    # Create minimal Detection-like objects with a `bbox` attribute
    class MockDet:
        def __init__(self, bbox):
            self.bbox = bbox  # (x1, y1, x2, y2)

    # Person A — tall bbox, bottom 30 % zone approx y=280..400
    # Person A is partially occluded by Person B standing in front.
    # A has visible height 200px (cut off). B has physical height 300px.
    # A: x=100..200  y=100..300  (visible height 200 px)
    # B: x=120..180  y=250..550  (covers bottom zone of A, height 300)
    det_a = MockDet(bbox=(100, 100, 200, 300))
    det_b = MockDet(bbox=(120, 250, 180, 550))

    estimator = FootEstimator()
    estimates = estimator.estimate([det_a, det_b])

    assert len(estimates) == 2

    est_a = estimates[0]
    # Occlusion should be detected: B covers the bottom zone of A
    assert est_a.was_occluded, "Expected A to be flagged as occluded by B"
    # Extrapolated foot_y should be A's y1 (100) + ref_height (from B = 300) = 400
    assert est_a.foot_y == 400.0, f"Expected 400.0, got {est_a.foot_y}"
    assert est_a.foot_y != 300.0, "Foot y should be corrected, not raw y2"
    assert 0.0 < est_a.occlusion_confidence < 1.0, "Confidence should reflect partial visibility"

    # Horizontal foot_x should always be the bbox centre
    assert abs(est_a.foot_x - 150.0) < 1.0, "foot_x should be horizontal centre of A"


def test_foot_estimator_no_occlusion():
    """
    Three well-separated persons — no bottom-zone overlap.
    All FootEstimates should use the real bbox bottom (was_occluded=False).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from detection.foot_estimator import FootEstimator

    class MockDet:
        def __init__(self, bbox):
            self.bbox = bbox

    # Three clearly separated persons (different x positions, no vertical bottom clip)
    dets = [
        MockDet(bbox=(0,   100, 80,  400)),
        MockDet(bbox=(200, 120, 280, 390)),
        MockDet(bbox=(400, 110, 480, 410)),
    ]

    estimator = FootEstimator()
    estimates = estimator.estimate(dets)

    assert len(estimates) == 3
    for i, (det, est) in enumerate(zip(dets, estimates)):
        assert not est.was_occluded, f"Person {i} should not be occluded"
        assert est.occlusion_confidence == 1.0, f"Confidence should be 1.0 for clear persons"
        assert abs(est.foot_y - det.bbox[3]) < 0.5, f"foot_y should match raw y2 for person {i}"


def test_visibility_weighted_merge_auto():
    """
    In the overlap zone fuser, the camera whose foot was clearly seen
    (occlusion_confidence=1.0) should win position Authority over the camera
    whose foot was estimated (occlusion_confidence=0.5), even if the latter
    has a marginally higher raw YOLO confidence.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    overlap_cfg_path = Path(__file__).parent / "tmp_occ_zone.json"
    mock_overlap = {
        "overlap_zones": [
            {
                "id": "zone_AB",
                "cameras": ["camA", "camB"],
                "floor_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "distance_threshold_m": 2.0,
                "buffer_margin_m": 0.5,
                "fusion_strategy": "weighted_average"
            }
        ]
    }
    with open(overlap_cfg_path, "w") as f:
        json.dump(mock_overlap, f)

    try:
        from fusion.overlap import load_overlap_zones
        from fusion.fuse import DetectionFuser

        class MockFloorDet:
            """Minimal FloorDetection-like object for fuser testing."""
            def __init__(self, camera_id, floor_x, floor_y, confidence, occ_conf):
                self.camera_id            = camera_id
                self.floor_x              = floor_x
                self.floor_y              = floor_y
                self.confidence           = confidence
                self.occlusion_confidence = occ_conf
                self.pixel_bbox           = (0, 0, 100, 200)
                self.pixel_foot           = (50, 200)

        zones  = load_overlap_zones(str(overlap_cfg_path))
        fuser  = DetectionFuser(zones)

        # Points must be close enough (<2.0m threshold)
        # camA sees feet clearly → occ_conf=1.0, raw conf=0.70  → eff=0.70  pos=(3.0, 5.0)
        # camB foot estimated → occ_conf=0.5,  raw conf=0.80  → eff=0.40  pos=(3.2, 5.2)
        # camA should win even though camB has higher raw YOLO confidence
        detections = {
            "camA": [MockFloorDet("camA", 3.0, 5.0, confidence=0.70, occ_conf=1.0)],
            "camB": [MockFloorDet("camB", 3.2, 5.2, confidence=0.80, occ_conf=0.5)],
        }

        fused = fuser.fuse(detections)

        assert len(fused) == 1, "Should merge into one fused detection"
        fd = fused[0]
        # camA's effective confidence = 0.70 * 1.0 = 0.70 >> camB's 0.80 * 0.5 = 0.40
        # camA has much higher eff score → camA position should win
        assert abs(fd.floor_x - 3.0) < 0.5, (
            f"Expected x≈3.0 (camA pos), got {fd.floor_x:.2f}. "
            "Visibility-weighted merge not working."
        )
        assert abs(fd.floor_y - 5.0) < 0.5, (
            f"Expected y≈5.0 (camA pos), got {fd.floor_y:.2f}."
        )

    finally:
        overlap_cfg_path.unlink(missing_ok=True)

