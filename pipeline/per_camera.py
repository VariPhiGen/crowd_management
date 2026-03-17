"""
pipeline/per_camera.py
======================
Orchestrates the per-camera processing pipeline:
- Opens the video source for a specific camera.
- Extracts OCR timestamps frame-by-frame (with fallback).
- Detects and tracks persons using YOLO (ByteTrack).
- Undistorts the frames if intrinsics are loaded.
- Projects tracked foot-coordinates to the floor plane via Homography.
- Detects line crossings and logs them to a per-camera CSV file.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Callable

import cv2

from calibration.homography import HomographyMapper
from calibration.lens_correction import LensCorrector
from calibration.ocr_timestamp import TimestampExtractor
from detection.detector import PersonDetector
from fusion.crossing import LineCrossingDetector
from pipeline.s3_source import S3VideoSource, _is_s3_uri

logger = logging.getLogger(__name__)


class PerCameraProcessor:
    """
    Orchestrates the processing pipeline for a single camera.
    Reads the video sequentially and extracts crossing events to a CSV.
    """

    def __init__(
        self,
        camera_id: str,
        config_dir: str,
        output_dir: str,
        model: PersonDetector,
        append_output: bool = False,
    ) -> None:
        self.camera_id = camera_id
        self.append_output = append_output
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.model = model

        # Load camera config
        cameras_cfg_path = self.config_dir / "cameras.json"
        if not cameras_cfg_path.exists():
            raise FileNotFoundError(f"Missing {cameras_cfg_path}")
            
        with open(cameras_cfg_path) as f:
            cameras_data = json.load(f)

        self.camera_config = next(
            (c for c in cameras_data.get("cameras", []) if c["id"] == camera_id), None
        )
        if self.camera_config is None:
            raise ValueError(f"Camera '{camera_id}' not found in cameras.json")

        source_str = self.camera_config["source"]

        # ── S3 source detection ──────────────────────────────────────────────
        self._s3_source: Optional[S3VideoSource] = None

        if _is_s3_uri(source_str):
            self._s3_source = S3VideoSource(
                s3_uri    = source_str,
                camera_id = camera_id,
                tmp_root  = str(self.output_dir / "tmp_s3"),
                config_dir= str(self.config_dir),
            )
            # video_paths is a placeholder; actual paths come from iter_videos()
            self.video_paths = []          # populated lazily during process_video
            self.source      = Path(self._s3_source.tmp_dir)
        else:
            # ── Local path (original behaviour) ─────────────────────────────
            self.source = Path(source_str)
            if self.source.is_dir():
                self.video_paths = sorted(
                    [p for p in self.source.iterdir()
                     if p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]]
                )
            else:
                self.video_paths = [self.source]

        # ── Initialize Sub-Modules ───────────────────────────────────────────

        # Per-camera track_point — read from cameras.json; fall back to "bottom"
        self._track_point = self.camera_config.get("track_point", "bottom")

        # 1. Lens Correction
        self.lens_corrector = LensCorrector(self.camera_id, str(self.config_dir) + "/")
        
        # 2. Homography Mapper
        self.homography_mapper = HomographyMapper(self.camera_id, str(self.config_dir) + "/")
        
        # 3. Timestamp Extractor
        self.timestamp_extractor = TimestampExtractor(self.camera_id, str(self.config_dir) + "/")
        
        # 4. Line Crossing Detector
        edges_json = self.config_dir / "edges.json"
        self.crossing_detector = LineCrossingDetector(
            edges_config_path=str(edges_json),
            camera_id=self.camera_id,
            output_dir=str(self.output_dir),
            append=self.append_output,
        )

        # ── Video Info ───────────────────────────────────────────────────────
        self._total_frames = 0
        self._fps = 15.0
        self._width = 0
        self._height = 0
        
        valid_paths = []
        for vpath in self.video_paths:
            cap = cv2.VideoCapture(str(vpath))
            if cap.isOpened():
                valid_paths.append(vpath)
                self._fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
                self._total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
        self.video_paths = valid_paths
        
        if not self.video_paths and not _is_s3_uri(source_str):
            logger.error("[%s] Failed to open any video sources in: %s", self.camera_id, self.source)
            self._video_info = None
        elif _is_s3_uri(source_str) and len(self._s3_source) == 0:
            logger.error("[%s] Failed to find any S3 videos in: %s", self.camera_id, source_str)
            self._video_info = None
        else:
            self._video_info = {
                "fps": self._fps,
                "total_frames": self._total_frames,
                "resolution": (self._width, self._height),
                "duration_sec": self._total_frames / self._fps if self._fps > 0 else 0
            }

    def get_video_info(self) -> Optional[dict]:
        """Returns metadata about the video source, if accessible."""
        return self._video_info

    def process_video(
        self, 
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> str:
        """
        Processes the video frame-by-frame through the complete pipeline.

        Returns
        -------
        str
            The absolute path to the generated CSV output file.
        """
        is_s3 = self._s3_source is not None

        if not is_s3 and (self._video_info is None or not self.video_paths):
            raise RuntimeError(f"Cannot process video: source '{self.source}' not accessible.")

        if is_s3:
            # For S3 we don't know total frames upfront — log video count instead
            logger.info(
                "[%s] Starting S3 processing: %d video(s) @ ~%.1f FPS",
                self.camera_id, len(self._s3_source), self._fps,
            )
        else:
            logger.info(
                "[%s] Starting processing: %d total frames across %d files @ %.1f FPS",
                self.camera_id, self._total_frames, len(self.video_paths), self._fps,
            )
        
        frame_idx = 0
        ocr_failure_count = 0

        # ── Tracks CSV: continuous floor positions per person per frame ───────
        # Written alongside the crossings CSV.  Used by fusion for trajectory
        # matching across cameras — more robust than event-only deduplication.
        import csv as _csv
        tracks_csv_path = str(Path(self.crossing_detector.csv_path).parent /
                              f"{self.camera_id}_tracks.csv")
        _tracks_exists = os.path.exists(tracks_csv_path)
        _tracks_mode = "a" if (self.append_output and _tracks_exists) else "w"
        _tracks_file = open(tracks_csv_path, _tracks_mode, newline="")
        _tracks_writer = _csv.writer(_tracks_file)
        if _tracks_mode == "w":
            _tracks_writer.writerow(["timestamp", "track_id", "floor_x", "floor_y", "camera_id"])
        logger.info("[%s] Opened tracks CSV log: %s (mode=%s)", self.camera_id, tracks_csv_path, _tracks_mode)
        
        # ── Choose video iterator: S3 (download one at a time) or local list ──
        if is_s3:
            _video_iter = self._s3_source.iter_videos()
        else:
            _video_iter = iter(self.video_paths)

        for vpath in _video_iter:
            logger.info("[%s] Processing file: %s", self.camera_id, vpath.name)
            logger.info("[%s] Opening video (first frame read + YOLO first inference may take 30–60s)...", self.camera_id)
            self.cap = cv2.VideoCapture(str(vpath))
            if not self.cap.isOpened():
                logger.error("[%s] Failed to open video: %s", self.camera_id, vpath)
                continue

            # Update FPS from actual file when processing S3 videos
            if is_s3:
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if actual_fps > 0:
                    self._fps = actual_fps
            # Ensure the YOLO tracker is reset so IDs don't bleed across identical runs or different videos
            if self.model.model is not None:
                try:
                    self.model.model.reset_tracker()
                except Exception:
                    pass

            _frames_in_this_file = 0
            try:
                while True:
                    ret, raw_frame = self.cap.read()
                    if not ret or raw_frame is None:
                        break

                    # a) Undistort if calibrated
                    if self.lens_corrector.is_calibrated:
                        frame = self.lens_corrector.undistort_frame(raw_frame)
                    else:
                        frame = raw_frame

                    # b) Extract OCR Timestamp
                    # Will fallback internally if unreadable but needs _base_timestamp via get_fps_adjusted_timestamp
                    ts_dt = self.timestamp_extractor.extract(frame)
                    
                    if ts_dt is None:
                        ocr_failure_count += 1
                        ts_dt = self.timestamp_extractor.get_fps_adjusted_timestamp(
                            frame_number=frame_idx, fps=self._fps
                        )
                        
                        if ts_dt is None:
                            # Hard fallback to a fixed Epoch if OCR is totally broken to guarantee sync
                            if not hasattr(self, '_fallback_base'):
                                self._fallback_base = datetime(2025, 1, 1, 12, 0, 0)
                            from datetime import timedelta
                            ts_dt = self._fallback_base + timedelta(seconds=frame_idx / max(1.0, self._fps))

                    # c) Detect and Track
                    # Using model.track to maintain stable ByteTrack IDs
                    detections = self.model.track(frame)
                    
                    if detections and self.homography_mapper.is_calibrated:
                        # Collect points
                        import numpy as np
                        pts = np.array([d.foot_point for d in detections], dtype=np.float64)
                        
                        # Prevent double-undistortion if already undistorted
                        already_undistorted = self.lens_corrector.is_calibrated
                        
                        floor_coords = self.homography_mapper.pixel_to_floor_batch(
                            pts, already_undistorted=already_undistorted
                        )
                        
                        if floor_coords is not None:
                            for det, (floor_x, floor_y) in zip(detections, floor_coords):
                                if det.track_id < 0:
                                    continue # ensure tracked
                                    
                                # Convert classes appropriately, YOLOv8 0=person
                                class_name = "person" if det.class_id == 0 else f"class_{det.class_id}"
                                
                                # Update crossing detector
                                self.crossing_detector.update(
                                    track_id=det.track_id,
                                    class_name=class_name,
                                    floor_x=float(floor_x),
                                    floor_y=float(floor_y),
                                    timestamp=ts_dt
                                )

                                # Write raw floor position to tracks CSV
                                _tracks_writer.writerow([
                                    ts_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
                                    det.track_id,
                                    round(float(floor_x), 3),
                                    round(float(floor_y), 3),
                                    self.camera_id,
                                ])

                    frame_idx += 1
                    _frames_in_this_file += 1
                    # First frame done = past slow YOLO init
                    if _frames_in_this_file == 1:
                        logger.info("[%s] First frame processed (pipeline warm).", self.camera_id)
                    # When total frames unknown (S3), log every 300 frames so run doesn't look stuck
                    if self._total_frames == 0 and frame_idx > 0 and frame_idx % 300 == 0:
                        logger.info("[%s] Frames processed: %d", self.camera_id, frame_idx)
                    
                    if progress_callback is not None:
                        progress_callback(frame_idx, self._total_frames)

            finally:
                self.cap.release()
                # ── S3: delete local copy immediately to free disk space ──
                if is_s3:
                    self._s3_source.delete_local(vpath)
        
        self.crossing_detector.close()
        _tracks_file.close()
        logger.info("[%s] Closed tracks CSV: %s", self.camera_id, tracks_csv_path)

        # Check OCR performance
        if frame_idx > 0:
            failure_rate = ocr_failure_count / frame_idx
            if failure_rate > 0.3:
                logger.warning(
                    "[%s] High OCR failure rate: %.1f%% (%d/%d frames). "
                    "Consider re-running '--ocr-region %s' to adjust the bounding box.",
                    self.camera_id, failure_rate * 100, ocr_failure_count, frame_idx, self.camera_id
                )

        logger.info("[%s] Finished processing %d frames.", self.camera_id, frame_idx)
        return self.crossing_detector.csv_path


class MultiCameraRunner:
    """
    Runs the pipeline over multiple cameras in the configuration.
    """

    def __init__(
        self,
        config_dir: str,
        output_dir: str,
        model_path: str = "yolov8n.pt",
        append_output: bool = False,
    ) -> None:
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.append_output = append_output
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        cameras_cfg_path = self.config_dir / "cameras.json"
        with open(cameras_cfg_path) as f:
            cameras_data = json.load(f)
            
        self.cameras = cameras_data.get("cameras", [])
        
        logger.info(
            "Initializing multi-camera runner with model: %s (append_output=%s)",
            model_path, append_output,
        )
        self.detector = PersonDetector(model_name=model_path)

    def run_all(self, sequential: bool = True) -> List[str]:
        """
        Process all configured cameras.
        
        Parameters
        ----------
        sequential : bool
            If true, process one camera entirely before starting the next.
            Minimizes VRAM footprints. If false, not implemented yet (requires
            complex frame interleaving or multiprocessing).
            
        Returns
        -------
        List[str]
            List of generated CSV output paths.
        """
        start_time = time.time()
        output_paths: List[str] = []
        
        if not sequential:
            logger.warning("Parallel processing not implemented. Defaulting to sequential.")
            sequential = True

        for cam in self.cameras:
            cam_id = cam["id"]
            
            logger.info("=" * 60)
            logger.info("Processing Camera: %s", cam_id)
            logger.info("=" * 60)
            
            try:
                processor = PerCameraProcessor(
                    camera_id=cam_id,
                    config_dir=str(self.config_dir),
                    output_dir=str(self.output_dir),
                    model=self.detector,
                    append_output=self.append_output,
                )
                
                if processor.get_video_info() is None:
                    logger.error("[%s] Skipping due to source initialization failure.", cam_id)
                    continue

                def _progress(cur: int, tot: int) -> None:
                    # Simple logging progression every 10% or so
                    if tot > 0 and cur % max(1, tot // 10) == 0:
                        pct = (cur / tot) * 100
                        logger.info("  [%s] Progress: %d / %d (%.0f%%)", cam_id, cur, tot, pct)

                csv_path = processor.process_video(progress_callback=_progress)
                output_paths.append(csv_path)
                
            except Exception as e:
                logger.exception("[%s] Unexpected error during pipeline: %s", cam_id, e)
                
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Multi-Camera Pipeline Finished in %.1f seconds", elapsed)
        logger.info("Files generated: %s", output_paths)
        logger.info("=" * 60)

        return output_paths
