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
from detection.detector import PersonDetector, COCO_CLASS_NAMES
from fusion.crossing import LineCrossingDetector
from pipeline.s3_source import S3VideoSource, _is_s3_uri

# ── Parallel camera processing ───────────────────────────────────────────────
# Worker entry-point (module-level so it is picklable).
# Each thread gets its own PersonDetector so ByteTrack state stays isolated.

def _run_camera_in_thread(args: tuple) -> str:
    """Called by ThreadPoolExecutor — one camera per thread."""
    (
        camera_id, config_dir, output_dir,
        model_name, model_confidence, model_classes,
        model_track_point,
        append_output, frame_stride, ocr_interval,
    ) = args

    # Per-thread model — own detector + ByteTrack state, own CUDA allocation
    cam_model = PersonDetector(
        model_name     = model_name,
        confidence     = model_confidence,
        target_classes = model_classes,
        track_point    = model_track_point,
    )

    processor = PerCameraProcessor(
        camera_id          = camera_id,
        config_dir         = config_dir,
        output_dir         = output_dir,
        model              = cam_model,
        append_output      = append_output,
        track_point_override = model_track_point,
    )
    return processor.process_video(
        frame_stride = frame_stride,
        ocr_interval = ocr_interval,
    )

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
        track_point_override: Optional[str] = None,
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

        # Per-camera track_point — CLI/UI override > cameras.json > default "bottom"
        self._track_point = (
            track_point_override
            if track_point_override is not None
            else self.camera_config.get("track_point", "bottom")
        )

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
        progress_callback: Optional[Callable[[int, int], None]] = None,
        frame_stride: int = 1,
        ocr_interval: int = 0,
    ) -> str:
        """
        Processes the video frame-by-frame through the complete pipeline.

        Parameters
        ----------
        progress_callback : callable(current_frame, total_frames), optional
        frame_stride : int
            Process every Nth frame.  Frames between strides are read but
            discarded (frame counter advances so timestamps stay accurate).
            Default 1 (every frame).  Set to 2 for a ~2x YOLO speedup with
            negligible tracking accuracy loss (ByteTrack Kalman predicts gaps).
        ocr_interval : int
            Run EasyOCR every N *video* frames.  0 = auto (once per second of
            video = fps).  EasyOCR is the pipeline's slowest component; calling
            it once per second instead of every frame gives a 10–15x speedup
            for a workload where the timestamp changes only once per second.

        Returns
        -------
        str
            The absolute path to the generated CSV output file.
        """
        import csv as _csv
        import numpy as np
        from datetime import timedelta

        is_s3 = self._s3_source is not None

        if not is_s3 and (self._video_info is None or not self.video_paths):
            raise RuntimeError(f"Cannot process video: source '{self.source}' not accessible.")

        # Resolve OCR interval — default: once per second of video
        _ocr_interval = ocr_interval if ocr_interval > 0 else max(1, int(round(self._fps)))

        if is_s3:
            logger.info(
                "[%s] Starting S3 processing: %d video(s) @ ~%.1f FPS  "
                "(stride=%d, ocr_every=%d frames)",
                self.camera_id, len(self._s3_source), self._fps,
                frame_stride, _ocr_interval,
            )
        else:
            logger.info(
                "[%s] Starting processing: %d total frames across %d files @ %.1f FPS  "
                "(stride=%d, ocr_every=%d frames)",
                self.camera_id, self._total_frames, len(self.video_paths), self._fps,
                frame_stride, _ocr_interval,
            )

        frame_idx = 0          # total frames read (includes skipped) — used for timestamps
        processed_count = 0    # frames actually run through YOLO
        ocr_call_count = 0
        ocr_failure_count = 0
        _last_ocr_ts: Optional[datetime] = None
        _last_ocr_frame_idx: int = 0   # frame_idx at which last OCR succeeded
        _t_start = time.time()

        # ── Tracks CSV ────────────────────────────────────────────────────────
        tracks_csv_path = str(Path(self.crossing_detector.csv_path).parent /
                              f"{self.camera_id}_tracks.csv")
        _tracks_exists = os.path.exists(tracks_csv_path)
        _tracks_mode = "a" if (self.append_output and _tracks_exists) else "w"
        _tracks_file = open(tracks_csv_path, _tracks_mode, newline="")
        _tracks_writer = _csv.writer(_tracks_file)
        if _tracks_mode == "w":
            _tracks_writer.writerow(["timestamp", "track_id", "floor_x", "floor_y", "camera_id"])
        logger.info("[%s] Opened tracks CSV log: %s (mode=%s)",
                    self.camera_id, tracks_csv_path, _tracks_mode)

        # ── Video iterator: S3 with prefetch, or local list ──────────────────
        if is_s3:
            _video_iter = self._s3_source.iter_videos_prefetch(ahead=1)
        else:
            _video_iter = iter(self.video_paths)

        for vpath in _video_iter:
            # S3 download failures surface as the value of the future —
            # the prefetch thread raises RuntimeError; catch it here so
            # one bad file doesn't abort ALL remaining files for this camera.
            if isinstance(vpath, Exception):
                logger.error("[%s] Skipping file — download error: %s", self.camera_id, vpath)
                continue

            logger.info("[%s] Processing file: %s", self.camera_id, vpath.name)
            logger.info(
                "[%s] Opening video (first frame + YOLO warm-up may take 30–60 s)…",
                self.camera_id,
            )
            self.cap = cv2.VideoCapture(str(vpath))
            if not self.cap.isOpened():
                logger.error("[%s] Failed to open video: %s", self.camera_id, vpath)
                continue

            if is_s3:
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if actual_fps > 0:
                    self._fps = actual_fps
                    # Re-resolve ocr_interval with actual fps
                    if ocr_interval == 0:
                        _ocr_interval = max(1, int(round(self._fps)))

            # Reset ByteTrack per file so IDs don't bleed across videos
            if self.model.model is not None:
                try:
                    self.model.model.reset_tracker()
                except Exception:
                    pass

            _frames_in_file = 0
            try:
                while True:
                    ret, raw_frame = self.cap.read()
                    if not ret or raw_frame is None:
                        break

                    frame_idx += 1
                    _frames_in_file += 1

                    # ── Frame stride: skip intermediate frames ────────────────
                    # frame_idx still advances so FPS-based timestamps are accurate.
                    if frame_stride > 1 and (_frames_in_file % frame_stride) != 1:
                        continue

                    # ── Undistort ────────────────────────────────────────────
                    frame = (self.lens_corrector.undistort_frame(raw_frame)
                             if self.lens_corrector.is_calibrated else raw_frame)

                    # ── OCR Timestamp (throttled) ────────────────────────────
                    # Always run OCR on the FIRST frame of each file (picks up
                    # the new day/time even when videos span different days).
                    # Otherwise run every _ocr_interval frames.
                    # Between OCR calls, extrapolate from the LAST known OCR
                    # timestamp rather than a hardcoded fallback date.
                    _run_ocr = (_frames_in_file == 1) or (frame_idx % _ocr_interval == 1)

                    if _run_ocr:
                        ts_dt = self.timestamp_extractor.extract(frame)
                        ocr_call_count += 1
                        if ts_dt is not None:
                            _last_ocr_ts = ts_dt
                            _last_ocr_frame_idx = frame_idx
                        else:
                            ocr_failure_count += 1
                            ts_dt = (
                                _last_ocr_ts + timedelta(
                                    seconds=(frame_idx - _last_ocr_frame_idx) / max(1.0, self._fps)
                                ) if _last_ocr_ts is not None else None
                            )
                    else:
                        ts_dt = (
                            _last_ocr_ts + timedelta(
                                seconds=(frame_idx - _last_ocr_frame_idx) / max(1.0, self._fps)
                            ) if _last_ocr_ts is not None else None
                        )

                    # Absolute last-resort fallback (only if OCR has NEVER succeeded)
                    if ts_dt is None:
                        if not hasattr(self, '_fallback_base'):
                            self._fallback_base = datetime(2025, 1, 1, 12, 0, 0)
                        ts_dt = self._fallback_base + timedelta(
                            seconds=frame_idx / max(1.0, self._fps)
                        )

                    # ── YOLO track ───────────────────────────────────────────
                    detections = self.model.track(frame)
                    processed_count += 1

                    if _frames_in_file == 1:
                        logger.info("[%s] First frame processed (pipeline warm).", self.camera_id)

                    if detections and self.homography_mapper.is_calibrated:
                        pts = np.array(
                            [d.foot_point for d in detections], dtype=np.float64
                        )
                        already_undistorted = self.lens_corrector.is_calibrated
                        floor_coords = self.homography_mapper.pixel_to_floor_batch(
                            pts, already_undistorted=already_undistorted
                        )

                        if floor_coords is not None:
                            for det, (floor_x, floor_y) in zip(detections, floor_coords):
                                if det.track_id < 0:
                                    continue
                                class_name = COCO_CLASS_NAMES.get(
                                    det.class_id, f"class_{det.class_id}"
                                )
                                self.crossing_detector.update(
                                    track_id  = det.track_id,
                                    class_name= class_name,
                                    floor_x   = float(floor_x),
                                    floor_y   = float(floor_y),
                                    timestamp = ts_dt,
                                )
                                _tracks_writer.writerow([
                                    ts_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
                                    det.track_id,
                                    round(float(floor_x), 3),
                                    round(float(floor_y), 3),
                                    self.camera_id,
                                ])

                    # Periodic flush — ensures data reaches disk even if the
                    # process is killed before normal close() is called.
                    if processed_count % 1000 == 0:
                        _tracks_file.flush()
                        self.crossing_detector._csv_file.flush()

                    # Progress reporting
                    if self._total_frames == 0 and frame_idx % 500 == 0:
                        elapsed = time.time() - _t_start
                        fps_actual = processed_count / elapsed if elapsed > 0 else 0
                        logger.info(
                            "[%s] Frames read: %d  processed: %d  throughput: %.1f fps",
                            self.camera_id, frame_idx, processed_count, fps_actual,
                        )
                    if progress_callback is not None:
                        progress_callback(frame_idx, self._total_frames)

            finally:
                self.cap.release()
                # Clear stale track positions so ByteTrack's recycled IDs in
                # the NEXT video don't trigger false crossings on their first frame.
                self.crossing_detector.reset()
                if is_s3:
                    self._s3_source.delete_local(vpath)

        self.crossing_detector.close()
        _tracks_file.close()

        elapsed_total = time.time() - _t_start
        throughput = processed_count / elapsed_total if elapsed_total > 0 else 0
        logger.info(
            "[%s] Finished — read %d frames, processed %d (stride=%d), "
            "OCR calls %d / %d failures, throughput %.1f fps, wall time %.1f s",
            self.camera_id, frame_idx, processed_count, frame_stride,
            ocr_call_count, ocr_failure_count, throughput, elapsed_total,
        )

        if ocr_call_count > 0 and (ocr_failure_count / ocr_call_count) > 0.3:
            logger.warning(
                "[%s] High OCR failure rate: %.1f%% (%d/%d calls). "
                "Consider adjusting '--ocr-region %s'.",
                self.camera_id, ocr_failure_count / ocr_call_count * 100,
                ocr_failure_count, ocr_call_count, self.camera_id,
            )

        return self.crossing_detector.csv_path


class MultiCameraRunner:
    """
    Runs the pipeline over multiple cameras, optionally in parallel.

    Parallel mode uses one OS thread per camera.  Each thread owns its own
    PersonDetector (separate YOLO weights + ByteTrack state).  PyTorch
    releases the GIL during GPU inference so threads genuinely overlap on
    the A10G — 4 cameras running simultaneously is well within the 23 GB
    VRAM budget (~1.5 GB × 4 ≈ 6 GB).
    """

    def __init__(
        self,
        config_dir: str,
        output_dir: str,
        model_path: str = "yolov8n.pt",
        append_output: bool = False,
        target_classes: "str | list[int] | None" = None,
        confidence: float = 0.50,
        track_point: str = "bottom",
    ) -> None:
        self.config_dir    = Path(config_dir)
        self.output_dir    = Path(output_dir)
        self.append_output = append_output
        self.track_point   = track_point
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cameras_cfg_path = self.config_dir / "cameras.json"
        with open(cameras_cfg_path) as f:
            cameras_data = json.load(f)

        self.cameras = cameras_data.get("cameras", [])

        # Reference detector carries model_name / confidence / track_point
        # for worker threads — each thread creates its own instance.
        logger.info(
            "MultiCameraRunner: %d cameras, model=%s, conf=%.2f, track_point=%s, append=%s",
            len(self.cameras), model_path, confidence, track_point, append_output,
        )
        self.detector = PersonDetector(
            model_name     = model_path,
            target_classes = target_classes,
            confidence     = confidence,
            track_point    = track_point,
        )

    def run_all(
        self,
        sequential: bool = False,
        max_workers: int = 4,
        frame_stride: int = 1,
        ocr_interval: int = 0,
    ) -> List[str]:
        """
        Process all configured cameras.

        Parameters
        ----------
        sequential : bool
            Force sequential (one-at-a-time) processing. Useful for debugging.
            Default False (parallel).
        max_workers : int
            Maximum cameras to run simultaneously.  Default 4.
            On an A10G (23 GB) with YOLOv8m each camera uses ~1.5 GB VRAM,
            so up to 12 cameras can run in parallel — keep at 4 for headroom.
        frame_stride : int
            Passed to each PerCameraProcessor.process_video().
            1 = every frame (default); 2 = every other frame (~2× faster YOLO).
        ocr_interval : int
            Passed to each PerCameraProcessor.process_video().
            0 = auto (once per second of video); recommended for long recordings.

        Returns
        -------
        List[str]
            CSV output paths, one per successfully processed camera.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        _t0 = time.time()
        output_paths: List[str] = []

        _workers = 1 if sequential else min(max_workers, len(self.cameras))

        # ── GPU VRAM pre-flight check ─────────────────────────────────────
        # YOLOv8m ≈ 1.5 GB + EasyOCR ≈ 0.4 GB + ByteTrack overhead ≈ 0.1 GB
        # per camera thread. Warn early rather than OOM mid-run.
        try:
            import torch
            if torch.cuda.is_available():
                free_vram_gb = (torch.cuda.get_device_properties(0).total_memory
                                - torch.cuda.memory_reserved(0)) / 1e9
                per_worker_gb = 2.1          # conservative estimate
                needed_gb     = per_worker_gb * _workers
                if needed_gb > free_vram_gb * 0.9:
                    logger.warning(
                        "GPU VRAM warning: %d workers × %.1f GB ≈ %.1f GB needed, "
                        "but only %.1f GB free on GPU. Consider reducing --workers "
                        "to %d to stay under 90%% VRAM.",
                        _workers, per_worker_gb, needed_gb, free_vram_gb,
                        max(1, int(free_vram_gb * 0.9 / per_worker_gb)),
                    )
                else:
                    logger.info(
                        "GPU VRAM: %.1f GB free — %d workers × %.1f GB = %.1f GB needed. OK.",
                        free_vram_gb, _workers, per_worker_gb, needed_gb,
                    )
        except Exception:
            pass  # non-critical check

        # ── Disk space pre-flight check ───────────────────────────────────
        try:
            import shutil as _shutil
            _, _, free_bytes = _shutil.disk_usage(str(self.output_dir))
            free_gb = free_bytes / 1e9
            # Rough estimate: tracks CSV ~50 bytes/row × 7.5 fps × 20 dets × cameras × hours
            # For 24h run, safe to warn if < 20 GB free
            if free_gb < 20:
                logger.warning(
                    "Disk space warning: only %.1f GB free in %s. "
                    "Tracks CSVs for a 24h 9-camera run can reach ~5-10 GB.",
                    free_gb, self.output_dir,
                )
            else:
                logger.info("Disk space: %.1f GB free in %s. OK.", free_gb, self.output_dir)
        except Exception:
            pass  # non-critical check

        logger.info(
            "run_all: %d cameras, workers=%d, stride=%d, ocr_interval=%d",
            len(self.cameras), _workers, frame_stride, ocr_interval,
        )

        # Build argument tuples for worker threads
        task_args = [
            (
                cam["id"],
                str(self.config_dir),
                str(self.output_dir),
                self.detector.model_name,
                self.detector.confidence,
                self.detector.target_classes,   # pass resolved class IDs
                self.track_point,
                self.append_output,
                frame_stride,
                ocr_interval,
            )
            for cam in self.cameras
        ]

        with ThreadPoolExecutor(
            max_workers=_workers,
            thread_name_prefix="cam_worker",
        ) as pool:
            future_to_cam = {
                pool.submit(_run_camera_in_thread, args): args[0]
                for args in task_args
            }
            for future in as_completed(future_to_cam):
                cam_id = future_to_cam[future]
                try:
                    csv_path = future.result()
                    if csv_path:
                        output_paths.append(csv_path)
                        logger.info("[%s] ✓ Done → %s", cam_id, csv_path)
                except Exception:
                    logger.exception("[%s] Pipeline failed", cam_id)

        elapsed = time.time() - _t0
        logger.info(
            "=" * 60 + "\nAll %d cameras finished in %.1f s (%.1f min)\n"
            "Outputs: %s\n" + "=" * 60,
            len(self.cameras), elapsed, elapsed / 60, output_paths,
        )
        return output_paths
