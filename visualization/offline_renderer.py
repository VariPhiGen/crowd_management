"""
offline_renderer.py

Provides the `CsvVisualizer` class, which replays the generated `crossings.csv`
or `fused_crossings.csv` files using the `FloorRenderer` to simulate
a top-down visualization temporally.
"""

import cv2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging

from visualization.floor_renderer import FloorRenderer
from fusion.fuse import FusedDetection

logger = logging.getLogger(__name__)


class CsvVisualizer:
    """
    Reads a line-crossing CSV (fused or per-camera) and plays it back
    visually over time. Since crossing events are instantaneous, this
    holds the detection on screen for a short 'persistence' duration.
    """

    def __init__(
        self,
        csv_path: str,
        floor_config: dict,
        cameras_config: dict,
        overlap_config: dict,
        playback_speed: float = 1.0,
        persistence_s: float = 1.0, # How long a dot stays visible after crossing
        output_video: str = None,   # Path to save headless MP4
    ):
        self.csv_path = csv_path
        self.playback_speed = playback_speed
        self.persistence_s = persistence_s
        self.output_video = output_video
        
        self.renderer = FloorRenderer(
            floor_config=floor_config,
            cameras_config=cameras_config,
            overlap_config=overlap_config,
            window_width=1200
        )
        
        self.df = None
        
    def _load_data(self) -> bool:
        try:
            self.df = pd.read_csv(self.csv_path)
            if 'timestamp' in self.df.columns:
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df.sort_values('timestamp', inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            
            logger.info("Loaded %d events from %s for playback.", len(self.df), self.csv_path)
            return True
        except Exception as e:
            logger.error("Failed to load CSV for visualization: %s", e)
            return False

    def run(self) -> bool:
        if not self._load_data() or self.df.empty:
            logger.warning("No data to visualize.")
            return False
            
        headless = bool(self.output_video)
        video_writer = None
        
        if headless:
            logger.info("Running headlessly. Capturing video to %s", self.output_video)
        else:
            cv2.namedWindow("Offline Playback", cv2.WINDOW_AUTOSIZE)
        
        start_time = self.df.iloc[0]['timestamp']
        end_time = self.df.iloc[-1]['timestamp']
        
        logger.info("Playback range: %s to %s", start_time, end_time)
        
        current_time = start_time
        
        # We simulate 30 FPS playback ticks
        fps = 30
        tick_delta_s = (1.0 / fps) * self.playback_speed
        tick_delta = timedelta(seconds=tick_delta_s)
        
        # Real-time pacing
        frame_delay_ms = int(1000 / fps)
        
        is_paused = False
        frame_count = 0
        
        while current_time <= end_time:
            # Handle UI controls (only if not headless)
            if not headless:
                key = cv2.waitKey(frame_delay_ms) & 0xFF
                if key == 27 or key == ord('q'): # ESC or q
                    break
                elif key == ord(' '): # Space to pause
                    is_paused = not is_paused
                    
            if is_paused and not headless:
                # Still process UI events so window doesn't hang, but don't advance time
                continue
                
            # Filter detections that occurred between (current_time - persistence) and current_time
            window_start = current_time - timedelta(seconds=self.persistence_s)
            
            # Efficient slice
            mask = (self.df['timestamp'] >= window_start) & (self.df['timestamp'] <= current_time)
            active_events = self.df[mask]
            
            # Convert to mock FusedDetections
            mock_detections = []
            fused_count = 0
            
            for _, row in active_events.iterrows():
                cam_id = str(row['camera_id'])
                is_fused = cam_id.startswith("fused:")
                if is_fused:
                    fused_count += 1
                    
                # The camera_id list just needs to have elements for rendering colors
                source_cams = cam_id.split("+") if is_fused else [cam_id]
                
                det = FusedDetection(
                    floor_x=float(row['crossing_x']),
                    floor_y=float(row['crossing_y']),
                    confidence=1.0, # CSV doesn't track confidence, assume 100%
                    source_cameras=source_cams,
                    is_fused=is_fused,
                    fusion_distance=0.0, # Assume 0.0 for replay
                    track_id=int(row['track_id']),
                )
                mock_detections.append(det)
                
            stats = {
                "total_persons": len(mock_detections),
                "fused_count": fused_count,
                "single_count": len(mock_detections) - fused_count,
            }
            
            frame = self.renderer.render(fused_detections=mock_detections, stats=stats)
            
            # Initialize VideoWriter on first frame if headless
            if headless and video_writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                video_writer = cv2.VideoWriter(self.output_video, fourcc, fps, (w, h))
            
            # Overlay current playback time on the frame strongly
            time_str = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            cv2.putText(
                frame, 
                f"Time: {time_str} | Speed: {self.playback_speed}x", 
                (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                frame, 
                f"Time: {time_str} | Speed: {self.playback_speed}x", 
                (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA
            )
            
            if is_paused and not headless:
                cv2.putText(frame, "PAUSED", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            
            if headless:
                video_writer.write(frame)
                frame_count += 1
                if frame_count % (fps * 5) == 0:  # Log every 5 seconds of video
                    logger.info("Rendered %d frames. Current time: %s", frame_count, time_str)
            else:
                cv2.imshow("Offline Playback", frame)
            
            # Advance time
            current_time += tick_delta
            
        if headless and video_writer is not None:
            video_writer.release()
            logger.info("Finished rendering video: %s (%d frames)", self.output_video, frame_count)
        elif not headless:
            cv2.destroyAllWindows()
            
        logger.info("Playback finished.")
        return True
