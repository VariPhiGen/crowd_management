import json
import csv
from pathlib import Path

import numpy as np
import cv2

def create_synthetic_frame(timestamp_str: str, text_position: tuple[int, int], frame_size: tuple[int, int] = (640, 480)):
    """
    Generate a black image with white text drawn at the given position.
    """
    w, h = frame_size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    
    # We use a standard OpenCV font. Size 1.0, Thickness 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    color = (255, 255, 255)
    thickness = 2
    
    # putText expects bottom-left corner of the text string
    # text_position is (x, y)
    cv2.putText(frame, timestamp_str, text_position, font, font_scale, color, thickness)
    
    return frame

def create_mock_edges_config(edges_list: list[dict], output_path: Path):
    """
    Creates a mock edges.json file. 
    `edges_list` format: [{"id": "E1", "line": [[0, 5], [10, 5]]}]
    """
    cfg = {"edges": edges_list}
    with open(output_path, "w") as f:
        json.dump(cfg, f)

def create_mock_crossings_csv(events_list: list[dict], output_path: Path):
    """
    Creates a mock crossings.csv file.
    `events_list` is a list of dicts.
    """
    
    fieldnames = ["timestamp", "track_id", "class_name", "edge_id", "direction", "crossing_x", "crossing_y", "camera_id"]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for e in events_list:
            row = {
                "timestamp": e.get("timestamp"),
                "track_id": e.get("track_id", 1),
                "class_name": e.get("class_name", "person"),
                "edge_id": e.get("edge_id"),
                "direction": e.get("direction", "forward"),
                "crossing_x": e.get("crossing_x"),
                "crossing_y": e.get("crossing_y"),
                "camera_id": e.get("camera_id")
            }
            writer.writerow(row)
