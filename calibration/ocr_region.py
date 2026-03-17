"""
ocr_region.py — Interactive OCR Timestamp Region Selection

Provides a GUI tool to select a rectangular region in a camera's field of view
where the timestamp is located. This region can then be used by an OCR system
to read the timestamp from the video frames.

The selected region is saved to config/cameras.json under the 'ocr_region' key
for the specified camera as **normalized [0, 1] fractions** so the pipeline
works correctly regardless of video resolution.

Backward compatibility
----------------------
Older configs that stored raw integer pixels (no ``coordinate_format`` key) are
still loaded correctly via ``OcrRegion.to_pixels()`` — no re-calibration needed.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Image file extensions treated as static frames
_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# Maximum display dimensions
_MAX_WIN_W = 1400
_MAX_WIN_H = 900


# ---------------------------------------------------------------------------
# OcrRegion — resolution-independent ROI
# ---------------------------------------------------------------------------

@dataclass
class OcrRegion:
    """
    Stores an OCR region either as raw pixels (legacy) or normalized [0, 1].

    Parameters
    ----------
    x, y, w, h : float
        Either absolute pixel values (legacy) or normalized fractions (new).
    normalized : bool
        True  → x/y/w/h are fractions in [0, 1] relative to image size.
        False → x/y/w/h are absolute pixel integers (legacy format).
    ref_w, ref_h : int | None
        Reference frame size stored alongside normalized values (informational
        only — ``to_pixels()`` uses the *actual* frame size at call-time).
    """
    x: float
    y: float
    w: float
    h: float
    normalized: bool = False
    ref_w: Optional[int] = None
    ref_h: Optional[int] = None

    def to_pixels(self, frame_w: int, frame_h: int) -> Dict[str, int]:
        """
        Return absolute pixel coords ``{x, y, w, h}`` for a frame of the given size.

        If normalized, scales fractions to the actual frame dimensions.
        If legacy (not normalized), returns the stored ints unchanged.
        Result is always clipped to frame bounds so the crop stays valid.
        """
        if self.normalized:
            x = int(round(self.x * frame_w))
            y = int(round(self.y * frame_h))
            w = max(1, int(round(self.w * frame_w)))
            h = max(1, int(round(self.h * frame_h)))
        else:
            x, y, w, h = int(self.x), int(self.y), int(self.w), max(1, int(self.h))
        # Clip to frame bounds so extraction never fails with "out of frame"
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = max(1, min(w, frame_w - x))
        h = max(1, min(h, frame_h - y))
        return {"x": x, "y": y, "w": w, "h": h}


def _is_image_path(source: str) -> bool:
    """Return True if *source* is a path to a still image file."""
    return Path(source).suffix.lower() in _IMAGE_EXTS


class OCRRegionSelector:
    """
    Interactive tool to select an OCR region from a camera's video feed.
    """
    
    def __init__(self, camera_id: str, source: str) -> None:
        self.camera_id = camera_id
        self.source = source
        self.roi: Optional[Dict[str, int]] = None
        
        self.current_frame: Optional[np.ndarray] = None
        self._disp_scale: float = 1.0
        
        # Mouse interaction state
        self._drawing = False
        self._start_pt: Optional[tuple[int, int]] = None
        self._current_pt: Optional[tuple[int, int]] = None

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        if self.source.isdigit():
            cap = cv2.VideoCapture(int(self.source))
        else:
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"  ✗  Cannot open source: {self.source}")
            return None
        return cap

    def _grab_frame(self, cap: cv2.VideoCapture) -> Optional[np.ndarray]:
        for _ in range(10):
            ret, frame = cap.read()
            if ret and frame is not None:
                return frame
        return None

    def _read_source_frame(self) -> Optional[np.ndarray]:
        if _is_image_path(self.source):
            frame = cv2.imread(self.source)
            if frame is None:
                print(f"  ✗  Cannot read image file: {self.source}")
                return None
            return frame

        cap = self._open_capture()
        if cap is None:
            return None
        frame = self._grab_frame(cap)
        cap.release()
        if frame is None:
            print(f"  ✗  Could not grab a frame from: {self.source}")
        return frame

    def _redraw(self) -> np.ndarray:
        if self.current_frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        display = self.current_frame.copy()
        
        # Draw the ROI if we have a start point
        if self._start_pt and self._current_pt:
            x1, y1 = self._start_pt
            x2, y2 = self._current_pt
            
            # Ensure valid dimensions
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            # Draw green rectangle with slight transparency
            overlay = display.copy()
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), cv2.FILLED)
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Add label
            cv2.putText(display, f"W:{x_max-x_min} H:{y_max-y_min}", 
                        (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        elif self.roi:
            # Draw existing confirmed ROI
            x, y = self.roi["x"], self.roi["y"]
            w, h = self.roi["w"], self.roi["h"]
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, f"OCR [{w}x{h}]", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Scale to fit screen
        h, w = display.shape[:2]
        if self._disp_scale < 1.0:
            dw = max(1, int(w * self._disp_scale))
            dh = max(1, int(h * self._disp_scale))
            display = cv2.resize(display, (dw, dh), interpolation=cv2.INTER_AREA)

        # Draw instructions
        dh, dw = display.shape[:2]
        instructions = "Drag to select ROI | ENTER: Confirm | R: Reset | ESC: Cancel"
        cv2.rectangle(display, (0, dh - 40), (dw, dh), (0, 0, 0), cv2.FILLED)
        cv2.putText(display, instructions, (10, dh - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return display

    def select_region(self) -> Optional[Dict[str, int]]:
        """
        Interactive GUI to select the OCR region.
        
        Returns
        -------
        dict
            {\"x\": int, \"y\": int, \"w\": int, \"h\": int} in *pixel* coords of the
            captured frame — normalization happens inside ``save_ocr_region()``.
            Returns None if cancelled.
        """
        self.current_frame = self._read_source_frame()
        if self.current_frame is None:
            return None

        h, w = self.current_frame.shape[:2]
        self._disp_scale = min(_MAX_WIN_W / max(w, 1), _MAX_WIN_H / max(h, 1), 1.0)

        win_name = f"Select OCR Timestamp Region - {self.camera_id}"
        
        def _mouse_cb(event, x, y, flags, param):
            # Convert screen coordinates back to original frame coordinates
            orig_x = int(x / self._disp_scale)
            orig_y = int(y / self._disp_scale)
            
            # Constrain to frame bounds
            orig_x = max(0, min(orig_x, w - 1))
            orig_y = max(0, min(orig_y, h - 1))

            if event == cv2.EVENT_LBUTTONDOWN:
                self._drawing = True
                self._start_pt = (orig_x, orig_y)
                self._current_pt = (orig_x, orig_y)
                self.roi = None # Clear previous selection

            elif event == cv2.EVENT_MOUSEMOVE:
                if self._drawing:
                    self._current_pt = (orig_x, orig_y)

            elif event == cv2.EVENT_LBUTTONUP:
                self._drawing = False
                self._current_pt = (orig_x, orig_y)
                
                # Finalize rect
                if self._start_pt and self._current_pt:
                    x1, y1 = self._start_pt
                    x2, y2 = self._current_pt
                    rw = abs(x2 - x1)
                    rh = abs(y2 - y1)
                    
                    if rw > 5 and rh > 5: # Minimum size threshold
                        self.roi = {
                            "x": min(x1, x2),
                            "y": min(y1, y2),
                            "w": rw,
                            "h": rh
                        }
                    else:
                        print("  ⚠  Selection too small, ignoring.")
                        self._start_pt = None
                        self._current_pt = None

        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(win_name, self._redraw())
        
        for _ in range(5):
            cv2.waitKey(20)
            
        cv2.setMouseCallback(win_name, _mouse_cb)

        print(f"\n  [{self.camera_id}] OCR Region Selection")
        print("  - Click and drag to draw a rectangle over the timestamp.")
        print("  - Press ENTER to save and exit.")
        print("  - Press R to reset the selection.")
        print("  - Press ESC to cancel without saving.\n")

        while True:
            cv2.imshow(win_name, self._redraw())
            key = cv2.waitKey(30) & 0xFF

            if key == 27: # ESC
                print("  ✗  Cancelled selection.")
                cv2.destroyWindow(win_name)
                return None
            
            elif key in (13, 10): # ENTER
                if self.roi is not None:
                    # Pass raw pixel roi — save_ocr_region normalizes it
                    frame_h, frame_w = self.current_frame.shape[:2]
                    print(f"  ✓  Region selected: {self.roi}  (frame: {frame_w}×{frame_h})")
                    cv2.destroyWindow(win_name)
                    # Store frame size for save_ocr_region
                    self.roi["_frame_w"] = frame_w
                    self.roi["_frame_h"] = frame_h
                    return self.roi
                else:
                    print("  ⚠  Please select a region first (click and drag).")
                    
            elif key in (ord('r'), ord('R')): # Reset
                print("  ↺  Selection reset.")
                self.roi = None
                self._start_pt = None
                self._current_pt = None


def save_ocr_region(
    camera_id: str,
    roi_dict: Dict[str, Union[int, float]],
    config_path: str = "config/",
) -> bool:
    """
    Save the OCR region to cameras.json as **normalized [0, 1] fractions**.

    The frame size is read from the ``_frame_w`` / ``_frame_h`` keys that
    ``OCRRegionSelector.select_region()`` embeds in *roi_dict*.  If those
    keys are absent (programmatic call), the raw pixel values are stored
    as-is for backward compatibility.

    Parameters
    ----------
    camera_id : str
    roi_dict : dict
        {\"x\": int, \"y\": int, \"w\": int, \"h\": int}
        Plus optional internal keys ``_frame_w`` / ``_frame_h`` (removed
        before writing to disk).
    config_path : str

    Returns
    -------
    bool
    """
    json_path = os.path.join(config_path, "cameras.json")
    if not os.path.exists(json_path):
        logger.error("[%s] cameras.json not found at %s", camera_id, json_path)
        return False

    try:
        with open(json_path, "r") as f:
            config = json.load(f)

        entry = next((c for c in config.get("cameras", []) if c["id"] == camera_id), None)
        if entry is None:
            logger.error("[%s] Camera not found in cameras.json", camera_id)
            return False

        # Extract the frame size embedded by OCRRegionSelector (if present)
        frame_w = roi_dict.pop("_frame_w", None)
        frame_h = roi_dict.pop("_frame_h", None)

        if frame_w and frame_h and frame_w > 0 and frame_h > 0:
            # Normalize to [0, 1]
            x, y, w, h = roi_dict["x"], roi_dict["y"], roi_dict["w"], roi_dict["h"]
            normalized_roi = {
                "x": round(x / frame_w, 6),
                "y": round(y / frame_h, 6),
                "w": round(w / frame_w, 6),
                "h": round(h / frame_h, 6),
                "coordinate_format": "normalized",
                "image_size": [frame_w, frame_h],
            }
            entry["ocr_region"] = normalized_roi
            logger.info(
                "[%s] Saved normalized OCR region: %s  (ref frame: %dx%d)",
                camera_id, normalized_roi, frame_w, frame_h,
            )
        else:
            # Fallback: save raw pixel values (legacy behaviour)
            entry["ocr_region"] = roi_dict
            logger.info("[%s] Saved OCR region (pixels): %s", camera_id, roi_dict)

        with open(json_path, "w") as f:
            json.dump(config, f, indent=2)

        return True

    except Exception as exc:
        logger.error("[%s] Failed to save OCR region: %s", camera_id, exc)
        return False


def load_ocr_region(
    camera_id: str,
    config_path: str = "config/",
) -> OcrRegion:
    """
    Load the OCR region for a specific camera from cameras.json.

    Returns an :class:`OcrRegion` that can be materialized to actual pixel
    coords for any frame resolution via ``roi.to_pixels(frame_w, frame_h)``.

    Backward compatible: old configs with raw integer pixel dicts (no
    ``coordinate_format`` key) are wrapped in a legacy ``OcrRegion``
    (``normalized=False``) and ``to_pixels()`` returns them unchanged.

    Raises
    ------
    ValueError
        If the camera or ``ocr_region`` key is not found.
    """
    json_path = os.path.join(config_path, "cameras.json")
    if not os.path.exists(json_path):
        raise ValueError(f"cameras.json not found at {json_path}")

    with open(json_path, "r") as f:
        config = json.load(f)

    entry = next((c for c in config.get("cameras", []) if c["id"] == camera_id), None)
    if entry is None:
        raise ValueError(f"Camera '{camera_id}' not found in cameras.json")

    if "ocr_region" not in entry:
        raise ValueError(f"No 'ocr_region' configured for camera '{camera_id}'")

    roi = entry["ocr_region"]

    if isinstance(roi, list):
        if len(roi) == 4:
            # Dashboard saves [x, y, w, h] as normalized 0-1; raw pixels are typically > 1
            all_unit = all(0 <= v <= 1 for v in roi)
            roi = {
                "x": roi[0],
                "y": roi[1],
                "w": roi[2],
                "h": roi[3],
                "coordinate_format": "normalized" if all_unit else None,
            }
        else:
            raise ValueError(f"Malformed list 'ocr_region' for camera '{camera_id}'")
            
    if roi.get("coordinate_format") == "normalized":
        ref = roi.get("image_size", [None, None])
        return OcrRegion(
            x=float(roi["x"]),
            y=float(roi["y"]),
            w=float(roi["w"]),
            h=float(roi["h"]),
            normalized=True,
            ref_w=ref[0],
            ref_h=ref[1],
        )

    # Legacy: raw integer pixels
    return OcrRegion(
        x=float(roi["x"]),
        y=float(roi["y"]),
        w=float(roi["w"]),
        h=float(roi["h"]),
        normalized=False,
    )
