"""
ocr_timestamp.py — Video Timestamp OCR Extractor

Extracts timestamps from camera frames using an ROI defined in cameras.json.
Uses `pytesseract` to read the digital text overlay. Contains robust preprocessing
(grayscale, adaptive thresholding, scaling) to improve OCR accuracy on small or
blended text, as well as heuristic replacement of common misread characters.

Provides a fallback computation mechanism when OCR fails (e.g., during minute transitions).
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Optional

import cv2
import numpy as np

try:
    import easyocr
except ImportError:
    easyocr = None
    logging.getLogger(__name__).warning("easyocr is not installed. OCR extraction will fail.")

from calibration.ocr_region import load_ocr_region, OcrRegion

logger = logging.getLogger(__name__)


class TimestampExtractor:
    """
    Extracts and parses timestamps from a camera's video feed.
    """

    def __init__(self, camera_id: str, config_path: str = "config/") -> None:
        """
        Initializes the extractor for a specific camera.

        Parameters
        ----------
        camera_id : str
            The camera ID configured in cameras.json.
        config_path : str
            Path to the directory containing cameras.json.
        """
        self.camera_id = camera_id
        
        try:
            self.roi: Optional[OcrRegion] = load_ocr_region(camera_id, config_path)
            fmt = "normalized" if self.roi.normalized else "pixels (legacy)"
            logger.info("[%s] OCR Region loaded (%s): %s", self.camera_id, fmt, self.roi)
        except ValueError as e:
            logger.error("[%s] Failed to load OCR region: %s", self.camera_id, e)
            self.roi = None

        if easyocr is None:
            logger.error("[%s] easyocr module not found. Extraction disabled.", self.camera_id)
            self.reader = None
        else:
            # Auto-detect GPU — EasyOCR is 5-10x faster on GPU
            try:
                import torch
                _use_gpu = torch.cuda.is_available()
            except Exception:
                _use_gpu = False
            logger.info("[%s] EasyOCR initializing (gpu=%s)...", self.camera_id, _use_gpu)
            self.reader = easyocr.Reader(['en'], gpu=_use_gpu)

        # State tracking for fallback and drift detection
        self._last_successful_time: Optional[datetime] = None
        self._base_timestamp: Optional[datetime] = None
        
        # Aggressively map all alphabetical characters to their nearest visual numeric equivalent.
        # Since the OCR region is strictly a YYYY-MM-DD HH:MM:SS clock, any letter is definitively a misread.
        self._replacements = {
            'A': '4', 'a': '4',
            'B': '8', 'b': '6',
            'C': '0', 'c': '0',
            'D': '0', 'd': '0',
            'E': '3', 'e': '3',
            'F': '5', 'f': '5',
            'G': '6', 'g': '9',
            'H': '4', 'h': '4',
            'I': '1', 'i': '1',
            'J': '1', 'j': '1',
            'K': '4', 'k': '4',
            'L': '1', 'l': '1',
            'M': '1', 'm': '1', # or '11' but keeping it 1 len
            'N': '1', 'n': '1',
            'O': '0', 'o': '0',
            'P': '9', 'p': '9',
            'Q': '0', 'q': '0',
            'R': '8', 'r': '1',
            'S': '5', 's': '5',
            'T': '7', 't': '7',
            'U': '0', 'u': '0',
            'V': '0', 'v': '0',
            'W': '3', 'w': '3',
            'X': '8', 'x': '8',
            'Y': '4', 'y': '4',
            'Z': '2', 'z': '2',
            '|': '1', '/': '1', '\\': '1', '_': '-', '=': '-'
        }

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocesses the cropped ROI for optimal OCR accuracy.
        
        1. Resizes 2x to enlarge small text
        2. Converts to grayscale
        3. Applies adaptive Gaussian thresholding to handle varied lighting
        """
        # 1. Resize 2x (Cubic interpolation works best for enlarging text)
        h, w = crop.shape[:2]
        resized = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        
        # 2. Convert to grayscale. 
        # EasyOCR uses deep learning models that often perform better on simple grayscale
        # compared to the harsh binary thresholding Tesseract required.
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        return gray

    def _clean_text(self, text: str) -> str:
        """
        Cleans the OCR output by stripping whitespace and replacing common misreads.
        """
        text = text.strip()
        # Convert commonly confused letters to numbers
        new_text = ""
        for char in text:
            # We only replace if it's a letter but shouldn't be (in a typical timestamp)
            new_text += self._replacements.get(char, char)
        
        # Keep only numbers, spaces, dashes, dots, and colons.
        cleaned = re.sub(r'[^0-9\-\s\:\.]', '', new_text)
        
        # EasyOCR often misreads the colon in 17:44:13 as a dot 17:44.13. 
        # We know colons only appear in the time section. If we see a dot after a space, it's likely a colon.
        # Simplest fix: replace all dots with colons.
        cleaned = cleaned.replace('.', ':')
        
        return cleaned

    def extract(self, frame: np.ndarray) -> Optional[datetime]:
        """
        Extracts a datetime object from the given video frame.

        Parameters
        ----------
        frame : np.ndarray
            The BGR video frame.

        Returns
        -------
        datetime or None
            The parsed datetime, or None if OCR failed / parsed time was invalid.
        """
        if self.roi is None or self.reader is None or frame is None:
            return None

        # Resolve to actual pixel coords for this frame's resolution
        fh, fw = frame.shape[:2]
        px = self.roi.to_pixels(fw, fh)
        x, y, w, h = px["x"], px["y"], px["w"], px["h"]

        # Ensure crop is within bounds
        if x < 0 or y < 0 or x + w > fw or y + h > fh:
            logger.warning(
                "[%s] ROI %s → pixel(%d,%d,%d,%d) out of frame bounds (%dx%d)",
                self.camera_id, self.roi, x, y, w, h, fw, fh,
            )
            return None

        crop = frame[y:y+h, x:x+w]
        
        # Preprocess
        processed = self._preprocess_crop(crop)
        
        # Run OCR via easyocr
        try:
            # Read text and extract just the strings
            results = self.reader.readtext(processed, detail=0)
            raw_text = " ".join(results)
        except Exception as e:
            logger.debug("[%s] easyocr error: %s", self.camera_id, e)
            return None
            
        clean_text = self._clean_text(raw_text)

        if not clean_text:
            logger.debug("[%s] OCR returned empty string after cleaning.", self.camera_id)
            return None

        # Parse datetime "YYYY-MM-DD HH:MM:SS"
        try:
            # Allow minor variations in whitespace
            parts = clean_text.split()
            if len(parts) >= 2:
                # E.g. "2025-03-14 10:32:15"
                # Sometimes EasyOCR outputs a third part for seconds, or squashes them
                date_part = parts[0]
                time_part = "".join(parts[1:]) # Rejoin 17 : 44 : 13 if split
                dt_str = f"{date_part} {time_part}"
                parsed_time = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            else:
                parsed_time = datetime.strptime(clean_text, "%Y-%m-%d %H:%M:%S")
                
            # Validation: monotonically increasing check
            if self._last_successful_time is not None:
                time_diff = (parsed_time - self._last_successful_time).total_seconds()
                
                # Reject if it jumps backward by more than 2 seconds
                if time_diff < -2.0:
                    logger.debug("[%s] Rejecting backward time jump: %s -> %s", 
                                 self.camera_id, self._last_successful_time, parsed_time)
                    return None
                    
            # Update state
            self._last_successful_time = parsed_time
            if self._base_timestamp is None:
                self._base_timestamp = parsed_time
                
            return parsed_time
            
        except ValueError as e:
            logger.debug("[%s] Failed to parse OCR text '%s': %s", self.camera_id, clean_text, e)
            return None

    def get_fps_adjusted_timestamp(
        self, 
        frame_number: int, 
        fps: float, 
        base_timestamp: Optional[datetime] = None
    ) -> Optional[datetime]:
        """
        Fallback mechanism to compute time based on frame index and FPS.
        
        Parameters
        ----------
        frame_number : int
            The current zero-indexed frame number.
        fps : float
            The video frames per second.
        base_timestamp : datetime, optional
            The reference timestamp at frame 0. If None, uses the first
            successfully OCR'd timestamp stored in self._base_timestamp.
            
        Returns
        -------
        datetime or None
            The computed timestamp, or None if no base timestamp is available.
        """
        ref_time = base_timestamp or self._base_timestamp
        
        if ref_time is None:
            return None
            
        if fps <= 0:
            logger.warning("[%s] Invalid FPS value: %s", self.camera_id, fps)
            return ref_time
            
        seconds_offset = frame_number / fps
        return ref_time + timedelta(seconds=seconds_offset)
