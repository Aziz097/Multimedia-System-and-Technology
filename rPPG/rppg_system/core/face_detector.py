"""Face detection and ROI extraction module for rPPG monitoring.

This module handles face detection using MediaPipe and extracts regions of
interest (ROI) suitable for photoplethysmography signal extraction.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple


class FaceDetector:
    """Detects faces and extracts regions of interest for rPPG analysis.
    
    This class uses MediaPipe Face Detection to locate faces in video frames
    and extract forehead ROI for optimal PPG signal quality.
    
    Attributes:
        min_detection_confidence: Minimum confidence threshold for face detection.
        model_selection: MediaPipe model (0 for short-range, 1 for full-range).
        face_detection: MediaPipe FaceDetection instance.
    """
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        model_selection: int = 0
    ):
        """Initialize the face detector.
        
        Args:
            min_detection_confidence: Minimum confidence score (0.0 to 1.0).
            model_selection: Model type - 0 for cameras within 2m, 1 for >2m.
        """
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        
        # Initialize MediaPipe Face Detection
        mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence
        )
        
        # Cache for last known face position
        self._last_bbox = None
        
    def detect_face(
        self, 
        frame_rgb: np.ndarray
    ) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Detect face and extract ROI from frame.
        
        Args:
            frame_rgb: Input frame in RGB format.
            
        Returns:
            Tuple of (roi_rgb, bbox) where:
                - roi_rgb: Extracted region of interest in RGB.
                - bbox: Bounding box as (x, y, width, height).
            Returns None if no face detected.
        """
        try:
            # Process frame with MediaPipe
            results = self.face_detection.process(frame_rgb)
            
            if not results.detections:
                return None
            
            # Get first detected face
            detection = results.detections[0]
            
            # Extract bounding box
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame_rgb.shape
            
            # Convert to pixel coordinates
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Ensure coordinates are within frame bounds
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)
            
            # Extract forehead region (upper 40% of face)
            forehead_height = int(height * 0.4)
            roi_y = max(0, y)
            roi_y_end = min(h, y + forehead_height)
            roi_x = max(0, x)
            roi_x_end = min(w, x + width)
            
            # Extract ROI
            roi = frame_rgb[roi_y:roi_y_end, roi_x:roi_x_end]
            
            # Validate ROI size
            if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
                return None
            
            # Cache bbox for visualization
            self._last_bbox = (x, y, width, height)
            
            return roi, (x, y, width, height)
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return None
    
    def draw_face_landmarks(
        self, 
        frame: np.ndarray, 
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """Draw face bounding box on frame.
        
        Args:
            frame: Input frame in BGR format.
            bbox: Bounding box as (x, y, width, height). Uses cached if None.
            
        Returns:
            Frame with drawn landmarks.
        """
        if bbox is None:
            bbox = self._last_bbox
        
        if bbox is None:
            return frame
        
        x, y, width, height = bbox
        
        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )
        
        # Draw ROI rectangle (forehead region)
        forehead_height = int(height * 0.4)
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + forehead_height),
            (255, 0, 0),
            2
        )
        
        return frame
    
    def get_last_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Get the last detected face bounding box.
        
        Returns:
            Last bbox as (x, y, width, height), or None if no face detected yet.
        """
        return self._last_bbox
    
    def reset(self) -> None:
        """Reset cached face position."""
        self._last_bbox = None
    
    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
