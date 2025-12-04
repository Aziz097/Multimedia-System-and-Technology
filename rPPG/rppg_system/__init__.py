"""rPPG System - Remote Photoplethysmography Heart Rate Monitor.

This package implements CHROM-based rPPG algorithm for non-contact
heart rate monitoring using webcam video.
"""

__version__ = "1.0.0"
__author__ = "122140097"

from .processing.rppg_analyzer import RPPGAnalyzer
from .core.face_detector import FaceDetector
from .processing.signal_processor import SignalProcessor

__all__ = ['RPPGAnalyzer', 'FaceDetector', 'SignalProcessor']
