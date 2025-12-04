"""Core rPPG analyzer for extracting heart rate from video signals.

This module implements the CHROM (Chrominance-based) method for remote
photoplethysmography signal extraction and heart rate estimation.
"""

import numpy as np
import time
from typing import Optional, Dict
from collections import deque

from .signal_processor import SignalProcessor
from ..core.face_detector import FaceDetector


class RPPGAnalyzer:
    """Analyzes video frames to extract heart rate using rPPG techniques.
    
    This class implements the CHROM method (De Haan & Jeanne, 2013) for
    extracting pulse signals from facial video and estimating heart rate.
    
    Attributes:
        fps: Frames per second of the video source.
        face_detector: FaceDetector instance for ROI extraction.
        signal_processor: SignalProcessor instance for signal analysis.
        bpm_history: Recent BPM estimates for smoothing.
        current_bpm: Most recent BPM estimate.
        last_update_time: Timestamp of last BPM update.
    """
    
    def __init__(
        self,
        fps: int = 30,
        bpm_history_size: int = 10,
        min_detection_confidence: float = 0.5
    ):
        """Initialize the rPPG analyzer.
        
        Args:
            fps: Video frame rate in frames per second.
            bpm_history_size: Number of recent BPM values to store for smoothing.
            min_detection_confidence: Face detection confidence threshold.
        """
        self.fps = fps
        
        # Initialize components
        self.face_detector = FaceDetector(
            min_detection_confidence=min_detection_confidence
        )
        self.signal_processor = SignalProcessor(
            fps=fps,
            retention_time=30  # Keep 30 seconds of data
        )
        
        # BPM tracking
        self.bpm_history = deque(maxlen=bpm_history_size)
        self.current_bpm = 0.0
        self.last_update_time = time.time()
        
        # CHROM temporal normalization buffers
        self.mean_r = 1.0
        self.mean_g = 1.0
        self.mean_b = 1.0
        self.alpha_chrom = 1.0
        
        # Temporal smoothing for RGB means (exponential moving average)
        self.ema_factor = 0.1  # Smoothing factor for running mean
        
        # Performance metrics
        self._frame_count = 0
        self._processing_times = deque(maxlen=30)
        
    def process_frame(self, frame_rgb: np.ndarray) -> Dict:
        """Process a single video frame and update rPPG signal.
        
        Args:
            frame_rgb: Input frame in RGB color space.
            
        Returns:
            Dictionary containing:
                - 'bpm': Current heart rate estimate
                - 'face_detected': Whether a face was found
                - 'bbox': Face bounding box if detected
                - 'signal_quality': Quality metric (0-1)
        """
        start_time = time.time()
        self._frame_count += 1
        
        # Get buffer info for status
        buffer_info = self.signal_processor.get_buffer_info()
        
        result = {
            'bpm': self.current_bpm,
            'face_detected': False,
            'bbox': None,
            'signal_quality': 0.0,
            'buffer_samples': buffer_info['sample_count'],
            'buffer_duration': buffer_info['duration']
        }
        
        # Detect face and extract ROI
        detection = self.face_detector.detect_face(frame_rgb)
        
        if detection is None:
            return result
        
        roi_rgb, bbox = detection
        result['face_detected'] = True
        result['bbox'] = bbox
        
        # Extract rPPG signal from ROI
        rppg_value = self._extract_chrom_signal(roi_rgb)
        
        # Validate signal value
        if not np.isfinite(rppg_value) or abs(rppg_value) > 100:
            return result
        
        # Add to signal buffer
        current_time = time.time()
        self.signal_processor.add_sample(rppg_value, current_time)
        
        # Update BPM estimate periodically (every 1 second for stability)
        if current_time - self.last_update_time >= 1.0:
            self._update_bpm_estimate()
            self.last_update_time = current_time
        
        # Get buffer info
        buffer_info = self.signal_processor.get_buffer_info()
        
        result['bpm'] = self.current_bpm
        result['signal_quality'] = self._estimate_signal_quality()
        result['buffer_samples'] = buffer_info['sample_count']
        result['buffer_duration'] = buffer_info['duration']
        
        # Track processing time
        processing_time = time.time() - start_time
        self._processing_times.append(processing_time)
        
        return result
    
    def _extract_chrom_signal(self, roi_rgb: np.ndarray) -> float:
        """Extract rPPG signal using CHROM method with temporal normalization.
        
        The CHROM method uses chrominance information to extract pulse signal
        while being robust to motion artifacts and lighting changes.
        
        Reference:
            De Haan, G., & Jeanne, V. (2013). Robust pulse rate from 
            chrominance-based rPPG. IEEE Transactions on Biomedical Engineering.
        
        Args:
            roi_rgb: Region of interest in RGB format.
            
        Returns:
            Scalar rPPG signal value for this frame.
        """
        # Compute spatial average of each color channel
        R = np.mean(roi_rgb[:, :, 0])
        G = np.mean(roi_rgb[:, :, 1])
        B = np.mean(roi_rgb[:, :, 2])
        
        # Update running mean with exponential moving average
        self.mean_r = (1 - self.ema_factor) * self.mean_r + self.ema_factor * R
        self.mean_g = (1 - self.ema_factor) * self.mean_g + self.ema_factor * G
        self.mean_b = (1 - self.ema_factor) * self.mean_b + self.ema_factor * B
        
        # Temporal normalization (crucial for CHROM)
        r_norm = R / (self.mean_r + 1e-6) - 1.0
        g_norm = G / (self.mean_g + 1e-6) - 1.0
        b_norm = B / (self.mean_b + 1e-6) - 1.0
        
        # CHROM projection
        x_chrom = 3 * r_norm - 2 * g_norm
        y_chrom = 1.5 * r_norm + g_norm - 1.5 * b_norm
        
        # Update alpha (ratio of standard deviations)
        # Use simple moving estimate
        if abs(y_chrom) > 1e-6:
            self.alpha_chrom = 0.9 * self.alpha_chrom + 0.1 * (x_chrom / y_chrom)
        
        # Final CHROM signal
        signal = x_chrom - self.alpha_chrom * y_chrom
        
        return signal
    
    def _update_bpm_estimate(self) -> None:
        """Update heart rate estimate from accumulated signal data."""
        signal, times = self.signal_processor.get_signal_array()
        
        if len(signal) < self.fps * 5:  # Need at least 5 seconds
            return
        
        # Detrend signal
        signal_detrended = self.signal_processor.detrend_signal(signal)
        
        # Check for flat signal (indicates issue)
        if np.std(signal_detrended) < 1e-10:
            print("Warning: Flat signal detected, clearing buffer")
            self.signal_processor.clear_buffers()
            return
        
        # Apply bandpass filter
        signal_filtered = self.signal_processor.apply_bandpass_filter(signal_detrended)
        
        if signal_filtered is None:
            return
        
        # Check for NaN or Inf in filtered signal
        if not np.all(np.isfinite(signal_filtered)):
            print("Warning: Invalid filtered signal, clearing buffer")
            self.signal_processor.clear_buffers()
            return
        
        # Estimate BPM
        bpm = self.signal_processor.estimate_bpm(signal_filtered)
        
        if bpm is not None and 40 <= bpm <= 180:  # Valid BPM range
            self.bpm_history.append(bpm)
            
            # Enhanced smoothing: Use weighted average of recent BPMs
            if len(self.bpm_history) >= 5:
                # Use median of last 5 for robustness against outliers
                recent_bpms = list(self.bpm_history)[-5:]
                median_bpm = np.median(recent_bpms)
                
                # Smooth transition with exponential moving average
                if self.current_bpm > 0:
                    # 70% old value, 30% new median (slower adaptation)
                    self.current_bpm = 0.7 * self.current_bpm + 0.3 * median_bpm
                else:
                    self.current_bpm = median_bpm
            elif len(self.bpm_history) >= 3:
                # Use mean for first few samples
                self.current_bpm = np.mean(list(self.bpm_history))
            else:
                self.current_bpm = bpm
    
    def _estimate_signal_quality(self) -> float:
        """Estimate quality of current signal.
        
        Returns:
            Quality score between 0 (poor) and 1 (excellent).
        """
        buffer_info = self.signal_processor.get_buffer_info()
        sample_count = buffer_info['sample_count']
        
        # Quality based on buffer fill
        min_samples = self.fps * 5  # 5 seconds minimum
        if sample_count < min_samples:
            return sample_count / min_samples
        
        # Check signal variance
        signal, _ = self.signal_processor.get_signal_array()
        signal_std = np.std(signal)
        
        # Good signal should have reasonable variance
        if signal_std < 0.01:
            return 0.3
        elif signal_std > 10.0:
            return 0.5
        else:
            return min(1.0, 0.7 + (0.3 * min(signal_std, 1.0)))
    
    def get_signal_data(self) -> Dict:
        """Get current signal data for visualization.
        
        Returns:
            Dictionary containing:
                - 'signal': Raw signal array
                - 'time': Time array
                - 'filtered_signal': Filtered signal (if available)
                - 'frequencies': Frequency array for spectrum
                - 'spectrum': Magnitude spectrum
        """
        signal, times = self.signal_processor.get_signal_array()
        
        result = {
            'signal': signal,
            'time': times,
            'filtered_signal': None,
            'frequencies': np.array([]),
            'spectrum': np.array([])
        }
        
        if len(signal) < self.fps * 2:
            return result
        
        # Process signal
        signal_detrended = self.signal_processor.detrend_signal(signal)
        signal_filtered = self.signal_processor.apply_bandpass_filter(signal_detrended)
        
        if signal_filtered is not None:
            result['filtered_signal'] = signal_filtered
            
            # Compute spectrum
            freqs, spectrum = self.signal_processor.compute_frequency_spectrum(signal_filtered)
            result['frequencies'] = freqs
            result['spectrum'] = spectrum
        
        return result
    
    def reset(self) -> None:
        """Reset analyzer state and clear all buffers."""
        self.signal_processor.clear_buffers()
        self.bpm_history.clear()
        self.current_bpm = 0.0
        self.face_detector.reset()
        self._frame_count = 0
        self._processing_times.clear()
    
    def get_performance_metrics(self) -> Dict:
        """Get performance statistics.
        
        Returns:
            Dictionary with metrics like processing time and frame count.
        """
        avg_processing_time = 0.0
        if self._processing_times:
            avg_processing_time = np.mean(list(self._processing_times))
        
        buffer_info = self.signal_processor.get_buffer_info()
        
        return {
            'frame_count': self._frame_count,
            'avg_processing_time': avg_processing_time,
            'buffer_samples': buffer_info['sample_count'],
            'buffer_duration': buffer_info['duration']
        }
