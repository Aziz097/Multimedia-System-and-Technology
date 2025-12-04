"""Signal processing module for rPPG heart rate monitoring.

This module provides core signal processing functionality including filtering,
detrending, and frequency analysis for photoplethysmography (PPG) signals.
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.fft import fft, fftfreq
from typing import Tuple, Optional
from collections import deque
import time


class SignalProcessor:
    """Handles signal processing operations for rPPG analysis.
    
    This class implements various signal processing techniques including
    bandpass filtering, detrending, and BPM estimation using FFT analysis.
    
    Attributes:
        fps: Sampling frequency (frames per second).
        lowcut: Lower cutoff frequency for bandpass filter (Hz).
        highcut: Higher cutoff frequency for bandpass filter (Hz).
        filter_order: Order of the Butterworth filter.
        signal_buffer: Circular buffer storing raw signal values.
        time_buffer: Circular buffer storing corresponding timestamps.
        retention_time: Duration to retain historical data (seconds).
    """
    
    def __init__(
        self, 
        fps: int = 30,
        lowcut: float = 0.67,
        highcut: float = 4.0,
        filter_order: int = 3,
        retention_time: int = 30
    ):
        """Initialize the signal processor.
        
        Args:
            fps: Sampling frequency in frames per second.
            lowcut: Lower cutoff frequency (Hz) corresponding to 40 BPM.
            highcut: Higher cutoff frequency (Hz) corresponding to 240 BPM.
            filter_order: Order of the Butterworth bandpass filter.
            retention_time: Time window for retaining signal data in seconds.
        """
        self.fps = fps
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.retention_time = retention_time
        
        # Unlimited buffers with manual cleanup
        self.signal_buffer = deque()
        self.time_buffer = deque()
        
        # Filter coefficients
        self._filter_coeffs = self._design_bandpass_filter()
        
    def _design_bandpass_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """Design Butterworth bandpass filter coefficients.
        
        Returns:
            Tuple containing numerator (b) and denominator (a) filter coefficients.
        """
        nyquist_freq = 0.5 * self.fps
        low_norm = self.lowcut / nyquist_freq
        high_norm = self.highcut / nyquist_freq
        b, a = butter(self.filter_order, [low_norm, high_norm], btype='band')
        return b, a
    
    def add_sample(self, value: float, timestamp: float) -> None:
        """Add a new sample to the signal buffer.
        
        Args:
            value: Signal amplitude value.
            timestamp: Time in seconds when sample was captured.
        """
        self.signal_buffer.append(value)
        self.time_buffer.append(timestamp)
        self._cleanup_old_data(timestamp)
    
    def _cleanup_old_data(self, current_time: float) -> None:
        """Remove data older than retention window.
        
        Args:
            current_time: Current timestamp in seconds.
        """
        cutoff_time = current_time - self.retention_time
        
        while self.time_buffer and self.time_buffer[0] < cutoff_time:
            self.time_buffer.popleft()
            self.signal_buffer.popleft()
    
    def get_signal_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get signal and time arrays from buffers.
        
        Returns:
            Tuple of (signal_array, time_array) as numpy arrays.
        """
        signal = np.array(list(self.signal_buffer), dtype=np.float64)
        times = np.array(list(self.time_buffer), dtype=np.float64)
        return signal, times
    
    def apply_bandpass_filter(self, signal: np.ndarray) -> Optional[np.ndarray]:
        """Apply bandpass filter to remove noise and trend.
        
        Args:
            signal: Input signal array.
            
        Returns:
            Filtered signal array, or None if signal is too short.
        """
        if len(signal) < self.fps:
            return None
        
        b, a = self._filter_coeffs
        
        try:
            filtered = filtfilt(b, a, signal)
            return filtered
        except Exception:
            return None
    
    def detrend_signal(self, signal: np.ndarray, window_size: int = 30) -> np.ndarray:
        """Remove linear trend using moving average subtraction.
        
        Args:
            signal: Input signal array.
            window_size: Window size for moving average computation.
            
        Returns:
            Detrended signal array normalized to zero mean and unit variance.
        """
        if len(signal) < window_size:
            mean_val = np.mean(signal)
            std_val = np.std(signal)
            if std_val > 1e-6:
                return (signal - mean_val) / std_val
            return signal - mean_val
        
        # Moving average detrending
        moving_avg = np.convolve(
            signal, 
            np.ones(window_size) / window_size, 
            mode='same'
        )
        detrended = signal - moving_avg
        
        # Normalize to prevent drift
        std_val = np.std(detrended)
        if std_val > 1e-6:
            detrended = detrended / std_val
        
        return detrended
    
    def estimate_bpm(self, signal: np.ndarray) -> Optional[float]:
        """Estimate heart rate in BPM using FFT with Welch's method.
        
        Args:
            signal: Preprocessed rPPG signal array.
            
        Returns:
            Estimated BPM value, or None if estimation fails.
        """
        if len(signal) < self.fps * 3:  # Need at least 3 seconds
            return None
        
        # Check signal variance
        if np.std(signal) < 1e-6:
            return None
        
        # Use Welch's method for better spectral estimation
        try:
            nperseg = min(len(signal), self.fps * 8)  # 8 second window
            freqs, power = welch(
                signal,
                fs=self.fps,
                nperseg=nperseg,
                noverlap=nperseg // 2,
                window='hamming'
            )
        except:
            # Fallback to standard FFT
            n = len(signal)
            freqs = fftfreq(n, 1.0 / self.fps)
            fft_vals = fft(signal)
            power = np.abs(fft_vals) ** 2
        
        # Focus on physiological frequency range (0.67-4.0 Hz = 40-240 BPM)
        valid_idx = np.where((freqs >= self.lowcut) & (freqs <= self.highcut))[0]
        
        if len(valid_idx) == 0:
            return None
        
        valid_freqs = freqs[valid_idx]
        valid_power = power[valid_idx]
        
        # Find dominant frequency (peak in power spectrum)
        max_idx = np.argmax(valid_power)
        dominant_freq = valid_freqs[max_idx]
        
        # Convert to BPM
        bpm = dominant_freq * 60.0
        
        # Additional validation: Check if peak is prominent
        peak_power = valid_power[max_idx]
        median_power = np.median(valid_power)
        
        # SNR check: Peak should be significantly higher than noise floor
        if peak_power < 2.0 * median_power:
            return None
        
        # Validate BPM range
        if bpm < 40 or bpm > 180:
            return None
        
        return bpm
    
    def compute_frequency_spectrum(
        self, 
        signal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute frequency spectrum for visualization.
        
        Args:
            signal: Input signal array.
            
        Returns:
            Tuple of (frequencies, magnitudes) in the physiological range.
        """
        n = len(signal)
        frequencies = fftfreq(n, 1.0 / self.fps)
        fft_magnitude = np.abs(fft(signal))
        
        # Filter to physiological range
        valid_idx = np.where((frequencies > 0.5) & (frequencies < 4.0))[0]
        
        if len(valid_idx) == 0:
            return np.array([]), np.array([])
        
        return frequencies[valid_idx], fft_magnitude[valid_idx]
    
    def clear_buffers(self) -> None:
        """Clear all buffered signal data."""
        self.signal_buffer.clear()
        self.time_buffer.clear()
    
    def get_buffer_info(self) -> dict:
        """Get information about current buffer state.
        
        Returns:
            Dictionary containing buffer statistics.
        """
        return {
            'sample_count': len(self.signal_buffer),
            'duration': self.time_buffer[-1] - self.time_buffer[0] if len(self.time_buffer) > 1 else 0,
            'retention_time': self.retention_time
        }
