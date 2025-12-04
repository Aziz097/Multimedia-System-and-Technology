"""PyQt6 GUI application for real-time rPPG heart rate monitoring.

This module provides a modern, minimalist graphical interface for monitoring
heart rate using remote photoplethysmography (rPPG) from webcam video.
"""

import sys
import cv2
import numpy as np
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPen

from ..processing.rppg_analyzer import RPPGAnalyzer


# Color Scheme
class AppColors:
    BG_DARK = "#121212"
    PANEL_BG = "#1E1E1E"
    BORDER = "#333333"
    
    ACCENT = "#00BCD4"
    ACCENT_HOVER = "#26C6DA"
    
    TEXT_MAIN = "#FFFFFF"
    TEXT_DIM = "#AAAAAA"
    
    STATUS_OK = "#4CAF50"
    STATUS_WARN = "#FFC107"
    STATUS_BAD = "#F44336"


def get_status_color(bpm):
    """Get color based on BPM value."""
    if bpm < 50 or bpm > 120:
        return AppColors.STATUS_BAD
    elif bpm < 60 or bpm > 100:
        return AppColors.STATUS_WARN
    else:
        return AppColors.STATUS_OK


class VideoThread(QThread):
    """Background thread for video capture and processing.
    
    Signals:
        frame_ready: Emitted when new frame is processed with (frame_bgr, result_dict).
    """
    
    frame_ready = pyqtSignal(np.ndarray, dict)
    
    def __init__(self, analyzer: 'RPPGAnalyzer', camera_id: int = 0):
        """Initialize video capture thread.
        
        Args:
            analyzer: RPPGAnalyzer instance.
            camera_id: Camera device index.
        """
        super().__init__()
        self.analyzer = analyzer
        self.camera_id = camera_id
        self.running = False
        self.cap = None
        
    def run(self):
        """Main thread loop for video processing."""
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print("Error: Cannot open camera")
                return
            
            print("Camera opened successfully")
        except Exception as e:
            print(f"Error during initialization: {e}")
            import traceback
            traceback.print_exc()
            return
        
        self.running = True
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Warning: Cannot read frame")
                    continue
                
                # Mirror frame
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB for processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                result = self.analyzer.process_frame(frame_rgb)
                
                # Draw face detection on frame
                if result['face_detected'] and result['bbox']:
                    x, y, w, h = result['bbox']
                    # Draw face rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Draw ROI (forehead)
                    forehead_h = int(h * 0.4)
                    cv2.rectangle(frame, (x, y), (x + w, y + forehead_h), (255, 0, 0), 2)
                
                # Emit frame and results
                self.frame_ready.emit(frame, result)
                
                # Control frame rate
                self.msleep(33)  # ~30 FPS
            except Exception as e:
                print(f"Error processing frame: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Cleanup
        if self.cap:
            self.cap.release()
            print("Camera released")
    
    def stop(self):
        """Stop the video thread."""
        self.running = False
        self.wait()


class BPMWidget(QWidget):
    """Simple BPM display widget."""
    
    def __init__(self):
        super().__init__()
        self.bpm = 0.0
        self.quality = 0.0
        self.setMinimumHeight(150)
        
    def set_bpm(self, bpm: float, quality: float = 0.0):
        self.bpm = bpm
        self.quality = quality
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        painter.fillRect(self.rect(), QColor(AppColors.PANEL_BG))
        
        if self.bpm > 0:
            color = get_status_color(self.bpm)
            painter.setPen(QColor(color))
            font = QFont("Segoe UI", 72, QFont.Weight.Bold)
            painter.setFont(font)
            bpm_text = f"{int(self.bpm)}"
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, bpm_text)
        else:
            painter.setPen(QColor(AppColors.TEXT_DIM))
            font = QFont("Segoe UI", 36)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "--")


class SignalPlotWidget(QWidget):
    """Minimalist signal plot widget."""
    
    def __init__(self):
        super().__init__()
        self.signal_data = None
        self.time_data = None
        self.setMinimumHeight(150)
        
    def set_data(self, signal: Optional[np.ndarray], time: Optional[np.ndarray]):
        self.signal_data = signal
        self.time_data = time
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(AppColors.PANEL_BG))
        
        # Border
        pen = QPen(QColor(AppColors.BORDER), 1)
        painter.setPen(pen)
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)
        
        # Plot signal
        if self.signal_data is not None and self.time_data is not None and len(self.signal_data) > 1:
            time_window = 10.0
            if len(self.time_data) > 0:
                current_time = self.time_data[-1]
                mask = self.time_data >= (current_time - time_window)
                signal = self.signal_data[mask]
                time = self.time_data[mask]
                
                if len(signal) > 1:
                    margin = 10
                    plot_width = self.width() - 2 * margin
                    plot_height = self.height() - 2 * margin
                    
                    sig_min, sig_max = np.min(signal), np.max(signal)
                    if sig_max - sig_min > 1e-6:
                        signal_norm = (signal - sig_min) / (sig_max - sig_min)
                    else:
                        signal_norm = np.ones_like(signal) * 0.5
                    
                    time_norm = (time - time[0]) / time_window if time_window > 0 else time * 0
                    
                    painter.setPen(QPen(QColor(AppColors.ACCENT), 2))
                    
                    for i in range(len(signal_norm) - 1):
                        x1 = margin + int(time_norm[i] * plot_width)
                        y1 = margin + int((1 - signal_norm[i]) * plot_height)
                        x2 = margin + int(time_norm[i + 1] * plot_width)
                        y2 = margin + int((1 - signal_norm[i + 1]) * plot_height)
                        painter.drawLine(x1, y1, x2, y2)


class SpectrumWidget(QWidget):
    """Minimalist spectrum widget."""
    
    def __init__(self):
        super().__init__()
        self.frequencies = None
        self.spectrum = None
        self.current_bpm = 0.0
        self.setMinimumHeight(120)
        
    def set_data(self, frequencies: Optional[np.ndarray], spectrum: Optional[np.ndarray], bpm: float = 0.0):
        self.frequencies = frequencies
        self.spectrum = spectrum
        self.current_bpm = bpm
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        painter.fillRect(self.rect(), QColor(AppColors.PANEL_BG))
        pen = QPen(QColor(AppColors.BORDER), 1)
        painter.setPen(pen)
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)
        
        if self.frequencies is not None and self.spectrum is not None and len(self.frequencies) > 0:
            margin = 10
            plot_width = self.width() - 2 * margin
            plot_height = self.height() - 2 * margin
            
            max_mag = np.max(self.spectrum)
            if max_mag > 1e-10:
                spectrum_norm = self.spectrum / max_mag
            else:
                spectrum_norm = self.spectrum
            
            bar_width = max(2, plot_width // len(self.frequencies))
            
            for i, (freq, mag) in enumerate(zip(self.frequencies, spectrum_norm)):
                x = margin + i * bar_width
                bar_height = int(mag * plot_height)
                y = margin + plot_height - bar_height
                
                bpm_freq = self.current_bpm / 60.0
                if abs(freq - bpm_freq) < 0.1:
                    color = QColor(AppColors.STATUS_BAD)
                else:
                    color = QColor(AppColors.ACCENT)
                
                painter.fillRect(x, y, bar_width - 1, bar_height, color)


class MainWindow(QMainWindow):
    """Main application window for rPPG heart rate monitoring."""
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("rPPG Heart Rate Monitor")
        self.setGeometry(100, 100, 1400, 800)
        
        # Apply dark theme
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {AppColors.BG_DARK};
            }}
            QLabel {{
                color: {AppColors.TEXT_MAIN};
                font-family: 'Segoe UI';
            }}
        """)
        
        # Initialize analyzer
        self.analyzer = RPPGAnalyzer(fps=30)
        
        # Central widget with horizontal layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # LEFT COLUMN: Camera feed
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(f"background-color: {AppColors.PANEL_BG}; border-radius: 8px;")
        self.video_label.setMinimumSize(640, 480)
        left_layout.addWidget(self.video_label)
        
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet(f"color: {AppColors.TEXT_DIM}; font-size: 11px;")
        left_layout.addWidget(self.fps_label)
        
        # RIGHT COLUMN: Info panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(20)
        right_panel.setMaximumWidth(500)
        
        # Title
        title_label = QLabel("Heart Rate Monitor")
        title_label.setStyleSheet(f"color: {AppColors.TEXT_MAIN}; font-size: 24px; font-weight: bold;")
        right_layout.addWidget(title_label)
        
        # BPM Display
        self.bpm_widget = BPMWidget()
        right_layout.addWidget(self.bpm_widget)
        
        # Status label
        self.status_label = QLabel("Status: Starting...")
        self.status_label.setStyleSheet(f"color: {AppColors.TEXT_DIM}; font-size: 13px;")
        right_layout.addWidget(self.status_label)
        
        # Signal plot
        signal_label = QLabel("PPG Signal")
        signal_label.setStyleSheet(f"color: {AppColors.TEXT_DIM}; font-size: 12px; margin-top: 10px;")
        right_layout.addWidget(signal_label)
        
        self.signal_plot = SignalPlotWidget()
        self.signal_plot.setMinimumHeight(150)
        right_layout.addWidget(self.signal_plot)
        
        # Spectrum plot
        spectrum_label = QLabel("Frequency Spectrum")
        spectrum_label.setStyleSheet(f"color: {AppColors.TEXT_DIM}; font-size: 12px; margin-top: 10px;")
        right_layout.addWidget(spectrum_label)
        
        self.spectrum_widget = SpectrumWidget()
        self.spectrum_widget.setMinimumHeight(150)
        right_layout.addWidget(self.spectrum_widget)
        
        right_layout.addStretch()
        
        # Add columns to main layout
        main_layout.addWidget(left_panel, stretch=2)
        main_layout.addWidget(right_panel, stretch=1)
        
        # Video thread
        self.video_thread = VideoThread(self.analyzer, camera_id=0)
        self.video_thread.frame_ready.connect(self.update_display)
        self.video_thread.start()
        
        # Status bar
        self.statusBar().showMessage("Application started")
        self.statusBar().setStyleSheet(f"color: {AppColors.TEXT_DIM}; background-color: {AppColors.BG_DARK};")
        
        self.frame_count = 0
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(1000)
    
    def update_fps(self):
        """Update FPS display."""
        self.fps_label.setText(f"FPS: {self.frame_count}")
        self.frame_count = 0
    
    def update_display(self, frame_bgr: np.ndarray, result: dict):
        """Update GUI with new frame and analysis results."""
        # Update video display
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
        
        # Update BPM
        if 'bpm' in result and result['bpm'] > 0:
            self.bpm_widget.set_bpm(result['bpm'], result.get('signal_quality', 0.0))
            status_color = get_status_color(result['bpm'])
            buffer_samples = result.get('buffer_samples', 0)
            self.status_label.setText(f"Status: Active • {buffer_samples} samples")
            self.status_label.setStyleSheet(f"color: {status_color}; font-size: 13px;")
        else:
            buffer_samples = result.get('buffer_samples', 0)
            self.status_label.setText(f"Status: Detecting face... • {buffer_samples} samples")
            self.status_label.setStyleSheet(f"color: {AppColors.TEXT_DIM}; font-size: 13px;")
        
        # Get signal data from analyzer
        signal_data = self.analyzer.get_signal_data()
        
        # Update signal plot
        if signal_data['filtered_signal'] is not None and len(signal_data['filtered_signal']) > 0:
            self.signal_plot.set_data(signal_data['filtered_signal'], signal_data['time'])
        elif signal_data['signal'] is not None and len(signal_data['signal']) > 0:
            self.signal_plot.set_data(signal_data['signal'], signal_data['time'])
        
        # Update spectrum
        if signal_data['frequencies'] is not None and signal_data['spectrum'] is not None:
            self.spectrum_widget.set_data(signal_data['frequencies'], signal_data['spectrum'])
        
        self.frame_count += 1
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        event.accept()

