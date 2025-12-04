import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import threading
import time
from collections import deque
import queue


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

class RealTimeRPPG:
    def __init__(self, window_size=30, fps=30):
        """
        Inisialisasi sistem rPPG real-time.
        
        Args:
            window_size (int): Ukuran window dalam detik untuk analisis sinyal
            fps (int): Frame rate kamera
        """
        self.window_size = window_size  
        self.fps = fps
        self.frame_count = 0
        self.start_time = time.time()
        self.signal_buffer = deque(maxlen=window_size * fps)
        self.time_buffer = deque(maxlen=window_size * fps)
        self.bpm_history = deque(maxlen=10)  
        self.current_bpm = 0
        self.roi_frames = deque(maxlen=10)  
        
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Tidak dapat membuka webcam")
        
        
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            self.cap.release()
            raise Exception("Webcam terbuka tapi tidak dapat membaca frame")
        print(f"Webcam berhasil diinisialisasi: {test_frame.shape}")
            
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)
            
        
        self.canvas_width = 1280
        self.canvas_height = 720
        
        
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        
        
        self.stop_event = threading.Event()
        self.frame_available = threading.Event()
        self.processing_available = threading.Event()
        
        
        self.running = True
        self.face_bbox = None
        
    def create_integrated_canvas(self, frame, frame_rgb):
        """
        Buat canvas terintegrasi dengan semua visualisasi dalam satu window.
        """
        
        canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255
        
        
        video_width = 840
        video_height = 480
        frame_resized = cv2.resize(frame, (video_width, video_height))
        
        
        canvas[10:10+video_height, 10:10+video_width] = frame_resized
        
        
        info_x = video_width + 20
        info_y = 10
        info_width = 410
        info_height = 230
        self.draw_info_panel(canvas, info_x, info_y, info_width, info_height)
        
        
        signal_x = info_x
        signal_y = info_y + info_height + 10
        signal_width = info_width
        signal_height = 240
        self.draw_signal_plot(canvas, signal_x, signal_y, signal_width, signal_height)
        
        
        freq_x = 10
        freq_y = video_height + 20
        freq_width = video_width
        freq_height = 210
        self.draw_frequency_spectrum(canvas, freq_x, freq_y, freq_width, freq_height)
        
        return canvas
    
    def draw_info_panel(self, canvas, x, y, width, height):
        """Gambar panel informasi."""
        
        cv2.rectangle(canvas, (x, y), (x+width, y+height), (240, 240, 240), -1)
        cv2.rectangle(canvas, (x, y), (x+width, y+height), (200, 200, 200), 2)
        
        
        cv2.putText(canvas, "Heart Rate Monitor", (x+10, y+30),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (50, 50, 50), 2)
        
        
        bpm_text = f"{self.current_bpm:.1f}" if self.current_bpm > 0 else "--"
        cv2.putText(canvas, bpm_text, (x+20, y+120),
                   cv2.FONT_HERSHEY_DUPLEX, 3, (0, 100, 255), 4)
        cv2.putText(canvas, "BPM", (x+20, y+155),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        
        
        status_y = y + 180
        if self.current_bpm > 0:
            status_text = "Measuring..."
            status_color = (0, 200, 0)
            
            cv2.circle(canvas, (x+250, y+100), 20, (0, 0, 255), -1)
            cv2.circle(canvas, (x+280, y+100), 20, (0, 0, 255), -1)
        else:
            status_text = "No face detected"
            status_color = (0, 0, 200)
        
        cv2.putText(canvas, status_text, (x+20, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        
        buffer_pct = int((len(self.signal_buffer) / (self.window_size * self.fps)) * 100)
        cv2.putText(canvas, f"Buffer: {buffer_pct}%", (x+20, status_y+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        
        cv2.putText(canvas, f"Samples: {len(self.signal_buffer)}", (x+20, status_y+45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    def draw_signal_plot(self, canvas, x, y, width, height):
        """Gambar plot sinyal rPPG."""
        
        cv2.rectangle(canvas, (x, y), (x+width, y+height), (250, 250, 250), -1)
        cv2.rectangle(canvas, (x, y), (x+width, y+height), (200, 200, 200), 2)
        
        
        cv2.putText(canvas, "rPPG Signal", (x+10, y+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
        
        if len(self.signal_buffer) < 2:
            cv2.putText(canvas, "Collecting data...", (x+width//2-80, y+height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            return
        
        
        signal_data = np.array(list(self.signal_buffer))
        time_data = np.array(list(self.time_buffer)) - self.time_buffer[0] if self.time_buffer else np.array([])
        
        
        valid_mask = np.isfinite(signal_data) & np.isfinite(time_data)
        if not np.any(valid_mask):
            return
        
        signal_data = signal_data[valid_mask]
        time_data = time_data[valid_mask]
        
        if len(signal_data) < 2:
            return
        
        
        plot_margin = 20
        plot_x = x + plot_margin
        plot_y = y + 35
        plot_w = width - 2 * plot_margin
        plot_h = height - 45
        
        
        sig_min, sig_max = np.min(signal_data), np.max(signal_data)
        if sig_max - sig_min > 1e-6:
            signal_norm = (signal_data - sig_min) / (sig_max - sig_min)
        else:
            signal_norm = np.ones_like(signal_data) * 0.5
        
        
        signal_norm = np.clip(signal_norm, 0, 1)
        
        
        for i in range(5):
            grid_y = plot_y + int(i * plot_h / 4)
            cv2.line(canvas, (plot_x, grid_y), (plot_x + plot_w, grid_y),
                    (220, 220, 220), 1)
        
        
        points = []
        time_window = 10  
        if len(time_data) > 0:
            time_max = time_data[-1]
            time_min = max(0, time_max - time_window)
            
            for i, (t, s) in enumerate(zip(time_data, signal_norm)):
                if t >= time_min:
                    px = plot_x + int((t - time_min) / time_window * plot_w)
                    py = plot_y + plot_h - int(s * plot_h)
                    points.append((px, py))
            
            
            if len(points) > 1:
                points_array = np.array(points, dtype=np.int32)
                cv2.polylines(canvas, [points_array], False, (0, 120, 255), 2)
    
    def draw_frequency_spectrum(self, canvas, x, y, width, height):
        """Gambar spektrum frekuensi."""
        
        cv2.rectangle(canvas, (x, y), (x+width, y+height), (250, 250, 250), -1)
        cv2.rectangle(canvas, (x, y), (x+width, y+height), (200, 200, 200), 2)
        
        
        cv2.putText(canvas, "Frequency Spectrum", (x+10, y+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
        
        if len(self.signal_buffer) < 64:
            cv2.putText(canvas, "Insufficient data for FFT...", (x+width//2-120, y+height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            return
        
        
        signal_data = np.array(list(self.signal_buffer))
        valid_mask = np.isfinite(signal_data)
        if not np.any(valid_mask):
            return
        
        signal_data = signal_data[valid_mask]
        
        n = len(signal_data)
        freq = fftfreq(n, 1/self.fps)
        
        with np.errstate(over='ignore', invalid='ignore'):
            fft_signal = np.abs(fft(signal_data))
        
        
        idx = np.where((freq > 0.5) & (freq < 4.0))[0]
        if len(idx) == 0:
            return
        
        freq = freq[idx]
        fft_signal = fft_signal[idx]
        
        
        max_fft = np.max(fft_signal)
        if np.isfinite(max_fft) and max_fft > 1e-10:
            fft_signal = fft_signal / max_fft
        else:
            return
        
        
        plot_margin = 20
        plot_x = x + plot_margin
        plot_y = y + 35
        plot_w = width - 2 * plot_margin
        plot_h = height - 55
        
        
        for i in range(5):
            grid_y = plot_y + int(i * plot_h / 4)
            cv2.line(canvas, (plot_x, grid_y), (plot_x + plot_w, grid_y),
                    (220, 220, 220), 1)
        
        
        freq_min, freq_max = 0.5, 4.0
        bar_width = max(2, plot_w // len(freq))
        
        for i, (f, amp) in enumerate(zip(freq, fft_signal)):
            bar_x = plot_x + int((f - freq_min) / (freq_max - freq_min) * plot_w)
            bar_h = int(amp * plot_h)
            bar_y = plot_y + plot_h - bar_h
            
            
            if 0.67 <= f <= 4.0:  
                color = (0, 200, 100)
            else:
                color = (180, 180, 180)
            
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, plot_y + plot_h),
                         color, -1)
        
        
        if self.current_bpm > 0:
            bpm_freq = self.current_bpm / 60
            if freq_min <= bpm_freq <= freq_max:
                marker_x = plot_x + int((bpm_freq - freq_min) / (freq_max - freq_min) * plot_w)
                cv2.line(canvas, (marker_x, plot_y), (marker_x, plot_y + plot_h),
                        (0, 0, 255), 2)
                cv2.putText(canvas, f"{self.current_bpm:.0f}", (marker_x - 15, plot_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        
        for bpm in [40, 60, 80, 100, 120, 150, 200]:
            f = bpm / 60
            if freq_min <= f <= freq_max:
                label_x = plot_x + int((f - freq_min) / (freq_max - freq_min) * plot_w)
                cv2.putText(canvas, str(bpm), (label_x - 10, y + height - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        
        cv2.putText(canvas, "BPM", (x + width - 50, y + height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    def skin_segmentation(self, frame_rgb):
        """
        Segmentasi kulit menggunakan HSV color space.
        
        Args:
            frame_rgb (numpy.ndarray): Frame dalam format RGB
            
        Returns:
            numpy.ndarray: Mask kulit (biner)
        """
        
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        
        
        lower_skin = np.array([0, 30, 80], dtype=np.uint8)
        upper_skin = np.array([20, 180, 255], dtype=np.uint8)
        
        
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)
        
        return mask
    
    def extract_rppg_signal(self, face_roi_rgb):
        """
        Ekstrak sinyal rPPG menggunakan metode CHROM.
        
        Args:
            face_roi_rgb (numpy.ndarray): ROI wajah dalam format RGB
            
        Returns:
            float: Nilai sinyal rPPG pada frame ini
        """
        
        
        
        
        R = face_roi_rgb[:,:,0].mean() / 255.0
        G = face_roi_rgb[:,:,1].mean() / 255.0
        B = face_roi_rgb[:,:,2].mean() / 255.0
        
        
        X_chrom = 3*R - 2*G
        Y_chrom = 1.5*R + G - 1.5*B
        
        
        chrom_magnitude = np.sqrt(X_chrom**2 + Y_chrom**2)
        if chrom_magnitude > 1e-6:  
            X_chrom_norm = X_chrom / chrom_magnitude
            Y_chrom_norm = Y_chrom / chrom_magnitude
        else:
            X_chrom_norm = 0
            Y_chrom_norm = 0
        
        
        s_chrom = X_chrom_norm - 0.75 * Y_chrom_norm
        
        return s_chrom
    
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        """
        Desain filter bandpass Butterworth.
        
        Args:
            lowcut (float): Frekuensi cutoff bawah (Hz)
            highcut (float): Frekuensi cutoff atas (Hz)
            fs (float): Sampling frequency
            order (int): Orde filter
            
        Returns:
            numpy.ndarray: Koefisien filter
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def bandpass_filter(self, data, lowcut=0.67, highcut=4.0, fs=30, order=5):
        """
        Terapkan filter bandpass pada sinyal.
        
        Args:
            data (numpy.ndarray): Sinyal input
            lowcut (float): Frekuensi cutoff bawah (Hz)
            highcut (float): Frekuensi cutoff atas (Hz)
            fs (float): Sampling frequency
            order (int): Orde filter
            
        Returns:
            numpy.ndarray: Sinyal hasil filter
        """
        if len(data) < fs:  
            return data
            
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        try:
            y = filtfilt(b, a, data)
            return y
        except:
            return data
    
    def detrend_signal(self, signal, window_size=30):
        """
        Hapus trend dari sinyal menggunakan moving average.
        
        Args:
            signal (numpy.ndarray): Sinyal input
            window_size (int): Ukuran window untuk moving average
            
        Returns:
            numpy.ndarray: Sinyal yang telah didetrend
        """
        if len(signal) < window_size:
            
            signal_mean = np.mean(signal)
            signal_std = np.std(signal)
            if signal_std > 1e-6:
                return (signal - signal_mean) / signal_std
            else:
                return signal - signal_mean
            
        
        moving_avg = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
        
        
        detrended = signal - moving_avg
        
        
        std_detrended = np.std(detrended)
        if std_detrended > 1e-6:
            detrended = detrended / std_detrended
        
        return detrended
    
    def estimate_bpm(self, signal, fps=30):
        """
        Estimasi BPM menggunakan FFT dan peak detection.
        
        Args:
            signal (numpy.ndarray): Sinyal rPPG yang telah diproses
            fps (float): Frame rate
            
        Returns:
            float: Estimasi BPM
        """
        if len(signal) < fps:  
            return 0
            
        
        if np.std(signal) < 1e-6:
            return 0
            
        
        n = len(signal)
        freq = fftfreq(n, 1/fps)
        fft_signal = np.abs(fft(signal))
        
        
        idx = np.where((freq > 0.67) & (freq < 4.0))[0]  
        if len(idx) == 0:
            return 0
            
        freq = freq[idx]
        fft_signal = fft_signal[idx]
        
        if len(freq) == 0:
            return 0
            
        
        max_idx = np.argmax(fft_signal)
        dominant_freq = freq[max_idx]
        
        
        bpm = dominant_freq * 60
        
        
        if bpm < 40 or bpm > 200:
            return 0
            
        
        peaks, _ = find_peaks(signal, distance=max(1, fps/4))  
        if len(peaks) > 2:  
            
            peak_intervals = np.diff(peaks) / fps  
            if len(peak_intervals) > 0 and np.mean(peak_intervals) > 0:
                avg_interval = np.mean(peak_intervals)
                bpm_peaks = 60 / avg_interval if avg_interval > 0 else 0
                
                
                if 40 <= bpm_peaks <= 200:
                    bpm = (bpm + bpm_peaks) / 2
        
        return bpm
    
    def detect_face_and_roi(self, frame_rgb):
        """
        Deteksi wajah dan ekstrak ROI menggunakan MediaPipe.
        
        Args:
            frame_rgb (numpy.ndarray): Frame dalam format RGB
            
        Returns:
            tuple: (face_roi_rgb, face_bbox) atau (None, None) jika tidak ada wajah terdeteksi
        """
        try:
            results = self.face_detection.process(frame_rgb)
            
            if not results.detections:
                return None, None
                
            
            detection = results.detections[0]
            
            
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame_rgb.shape
            
            
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            
            if x < 0: x = 0
            if y < 0: y = 0
            if x + width > w: width = w - x
            if y + height > h: height = h - y
            
            
            if width < 20 or height < 20:
                return None, None
                
            
            margin = int(0.1 * min(width, height))
            x = max(0, x - margin)
            y = max(0, y - margin)
            width = min(w - x, width + 2*margin)
            height = min(h - y, height + 2*margin)
            
            
            face_roi = frame_rgb[y:y+height, x:x+width]
            
            
            if len(self.roi_frames) == 0 or self.frame_count % 5 == 0:
                self.roi_frames.append(face_roi.copy())
            
            return face_roi, (x, y, width, height)
        except Exception as e:
            print(f"Error dalam deteksi wajah: {e}")
            return None, None
    
    def video_capture_thread(self):
        """Thread untuk menangkap frame dari webcam."""
        print("Thread capture dimulai")
        frame_count = 0
        last_time = time.time()
        fps_display = 0
        window_created = False
        
        try:
            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("Gagal membaca frame dari kamera")
                    time.sleep(0.1)
                    continue
                    
                current_time = time.time()
                frame_count += 1
                
                
                if current_time - last_time >= 1.0:
                    fps_display = frame_count
                    frame_count = 0
                    last_time = current_time
                
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (640, 480))
                
                
                frame_mirror = cv2.flip(frame, 1)
                
                
                cv2.putText(frame_mirror, f"FPS: {fps_display}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                
                if self.face_bbox is not None:
                    x, y, w, h = self.face_bbox
                    
                    x_mirror = 640 - x - w
                    cv2.rectangle(frame_mirror, (x_mirror, y), (x_mirror+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame_mirror, "Face Detected", (x_mirror, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                
                canvas = self.create_integrated_canvas(frame_mirror, frame_rgb)
                
                
                try:
                    cv2.imshow('rPPG Heart Rate Monitor - Press Q to Exit', canvas)
                    if not window_created:
                        window_created = True
                        print("Window terintegrasi berhasil dibuat")
                except Exception as e:
                    print(f"Error menampilkan canvas: {e}")
                    self.stop_event.set()
                    break
                
                
                if not self.frame_queue.full():
                    self.frame_queue.put((frame_rgb, time.time()))
                
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Tombol 'q' ditekan, menghentikan...")
                    self.stop_event.set()
                    break
                
                
                time.sleep(0.01)
        
        except Exception as e:
            print(f"Error di video_capture_thread: {e}")
            import traceback
            traceback.print_exc()
            self.stop_event.set()
        finally:
            print("Thread capture berhenti")
    
    def processing_thread(self):
        """Thread untuk memproses frame dan menghitung BPM."""
        print("Thread pemrosesan dimulai")
        last_bpm_time = time.time()
        last_face_time = time.time()  
        
        try:
            while not self.stop_event.is_set():
                try:
                    
                    frame_rgb, timestamp = self.frame_queue.get(timeout=1.0)
                    self.frame_count += 1
                    
                    
                    face_roi, self.face_bbox = self.detect_face_and_roi(frame_rgb)
                    
                    if face_roi is not None:
                        last_face_time = time.time()  
                        
                        rppg_value = self.extract_rppg_signal(face_roi)
                        
                        
                        if np.isfinite(rppg_value) and abs(rppg_value) < 100:  
                            
                            self.signal_buffer.append(rppg_value)
                            self.time_buffer.append(timestamp)
                        else:
                            
                            continue
                        
                        
                        if len(self.signal_buffer) >= self.fps * 2:  
                            signal_array = np.array(self.signal_buffer)
                            
                            
                            if np.std(signal_array) < 1e-10:
                                
                                print("Warning: Sinyal terlalu flat, membersihkan buffer...")
                                self.signal_buffer.clear()
                                self.time_buffer.clear()
                                self.current_bpm = 0
                                continue
                            
                            
                            detrended_signal = self.detrend_signal(signal_array, window_size=self.fps)
                            
                            
                            if not np.all(np.isfinite(detrended_signal)):
                                print("Warning: Detrending menghasilkan nilai tidak valid")
                                continue
                            
                            
                            filtered_signal = self.bandpass_filter(
                                detrended_signal, lowcut=0.67, highcut=4.0, fs=self.fps, order=3
                            )
                            
                            
                            if not np.all(np.isfinite(filtered_signal)):
                                print("Warning: Filter menghasilkan nilai tidak valid")
                                continue
                            
                            
                            for i in range(min(len(filtered_signal), len(self.signal_buffer))):
                                self.signal_buffer[i] = filtered_signal[i]
                            
                            
                            current_time = time.time()
                            if current_time - last_bpm_time >= 0.5:
                                
                                new_bpm = self.estimate_bpm(filtered_signal, fps=self.fps)
                                
                                
                                if new_bpm > 0:
                                    if abs(new_bpm - self.current_bpm) < 40 or self.current_bpm == 0:
                                        self.current_bpm = new_bpm
                                        self.bpm_history.append(new_bpm)
                                    
                                    
                                    if len(self.bpm_history) >= 3:
                                        median_bpm = np.median(self.bpm_history)
                                        
                                        if abs(self.current_bpm - median_bpm) > 20:
                                            self.current_bpm = median_bpm
                                
                                last_bpm_time = current_time
                    else:
                        
                        
                        if time.time() - last_face_time > 3.0:
                            if len(self.signal_buffer) > 0:
                                print("Info: Tidak ada wajah terdeteksi, membersihkan buffer...")
                                self.signal_buffer.clear()
                                self.time_buffer.clear()
                                self.current_bpm = 0
                                self.bpm_history.clear()
                    
                    
                    time.sleep(0.01)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Error dalam pemrosesan: {e}")
                    time.sleep(0.1)
        
        except Exception as e:
            print(f"Error fatal di processing_thread: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Thread pemrosesan berhenti")
    
    def run(self):
        """Jalankan sistem rPPG real-time."""
        print("="*60)
        print("  rPPG Heart Rate Monitor - Integrated Display")
        print("="*60)
        print("Tekan 'Q' pada jendela untuk keluar")
        print("Fitur:")
        print("  - Mirror camera view")
        print("  - Real-time heart rate (BPM)")
        print("  - Signal waveform visualization")
        print("  - Frequency spectrum analysis")
        print("="*60)
        
        
        capture_thread = threading.Thread(target=self.video_capture_thread)
        process_thread = threading.Thread(target=self.processing_thread)
        
        capture_thread.daemon = True
        process_thread.daemon = True
        
        capture_thread.start()
        process_thread.start()
        
        try:
            
            print("Menunggu window...")
            
            while not self.stop_event.is_set():
                
                if not capture_thread.is_alive():
                    print("Thread capture mati, menghentikan...")
                    self.stop_event.set()
                    break
                
                time.sleep(0.1)
            
            print("Loop utama selesai")
                
        except KeyboardInterrupt:
            print("\nMenerima sinyal KeyboardInterrupt")
            self.stop_event.set()
        except Exception as e:
            print(f"Error di loop utama: {e}")
            import traceback
            traceback.print_exc()
            self.stop_event.set()
        finally:
            
            self.cleanup()
    
    def cleanup(self):
        """Bersihkan resource."""
        print("Membersihkan resource...")
        self.stop_event.set()
        
        
        print("Menunggu thread selesai...")
        time.sleep(1.0)
        
        
        if self.cap.isOpened():
            self.cap.release()
        
        
        cv2.destroyAllWindows()
        
        
        time.sleep(0.5)
        
        print("Semua resource telah dibersihkan.")


if __name__ == "__main__":
    try:
        
        rppg_system = RealTimeRPPG(window_size=30, fps=30)
        rppg_system.run()
    except Exception as e:
        print(f"Terjadi error kritis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Program selesai.")