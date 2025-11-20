"""
Vision Mamba Control - GUI Application

실시간 웹캠 비전 제어 시스템 GUI
- 웹캠 비디오 스트림
- Vision Mamba 실시간 추론
- 제어 신호 시각화
- 성능 모니터링
"""

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import time
import threading
from collections import deque
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capture.webcam import WebcamCapture


class VisionMambaGUI:
    """Vision Mamba Control GUI"""

    def __init__(self, root, demo_mode=True):
        """
        Args:
            root: Tkinter root window
            demo_mode: True면 모델 없이 데모 모드로 실행
        """
        self.root = root
        self.demo_mode = demo_mode
        self.root.title("Vision Mamba Control - Real-time Vision Control System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')

        # 웹캠
        self.webcam = None
        self.is_running = False
        self.update_thread = None

        # 모델 (demo 모드면 None)
        self.model = None
        if not demo_mode:
            try:
                import torch
                from models.control_model import create_control_model_tiny
                self.model = create_control_model_tiny(use_film=True)
                self.model.eval()
                print("Model loaded successfully")
            except Exception as e:
                print(f"Failed to load model: {e}")
                print("Running in demo mode")
                self.demo_mode = True

        # 통계
        self.fps_history = deque(maxlen=30)
        self.steering_history = deque(maxlen=100)
        self.throttle_history = deque(maxlen=100)
        self.brake_history = deque(maxlen=100)
        self.inference_time_history = deque(maxlen=30)

        self.last_frame_time = time.time()

        # GUI 구축
        self._build_gui()

    def _build_gui(self):
        """GUI 레이아웃 구축"""

        # Top bar - Title and status
        top_frame = tk.Frame(self.root, bg='#2d2d2d', height=60)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        title_label = tk.Label(
            top_frame,
            text="🚗 Vision Mamba Control System",
            font=('Arial', 24, 'bold'),
            bg='#2d2d2d',
            fg='#00ff00'
        )
        title_label.pack(side=tk.LEFT, padx=20)

        mode_text = "DEMO MODE" if self.demo_mode else "AI MODE"
        mode_color = "#ffaa00" if self.demo_mode else "#00ff00"
        self.mode_label = tk.Label(
            top_frame,
            text=mode_text,
            font=('Arial', 16, 'bold'),
            bg='#2d2d2d',
            fg=mode_color
        )
        self.mode_label.pack(side=tk.RIGHT, padx=20)

        # Main content area
        content_frame = tk.Frame(self.root, bg='#1e1e1e')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left panel - Video feed
        left_panel = tk.Frame(content_frame, bg='#2d2d2d')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        video_label = tk.Label(
            left_panel,
            text="📹 Video Feed",
            font=('Arial', 14, 'bold'),
            bg='#2d2d2d',
            fg='#ffffff'
        )
        video_label.pack(pady=10)

        self.video_canvas = tk.Label(left_panel, bg='#000000')
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Right panel - Controls and stats
        right_panel = tk.Frame(content_frame, bg='#2d2d2d', width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        right_panel.pack_propagate(False)

        # Control buttons
        control_frame = tk.LabelFrame(
            right_panel,
            text="Control",
            font=('Arial', 12, 'bold'),
            bg='#2d2d2d',
            fg='#ffffff'
        )
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        self.start_button = tk.Button(
            control_frame,
            text="▶ Start",
            font=('Arial', 12, 'bold'),
            bg='#00aa00',
            fg='#ffffff',
            command=self.start_capture,
            width=15,
            height=2
        )
        self.start_button.pack(pady=5)

        self.stop_button = tk.Button(
            control_frame,
            text="⏸ Stop",
            font=('Arial', 12, 'bold'),
            bg='#aa0000',
            fg='#ffffff',
            command=self.stop_capture,
            width=15,
            height=2,
            state=tk.DISABLED
        )
        self.stop_button.pack(pady=5)

        # Stats panel
        stats_frame = tk.LabelFrame(
            right_panel,
            text="Performance",
            font=('Arial', 12, 'bold'),
            bg='#2d2d2d',
            fg='#ffffff'
        )
        stats_frame.pack(fill=tk.X, padx=10, pady=10)

        self.fps_label = tk.Label(
            stats_frame,
            text="FPS: 0.0",
            font=('Arial', 11),
            bg='#2d2d2d',
            fg='#00ff00',
            anchor='w'
        )
        self.fps_label.pack(fill=tk.X, padx=10, pady=3)

        self.inference_label = tk.Label(
            stats_frame,
            text="Inference: 0 ms",
            font=('Arial', 11),
            bg='#2d2d2d',
            fg='#00ff00',
            anchor='w'
        )
        self.inference_label.pack(fill=tk.X, padx=10, pady=3)

        # Control signals panel
        signals_frame = tk.LabelFrame(
            right_panel,
            text="Control Signals",
            font=('Arial', 12, 'bold'),
            bg='#2d2d2d',
            fg='#ffffff'
        )
        signals_frame.pack(fill=tk.X, padx=10, pady=10)

        self.steering_label = tk.Label(
            signals_frame,
            text="Steering: 0.00",
            font=('Arial', 11),
            bg='#2d2d2d',
            fg='#ffff00',
            anchor='w'
        )
        self.steering_label.pack(fill=tk.X, padx=10, pady=3)

        self.steering_bar = ttk.Progressbar(
            signals_frame,
            orient='horizontal',
            mode='determinate',
            length=300
        )
        self.steering_bar.pack(padx=10, pady=3)

        self.throttle_label = tk.Label(
            signals_frame,
            text="Throttle: 0.00",
            font=('Arial', 11),
            bg='#2d2d2d',
            fg='#00ff00',
            anchor='w'
        )
        self.throttle_label.pack(fill=tk.X, padx=10, pady=3)

        self.throttle_bar = ttk.Progressbar(
            signals_frame,
            orient='horizontal',
            mode='determinate',
            length=300
        )
        self.throttle_bar.pack(padx=10, pady=3)

        self.brake_label = tk.Label(
            signals_frame,
            text="Brake: 0.00",
            font=('Arial', 11),
            bg='#2d2d2d',
            fg='#ff0000',
            anchor='w'
        )
        self.brake_label.pack(fill=tk.X, padx=10, pady=3)

        self.brake_bar = ttk.Progressbar(
            signals_frame,
            orient='horizontal',
            mode='determinate',
            length=300
        )
        self.brake_bar.pack(padx=10, pady=3)

        # Camera stats panel
        camera_frame = tk.LabelFrame(
            right_panel,
            text="Camera Stats",
            font=('Arial', 12, 'bold'),
            bg='#2d2d2d',
            fg='#ffffff'
        )
        camera_frame.pack(fill=tk.X, padx=10, pady=10)

        self.brightness_label = tk.Label(
            camera_frame,
            text="Brightness: 0.00",
            font=('Arial', 10),
            bg='#2d2d2d',
            fg='#aaaaaa',
            anchor='w'
        )
        self.brightness_label.pack(fill=tk.X, padx=10, pady=2)

        self.contrast_label = tk.Label(
            camera_frame,
            text="Contrast: 0.00",
            font=('Arial', 10),
            bg='#2d2d2d',
            fg='#aaaaaa',
            anchor='w'
        )
        self.contrast_label.pack(fill=tk.X, padx=10, pady=2)

        self.saturation_label = tk.Label(
            camera_frame,
            text="Saturation: 0.00",
            font=('Arial', 10),
            bg='#2d2d2d',
            fg='#aaaaaa',
            anchor='w'
        )
        self.saturation_label.pack(fill=tk.X, padx=10, pady=2)

    def start_capture(self):
        """웹캠 캡처 시작"""
        if self.is_running:
            return

        self.webcam = WebcamCapture(camera_id=0, width=640, height=480, fps=30)

        if not self.webcam.start():
            print("Failed to start webcam")
            return

        self.is_running = True
        self.last_frame_time = time.time()

        # UI 업데이트
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # 별도 스레드에서 업데이트
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

        print("Capture started")

    def stop_capture(self):
        """웹캠 캡처 중지"""
        if not self.is_running:
            return

        self.is_running = False

        if self.update_thread:
            self.update_thread.join(timeout=2.0)

        if self.webcam:
            self.webcam.stop()
            self.webcam = None

        # UI 업데이트
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        print("Capture stopped")

    def _update_loop(self):
        """메인 업데이트 루프 (별도 스레드)"""
        while self.is_running:
            try:
                # 프레임 읽기
                frame = self.webcam.read_frame()

                if frame is None:
                    time.sleep(0.01)
                    continue

                # FPS 계산
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_frame_time) if current_time > self.last_frame_time else 0
                self.last_frame_time = current_time
                self.fps_history.append(fps)

                # 카메라 통계
                brightness, contrast, saturation = self.webcam.compute_camera_stats(frame)

                # 제어 신호 예측
                inference_start = time.time()

                if self.demo_mode:
                    # Demo mode - 더미 신호
                    t = time.time()
                    steering = np.sin(t * 0.5) * 0.7
                    throttle = (np.sin(t * 0.3) + 1) / 2
                    brake = max(0, -steering * 0.3)
                else:
                    # AI mode - 실제 모델 추론
                    steering, throttle, brake = self.model.predict_from_webcam(
                        frame, brightness, contrast, saturation
                    )

                inference_time = (time.time() - inference_start) * 1000  # ms
                self.inference_time_history.append(inference_time)

                # 히스토리 저장
                self.steering_history.append(steering)
                self.throttle_history.append(throttle)
                self.brake_history.append(brake)

                # 비디오 오버레이
                frame_display = self.webcam.draw_overlay(frame, steering, throttle, brake, fps)

                # GUI 업데이트 (메인 스레드에서)
                self.root.after(0, self._update_gui, frame_display, fps, inference_time,
                               steering, throttle, brake, brightness, contrast, saturation)

            except Exception as e:
                print(f"Error in update loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def _update_gui(self, frame, fps, inference_time, steering, throttle, brake,
                    brightness, contrast, saturation):
        """GUI 업데이트 (메인 스레드에서 호출)"""

        # 비디오 프레임 표시
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        # Canvas 크기에 맞게 리사이즈
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            # 종횡비 유지하면서 리사이즈
            img_ratio = image.width / image.height
            canvas_ratio = canvas_width / canvas_height

            if img_ratio > canvas_ratio:
                new_width = canvas_width
                new_height = int(canvas_width / img_ratio)
            else:
                new_height = canvas_height
                new_width = int(canvas_height * img_ratio)

            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(image)
        self.video_canvas.config(image=photo)
        self.video_canvas.image = photo  # Keep reference

        # Performance stats
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        avg_inference = np.mean(self.inference_time_history) if self.inference_time_history else 0

        self.fps_label.config(text=f"FPS: {avg_fps:.1f}")
        self.inference_label.config(text=f"Inference: {avg_inference:.1f} ms")

        # Control signals
        self.steering_label.config(text=f"Steering: {steering:+.2f}")
        self.steering_bar['value'] = (steering + 1) * 50  # -1~1 → 0~100

        self.throttle_label.config(text=f"Throttle: {throttle:.2f}")
        self.throttle_bar['value'] = throttle * 100

        self.brake_label.config(text=f"Brake: {brake:.2f}")
        self.brake_bar['value'] = brake * 100

        # Camera stats
        self.brightness_label.config(text=f"Brightness: {brightness:.2f}")
        self.contrast_label.config(text=f"Contrast: {contrast:.2f}")
        self.saturation_label.config(text=f"Saturation: {saturation:.2f}")

    def on_closing(self):
        """창 닫기 이벤트"""
        self.stop_capture()
        self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()

    # AI mode - 실제 Vision Mamba 모델 사용
    app = VisionMambaGUI(root, demo_mode=False)

    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
