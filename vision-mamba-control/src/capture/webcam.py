"""
Webcam Capture System - 웹캠에서 실시간 비디오 스트림 캡처

OpenCV를 사용하여 웹캠 입력을 받고 전처리
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import time


class WebcamCapture:
    """웹캠 캡처 및 전처리"""

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30
    ):
        """
        Args:
            camera_id: 카메라 디바이스 ID (기본: 0)
            width: 캡처 해상도 너비
            height: 캡처 해상도 높이
            fps: 목표 FPS
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_running = False

        # 통계
        self.frame_count = 0
        self.start_time = None
        self.last_brightness = 0.5
        self.last_contrast = 0.5
        self.last_saturation = 0.5

    def start(self) -> bool:
        """
        웹캠 시작

        Returns:
            성공 여부
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)

            if not self.cap.isOpened():
                print(f"Error: Cannot open camera {self.camera_id}")
                return False

            # 해상도 설정
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # 실제 설정된 값 확인
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            print(f"Camera {self.camera_id} opened successfully")
            print(f"Resolution: {actual_width}x{actual_height} @ {actual_fps} FPS")

            self.is_running = True
            self.start_time = time.time()
            self.frame_count = 0

            return True

        except Exception as e:
            print(f"Error starting camera: {e}")
            return False

    def stop(self):
        """웹캠 중지"""
        if self.cap is not None:
            self.cap.release()
            self.is_running = False
            print(f"Camera {self.camera_id} stopped")

            if self.start_time:
                duration = time.time() - self.start_time
                avg_fps = self.frame_count / duration if duration > 0 else 0
                print(f"Captured {self.frame_count} frames in {duration:.1f}s ({avg_fps:.1f} FPS)")

    def read_frame(self) -> Optional[np.ndarray]:
        """
        프레임 읽기

        Returns:
            frame (H, W, 3) in BGR or None if failed
        """
        if not self.is_running or self.cap is None:
            return None

        ret, frame = self.cap.read()

        if not ret:
            print("Failed to read frame")
            return None

        self.frame_count += 1
        return frame

    def compute_camera_stats(self, frame: np.ndarray) -> Tuple[float, float, float]:
        """
        카메라 통계 계산 (FiLM 레이어용)

        Args:
            frame: (H, W, 3) in BGR

        Returns:
            (brightness, contrast, saturation) - 모두 0-1 범위로 정규화
        """
        # BGR → HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Brightness: V 채널의 평균 (0-255 → 0-1)
        brightness = np.mean(hsv[:, :, 2]) / 255.0

        # Contrast: V 채널의 표준편차 (0-127 → 0-1)
        contrast = np.std(hsv[:, :, 2]) / 127.0

        # Saturation: S 채널의 평균 (0-255 → 0-1)
        saturation = np.mean(hsv[:, :, 1]) / 255.0

        # 캐시 (GUI 표시용)
        self.last_brightness = brightness
        self.last_contrast = contrast
        self.last_saturation = saturation

        return brightness, contrast, saturation

    def get_fps(self) -> float:
        """현재 실제 FPS 계산"""
        if self.start_time is None or self.frame_count == 0:
            return 0.0

        duration = time.time() - self.start_time
        return self.frame_count / duration if duration > 0 else 0.0

    def draw_overlay(
        self,
        frame: np.ndarray,
        steering: float,
        throttle: float,
        brake: float,
        fps: float = None
    ) -> np.ndarray:
        """
        프레임에 오버레이 그리기 (제어 신호 시각화)

        Args:
            frame: 원본 프레임
            steering: -1 ~ 1
            throttle: 0 ~ 1
            brake: 0 ~ 1
            fps: 현재 FPS (optional)

        Returns:
            오버레이가 그려진 프레임
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # 반투명 배경
        cv2.rectangle(overlay, (10, 10), (300, 150), (0, 0, 0), -1)
        frame_display = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # FPS 표시
        if fps is None:
            fps = self.get_fps()
        cv2.putText(frame_display, f"FPS: {fps:.1f}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Steering 표시
        steer_color = (0, 255, 0) if abs(steering) < 0.3 else (0, 165, 255)
        cv2.putText(frame_display, f"Steering: {steering:+.2f}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, steer_color, 2)

        # Throttle 표시
        throttle_color = (0, 255, 0) if throttle < 0.5 else (0, 165, 255)
        cv2.putText(frame_display, f"Throttle: {throttle:.2f}", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, throttle_color, 2)

        # Brake 표시
        brake_color = (0, 255, 0) if brake < 0.3 else (0, 0, 255)
        cv2.putText(frame_display, f"Brake: {brake:.2f}", (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, brake_color, 2)

        # Steering 바 (가로)
        bar_x = w - 250
        bar_y = h - 80
        bar_w = 200
        bar_h = 20

        # 배경
        cv2.rectangle(frame_display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                     (100, 100, 100), -1)

        # 중앙선
        center_x = bar_x + bar_w // 2
        cv2.line(frame_display, (center_x, bar_y), (center_x, bar_y + bar_h),
                (255, 255, 255), 2)

        # Steering 표시
        steer_x = int(center_x + steering * (bar_w // 2))
        cv2.circle(frame_display, (steer_x, bar_y + bar_h // 2), 12,
                  (0, 255, 255), -1)

        # Throttle/Brake 바 (세로)
        bar_x2 = w - 40
        bar_y2 = h - 200
        bar_w2 = 20
        bar_h2 = 150

        # Throttle (초록)
        throttle_h = int(throttle * bar_h2)
        cv2.rectangle(frame_display, (bar_x2, bar_y2 + bar_h2 - throttle_h),
                     (bar_x2 + bar_w2, bar_y2 + bar_h2), (0, 255, 0), -1)

        # Brake (빨강)
        brake_h = int(brake * bar_h2)
        cv2.rectangle(frame_display, (bar_x2 - 30, bar_y2 + bar_h2 - brake_h),
                     (bar_x2 - 10, bar_y2 + bar_h2), (0, 0, 255), -1)

        # 레이블
        cv2.putText(frame_display, "T", (bar_x2 + 5, bar_y2 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame_display, "B", (bar_x2 - 25, bar_y2 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame_display

    def __del__(self):
        """소멸자 - 자동으로 카메라 해제"""
        self.stop()


if __name__ == "__main__":
    # 테스트
    print("Testing webcam capture...")
    print("Press 'q' to quit")

    webcam = WebcamCapture(camera_id=0)

    if not webcam.start():
        print("Failed to start webcam")
        exit(1)

    try:
        while True:
            frame = webcam.read_frame()

            if frame is None:
                break

            # 카메라 통계 계산
            brightness, contrast, saturation = webcam.compute_camera_stats(frame)

            # 더미 제어 신호 (테스트용)
            steering = np.sin(time.time() * 0.5) * 0.8  # -0.8 ~ 0.8
            throttle = (np.sin(time.time() * 0.3) + 1) / 2  # 0 ~ 1
            brake = max(0, -steering * 0.5)  # steering에 따라

            # 오버레이 그리기
            frame_display = webcam.draw_overlay(frame, steering, throttle, brake)

            # 카메라 통계 표시
            cv2.putText(frame_display, f"Bright: {brightness:.2f}", (20, h - 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame_display, f"Contrast: {contrast:.2f}", (20, h - 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame_display, f"Saturation: {saturation:.2f}", (20, h - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 화면에 표시
            h, w = frame_display.shape[:2]
            cv2.imshow('Vision Mamba Control - Webcam Test', frame_display)

            # 'q' 키로 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        webcam.stop()
        cv2.destroyAllWindows()

    print("Test completed")
