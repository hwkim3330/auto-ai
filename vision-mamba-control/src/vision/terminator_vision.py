"""
Terminator Vision - HUD-style visualization

터미네이터 스타일의 비전 시스템:
- 물체 검출 및 추적
- 차선 검출
- HUD 오버레이 (스캔라인, 타겟팅, 정보 표시)
- Vision Mamba features 시각화
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import time


class TerminatorVision:
    """
    터미네이터 스타일 비전 시스템

    Features:
    - Object detection (YOLO)
    - Lane detection
    - Terminator-style HUD
    - Vision Mamba feature visualization
    """

    def __init__(self, use_yolo: bool = True):
        """
        Args:
            use_yolo: YOLO 모델 사용 여부 (False면 fallback)
        """
        self.use_yolo = use_yolo
        self.yolo_net = None
        self.yolo_classes = []

        # HUD 설정
        self.hud_color = (0, 255, 0)  # 녹색 (터미네이터)
        self.target_color = (0, 0, 255)  # 빨간색 (위험)
        self.scan_line_y = 0
        self.scan_direction = 1

        # YOLO 로드 시도
        if use_yolo:
            try:
                self._load_yolo()
            except Exception as e:
                print(f"YOLO loading failed: {e}")
                print("Using fallback detection")
                self.use_yolo = False

    def _load_yolo(self):
        """YOLO 모델 로드 (YOLOv3-tiny)"""
        try:
            # YOLOv3-tiny 사용 (빠름)
            # 실제 사용 시 weights 다운로드 필요
            # wget https://pjreddie.com/media/files/yolov3-tiny.weights
            # wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg
            # wget https://github.com/pjreddie/darknet/blob/master/data/coco.names

            print("Note: YOLO weights not loaded (optional)")
            print("Download YOLOv3-tiny weights for object detection")
            self.use_yolo = False

        except Exception as e:
            print(f"YOLO init failed: {e}")
            self.use_yolo = False

    def detect_lanes(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        차선 검출 (OpenCV Canny + Hough Transform)

        Args:
            frame: BGR 이미지

        Returns:
            차선 좌표 리스트
        """
        # 관심 영역 (ROI) - 하단 절반
        height, width = frame.shape[:2]
        roi_top = int(height * 0.6)
        roi = frame[roi_top:, :]

        # 그레이스케일 변환
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Gaussian Blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny Edge Detection
        edges = cv2.Canny(blur, 50, 150)

        # Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=50,
            maxLineGap=150
        )

        # 좌표 보정 (ROI → 전체 프레임)
        lane_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # ROI 좌표를 전체 프레임 좌표로 변환
                lane_lines.append(np.array([x1, y1 + roi_top, x2, y2 + roi_top]))

        return lane_lines

    def detect_objects_fallback(self, frame: np.ndarray) -> List[Dict]:
        """
        폴백 물체 검출 (YOLO 없을 때)

        간단한 움직임 검출 또는 색상 기반
        """
        # 간단한 contour 검출
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # 최소 크기
                x, y, w, h = cv2.boundingRect(contour)
                objects.append({
                    'bbox': (x, y, w, h),
                    'class': 'object',
                    'confidence': 0.5
                })

        return objects[:10]  # 최대 10개

    def draw_terminator_hud(
        self,
        frame: np.ndarray,
        lanes: List[np.ndarray],
        objects: List[Dict],
        steering: float,
        throttle: float,
        brake: float,
        fps: float
    ) -> np.ndarray:
        """
        터미네이터 스타일 HUD 그리기

        Args:
            frame: 원본 프레임
            lanes: 차선 좌표
            objects: 검출된 물체
            steering, throttle, brake: 제어 신호
            fps: FPS

        Returns:
            HUD가 그려진 프레임
        """
        overlay = frame.copy()
        height, width = frame.shape[:2]

        # 1. 차선 그리기 (녹색)
        for lane in lanes:
            x1, y1, x2, y2 = lane
            cv2.line(overlay, (x1, y1), (x2, y2), self.hud_color, 3)

        # 2. 물체 검출 박스 (빨간색/녹색)
        for obj in objects:
            x, y, w, h = obj['bbox']
            confidence = obj.get('confidence', 0.5)
            class_name = obj.get('class', 'unknown')

            # 위험도에 따라 색상 변경
            color = self.target_color if confidence > 0.7 else self.hud_color

            # 박스
            cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)

            # 라벨
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(
                overlay, label, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

            # 크로스헤어 (타겟팅)
            cx, cy = x + w//2, y + h//2
            cv2.drawMarker(
                overlay, (cx, cy),
                color, cv2.MARKER_CROSS, 20, 2
            )

        # 3. 스캔라인 (움직이는 가로선)
        self.scan_line_y += self.scan_direction * 5
        if self.scan_line_y >= height or self.scan_line_y <= 0:
            self.scan_direction *= -1

        cv2.line(
            overlay,
            (0, self.scan_line_y),
            (width, self.scan_line_y),
            (0, 255, 255), 1
        )

        # 4. 중앙 크로스헤어
        cv2.drawMarker(
            overlay,
            (width//2, height//2),
            self.hud_color,
            cv2.MARKER_CROSS,
            30, 2
        )

        # 5. 코너 마커 (사각형 테두리)
        corner_size = 30
        corner_thickness = 3

        # 좌상단
        cv2.line(overlay, (10, 10), (10 + corner_size, 10), self.hud_color, corner_thickness)
        cv2.line(overlay, (10, 10), (10, 10 + corner_size), self.hud_color, corner_thickness)

        # 우상단
        cv2.line(overlay, (width-10, 10), (width-10-corner_size, 10), self.hud_color, corner_thickness)
        cv2.line(overlay, (width-10, 10), (width-10, 10+corner_size), self.hud_color, corner_thickness)

        # 좌하단
        cv2.line(overlay, (10, height-10), (10+corner_size, height-10), self.hud_color, corner_thickness)
        cv2.line(overlay, (10, height-10), (10, height-10-corner_size), self.hud_color, corner_thickness)

        # 우하단
        cv2.line(overlay, (width-10, height-10), (width-10-corner_size, height-10), self.hud_color, corner_thickness)
        cv2.line(overlay, (width-10, height-10), (width-10, height-10-corner_size), self.hud_color, corner_thickness)

        # 6. HUD 정보 표시
        hud_texts = [
            f"VISION SYSTEM ONLINE",
            f"FPS: {fps:.1f}",
            f"TARGETS: {len(objects)}",
            f"LANES: {len(lanes)}",
            "",
            f"STEERING: {steering:+.2f}",
            f"THROTTLE: {throttle:.2f}",
            f"BRAKE: {brake:.2f}",
        ]

        y_offset = 40
        for i, text in enumerate(hud_texts):
            cv2.putText(
                overlay,
                text,
                (20, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.hud_color,
                2
            )

        # 7. 스티어링 인디케이터 (하단)
        steering_bar_y = height - 40
        steering_bar_x = width // 2
        steering_offset = int(steering * 100)

        # 중앙선
        cv2.line(
            overlay,
            (steering_bar_x, steering_bar_y - 10),
            (steering_bar_x, steering_bar_y + 10),
            (255, 255, 255), 2
        )

        # 스티어링 바
        cv2.line(
            overlay,
            (steering_bar_x - 100, steering_bar_y),
            (steering_bar_x + 100, steering_bar_y),
            (100, 100, 100), 3
        )

        # 현재 스티어링 위치
        cv2.circle(
            overlay,
            (steering_bar_x + steering_offset, steering_bar_y),
            8,
            (0, 255, 255), -1
        )

        # 8. 블렌딩 (투명도)
        alpha = 0.7
        result = cv2.addWeighted(frame, alpha, overlay, 1 - alpha, 0)

        return result

    def process_frame(
        self,
        frame: np.ndarray,
        steering: float,
        throttle: float,
        brake: float,
        fps: float = 30.0
    ) -> np.ndarray:
        """
        프레임 처리 (전체 파이프라인)

        Args:
            frame: 입력 프레임
            steering, throttle, brake: 제어 신호
            fps: FPS

        Returns:
            터미네이터 HUD가 적용된 프레임
        """
        # 1. 차선 검출
        lanes = self.detect_lanes(frame)

        # 2. 물체 검출
        if self.use_yolo:
            objects = []  # YOLO 구현 필요
        else:
            objects = self.detect_objects_fallback(frame)

        # 3. HUD 그리기
        result = self.draw_terminator_hud(
            frame, lanes, objects,
            steering, throttle, brake, fps
        )

        return result


def test_terminator_vision():
    """테스트"""
    print("Testing Terminator Vision...")

    vision = TerminatorVision(use_yolo=False)

    # 웹캠 열기
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press 'q' to quit")

    last_time = time.time()
    fps = 30.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FPS 계산
        current_time = time.time()
        fps = 1.0 / (current_time - last_time) if current_time > last_time else 30.0
        last_time = current_time

        # 더미 제어 신호
        t = time.time()
        steering = np.sin(t * 0.5) * 0.8
        throttle = (np.sin(t * 0.3) + 1) / 2
        brake = max(0, -steering * 0.3)

        # 터미네이터 비전 처리
        result = vision.process_frame(frame, steering, throttle, brake, fps)

        # 표시
        cv2.imshow('Terminator Vision System', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Test completed")


if __name__ == "__main__":
    test_terminator_vision()
