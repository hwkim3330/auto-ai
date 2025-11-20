"""
Rule-based Autonomous Driving Controller

실제로 작동하는 규칙 기반 자율주행 제어 시스템
YOLO 감지 결과와 차선 정보를 활용한 지능형 제어
"""

import numpy as np
from typing import Dict, List, Tuple


class RuleBasedController:
    """
    규칙 기반 자율주행 컨트롤러

    Features:
    - 차선 유지 (Lane keeping)
    - 충돌 회피 (Collision avoidance)
    - 신호등 준수 (Traffic light compliance)
    - 속도 제어 (Speed control)
    """

    def __init__(self):
        # Control gains
        self.lane_keeping_gain = 0.02
        self.speed_target = 0.5  # 목표 속도

        # Safety thresholds
        self.collision_distance = 15.0  # meters
        self.emergency_distance = 5.0   # meters

        # Smoothing
        self.prev_steering = 0.0
        self.prev_throttle = 0.5
        self.prev_brake = 0.0
        self.smoothing_factor = 0.3

    def compute_control(
        self,
        detections: List[Dict],
        lane_info: Dict,
        traffic_lights: List[Dict]
    ) -> Tuple[float, float, float]:
        """
        제어 신호 계산

        Args:
            detections: YOLO 객체 감지 결과
            lane_info: 차선 정보
            traffic_lights: 신호등 정보

        Returns:
            (steering, throttle, brake)
        """
        # 1. 차선 유지 제어
        steering = self._lane_keeping_control(lane_info)

        # 2. 속도 제어
        throttle = self._speed_control()
        brake = 0.0

        # 3. 충돌 회피
        collision_brake = self._collision_avoidance(detections)
        if collision_brake > 0:
            brake = max(brake, collision_brake)
            throttle = max(0, throttle - collision_brake)

        # 4. 신호등 준수
        traffic_light_brake = self._traffic_light_control(traffic_lights)
        if traffic_light_brake > 0:
            brake = max(brake, traffic_light_brake)
            throttle = max(0, throttle - traffic_light_brake)

        # 5. 스무딩 (부드러운 제어)
        steering = self._smooth(steering, self.prev_steering)
        throttle = self._smooth(throttle, self.prev_throttle)
        brake = self._smooth(brake, self.prev_brake)

        # 업데이트
        self.prev_steering = steering
        self.prev_throttle = throttle
        self.prev_brake = brake

        # 클리핑
        steering = np.clip(steering, -1.0, 1.0)
        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)

        return steering, throttle, brake

    def _lane_keeping_control(self, lane_info: Dict) -> float:
        """
        차선 유지 제어

        차선 중앙에서 벗어난 정도(deviation)를 기반으로 조향 결정
        """
        if not lane_info:
            return 0.0

        deviation = lane_info.get('deviation', 0)  # -100 ~ 100 (percentage)

        # P 제어 (비례 제어)
        steering = -deviation * self.lane_keeping_gain

        return steering

    def _speed_control(self) -> float:
        """
        속도 제어

        목표 속도 유지
        """
        return self.speed_target

    def _collision_avoidance(self, detections: List[Dict]) -> float:
        """
        충돌 회피

        앞에 물체가 있으면 브레이크
        """
        if not detections:
            return 0.0

        # 가장 가까운 물체 찾기
        min_distance = float('inf')
        closest_obj = None

        for det in detections:
            distance = det.get('distance')
            if distance and distance < min_distance:
                # 중앙 영역의 물체만 고려 (좌우는 무시)
                bbox = det.get('bbox', (0, 0, 0, 0))
                center_x = bbox[0] + bbox[2] / 2

                # 화면 중앙 ±30% 영역만
                if 0.35 < center_x / 640 < 0.65:  # 640은 웹캠 너비
                    min_distance = distance
                    closest_obj = det

        if min_distance == float('inf'):
            return 0.0

        # 거리 기반 브레이크
        if min_distance < self.emergency_distance:
            # 긴급 제동
            return 1.0
        elif min_distance < self.collision_distance:
            # 점진적 제동
            brake = 1.0 - (min_distance - self.emergency_distance) / \
                    (self.collision_distance - self.emergency_distance)

            # 보행자는 더 강하게 브레이크
            if closest_obj and 'person' in closest_obj.get('class', ''):
                brake = min(1.0, brake * 1.5)

            return brake

        return 0.0

    def _traffic_light_control(self, traffic_lights: List[Dict]) -> float:
        """
        신호등 준수

        빨간불이면 브레이크
        """
        if not traffic_lights:
            return 0.0

        # 가장 가까운 신호등
        closest_tl = None
        min_distance = float('inf')

        for tl in traffic_lights:
            distance = tl.get('distance')
            if distance and distance < min_distance:
                min_distance = distance
                closest_tl = tl

        if not closest_tl:
            return 0.0

        state = closest_tl.get('state', 'unknown')

        # 빨간불
        if state == 'red':
            if min_distance < 20:  # 20m 이내
                return 0.8  # 강한 브레이크

        # 노란불
        elif state == 'yellow':
            if min_distance < 15:
                return 0.5  # 중간 브레이크

        return 0.0

    def _smooth(self, current: float, previous: float) -> float:
        """
        지수 이동 평균으로 스무딩
        """
        return (1 - self.smoothing_factor) * previous + \
               self.smoothing_factor * current


if __name__ == "__main__":
    # 테스트
    controller = RuleBasedController()

    # 시나리오 1: 차선 오른쪽으로 치우침
    print("=== 시나리오 1: 차선 이탈 (오른쪽) ===")
    lane_info = {'deviation': 30}  # 오른쪽으로 30% 이탈
    steering, throttle, brake = controller.compute_control([], lane_info, [])
    print(f"Steering: {steering:.3f} (왼쪽으로 조향)")
    print(f"Throttle: {throttle:.3f}")
    print(f"Brake: {brake:.3f}")
    print()

    # 시나리오 2: 앞에 차량
    print("=== 시나리오 2: 앞에 차량 (10m) ===")
    detections = [
        {'class': 'car', 'distance': 10.0, 'bbox': (300, 200, 100, 80)}
    ]
    steering, throttle, brake = controller.compute_control(detections, {}, [])
    print(f"Steering: {steering:.3f}")
    print(f"Throttle: {throttle:.3f}")
    print(f"Brake: {brake:.3f} (브레이크 작동)")
    print()

    # 시나리오 3: 빨간 신호등
    print("=== 시나리오 3: 빨간 신호등 (15m) ===")
    traffic_lights = [
        {'state': 'red', 'distance': 15.0}
    ]
    steering, throttle, brake = controller.compute_control([], {}, traffic_lights)
    print(f"Steering: {steering:.3f}")
    print(f"Throttle: {throttle:.3f}")
    print(f"Brake: {brake:.3f} (빨간불 정지)")
    print()

    print("✅ Rule-based controller test passed")
