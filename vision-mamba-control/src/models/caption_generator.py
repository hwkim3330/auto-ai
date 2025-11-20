"""
Vision Mamba Caption Generator - 제어 신호를 자연어로 설명

Vision Mamba의 features와 제어 신호를 분석하여
현재 상황을 자연어로 설명
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


class CaptionGenerator:
    """
    Vision Mamba 출력을 자연어 설명으로 변환

    향후 확장:
    - Vision-Language Model 통합
    - Mamba-based Text Decoder
    - Multi-modal understanding
    """

    def __init__(self):
        """초기화"""
        self.history_length = 10
        self.steering_history = []
        self.throttle_history = []
        self.brake_history = []

    def generate_caption(
        self,
        steering: float,
        throttle: float,
        brake: float,
        features: torch.Tensor = None,
        camera_stats: Dict[str, float] = None
    ) -> Dict[str, str]:
        """
        제어 신호와 features를 분석하여 설명 생성

        Args:
            steering: -1 ~ 1
            throttle: 0 ~ 1
            brake: 0 ~ 1
            features: Vision Mamba features (optional)
            camera_stats: 카메라 통계 (optional)

        Returns:
            설명 딕셔너리
        """
        # 히스토리 업데이트
        self.steering_history.append(steering)
        self.throttle_history.append(throttle)
        self.brake_history.append(brake)

        if len(self.steering_history) > self.history_length:
            self.steering_history.pop(0)
            self.throttle_history.pop(0)
            self.brake_history.pop(0)

        # 1. 주행 상태 분석
        driving_state = self._analyze_driving_state(steering, throttle, brake)

        # 2. 조향 행동 분석
        steering_action = self._analyze_steering(steering)

        # 3. 속도 제어 분석
        speed_control = self._analyze_speed_control(throttle, brake)

        # 4. 트렌드 분석 (히스토리 기반)
        trend = self._analyze_trend()

        # 5. 카메라 환경 분석
        environment = self._analyze_environment(camera_stats) if camera_stats else ""

        # 6. Feature 기반 장면 이해 (향후 확장)
        scene_understanding = self._analyze_scene(features) if features is not None else ""

        # 종합 설명 생성
        main_caption = self._generate_main_caption(
            driving_state, steering_action, speed_control
        )

        detail_caption = self._generate_detail_caption(
            trend, environment, scene_understanding
        )

        return {
            "main": main_caption,
            "detail": detail_caption,
            "state": driving_state,
            "trend": trend,
            "environment": environment
        }

    def _analyze_driving_state(
        self,
        steering: float,
        throttle: float,
        brake: float
    ) -> str:
        """전체 주행 상태 판단"""

        # 긴급 제동
        if brake > 0.7:
            return "emergency_brake"

        # 정지
        if throttle < 0.1 and brake < 0.1:
            return "stopped"

        # 감속
        if brake > 0.3:
            return "slowing_down"

        # 커브
        if abs(steering) > 0.5:
            if throttle > 0.5:
                return "fast_turn"
            else:
                return "cautious_turn"

        # 직진 가속
        if abs(steering) < 0.2 and throttle > 0.6:
            return "accelerating"

        # 일반 주행
        if abs(steering) < 0.3 and throttle > 0.3:
            return "cruising"

        # 저속 주행
        return "slow_driving"

    def _analyze_steering(self, steering: float) -> str:
        """조향 분석"""
        abs_steer = abs(steering)

        if abs_steer < 0.1:
            return "직진 중"
        elif abs_steer < 0.3:
            direction = "왼쪽" if steering < 0 else "오른쪽"
            return f"약간 {direction}으로"
        elif abs_steer < 0.6:
            direction = "왼쪽" if steering < 0 else "오른쪽"
            return f"{direction}으로 선회"
        else:
            direction = "왼쪽" if steering < 0 else "오른쪽"
            return f"{direction}으로 급선회"

    def _analyze_speed_control(self, throttle: float, brake: float) -> str:
        """속도 제어 분석"""
        if brake > 0.7:
            return "긴급 제동"
        elif brake > 0.3:
            return "감속 중"
        elif throttle > 0.7:
            return "가속 중"
        elif throttle > 0.4:
            return "순항 중"
        elif throttle < 0.2:
            return "저속 주행"
        else:
            return "정속 주행"

    def _analyze_trend(self) -> str:
        """최근 행동 트렌드 분석"""
        if len(self.steering_history) < 3:
            return "데이터 수집 중"

        # 최근 평균
        recent_steer = np.mean(self.steering_history[-5:])
        recent_throttle = np.mean(self.throttle_history[-5:])

        # 변화율
        steer_change = abs(self.steering_history[-1] - self.steering_history[-3])

        if steer_change > 0.4:
            return "급격한 조향 변화"
        elif abs(recent_steer) > 0.4:
            direction = "왼쪽" if recent_steer < 0 else "오른쪽"
            return f"{direction} 커브 주행 중"
        elif recent_throttle > 0.6:
            return "고속 주행 중"
        elif recent_throttle < 0.3:
            return "저속 주행 중"
        else:
            return "안정적 주행"

    def _analyze_environment(self, camera_stats: Dict[str, float]) -> str:
        """카메라 환경 분석"""
        if not camera_stats:
            return ""

        brightness = camera_stats.get('brightness', 0.5)
        contrast = camera_stats.get('contrast', 0.5)
        saturation = camera_stats.get('saturation', 0.5)

        conditions = []

        # 밝기
        if brightness < 0.3:
            conditions.append("어두운 환경")
        elif brightness > 0.7:
            conditions.append("밝은 환경")

        # 대비
        if contrast < 0.3:
            conditions.append("저대비")
        elif contrast > 0.7:
            conditions.append("고대비")

        # 채도
        if saturation < 0.3:
            conditions.append("저채도")

        return ", ".join(conditions) if conditions else "정상 환경"

    def _analyze_scene(self, features: torch.Tensor) -> str:
        """
        Vision Mamba features로 장면 이해 (향후 확장)

        현재: feature 통계 기반 간단한 분석
        향후: Vision-Language Model 통합
        """
        if features is None:
            return ""

        # Feature statistics
        feat_mean = features.mean().item()
        feat_std = features.std().item()
        feat_max = features.max().item()

        # 간단한 휴리스틱
        if feat_std > 0.5:
            return "복잡한 장면"
        elif feat_std < 0.2:
            return "단순한 장면"
        else:
            return "일반적인 장면"

    def _generate_main_caption(
        self,
        driving_state: str,
        steering_action: str,
        speed_control: str
    ) -> str:
        """메인 설명 생성"""

        state_descriptions = {
            "emergency_brake": "⚠️ 긴급 제동!",
            "stopped": "🛑 정지 상태",
            "slowing_down": "🔽 감속 중",
            "fast_turn": "🏎️ 빠른 속도로 커브 진입",
            "cautious_turn": "🚗 신중하게 커브 주행",
            "accelerating": "⚡ 가속 중",
            "cruising": "✅ 안정적으로 순항 중",
            "slow_driving": "🐢 저속 주행 중"
        }

        main = state_descriptions.get(driving_state, "주행 중")

        # 세부 동작 추가
        if driving_state not in ["stopped", "emergency_brake"]:
            main += f" - {steering_action}, {speed_control}"

        return main

    def _generate_detail_caption(
        self,
        trend: str,
        environment: str,
        scene: str
    ) -> str:
        """상세 설명 생성"""
        details = []

        if trend:
            details.append(f"📊 {trend}")

        if environment:
            details.append(f"🌤️ {environment}")

        if scene:
            details.append(f"👁️ {scene}")

        return " | ".join(details) if details else "정상 작동 중"

    def reset(self):
        """히스토리 초기화"""
        self.steering_history.clear()
        self.throttle_history.clear()
        self.brake_history.clear()


def generate_simple_caption(
    steering: float,
    throttle: float,
    brake: float
) -> str:
    """간단한 1줄 설명 (standalone 함수)"""

    # 우선순위: 브레이크 > 조향 > 스로틀
    if brake > 0.5:
        return f"⚠️ 제동 중 (브레이크: {brake:.0%})"

    steer_text = ""
    if abs(steering) > 0.3:
        direction = "왼쪽" if steering < 0 else "오른쪽"
        steer_text = f"{direction} 선회"
    else:
        steer_text = "직진"

    speed_text = ""
    if throttle > 0.6:
        speed_text = "가속"
    elif throttle > 0.3:
        speed_text = "순항"
    else:
        speed_text = "저속"

    return f"🚗 {steer_text} - {speed_text} (스로틀: {throttle:.0%})"


if __name__ == "__main__":
    # 테스트
    generator = CaptionGenerator()

    # 테스트 시나리오
    scenarios = [
        (0.0, 0.7, 0.0, "직진 가속"),
        (-0.6, 0.5, 0.0, "왼쪽 커브"),
        (0.8, 0.3, 0.0, "오른쪽 급커브"),
        (0.0, 0.0, 0.8, "긴급 제동"),
        (0.1, 0.4, 0.0, "안정적 순항"),
    ]

    print("Vision Mamba Caption Generator - 테스트")
    print("=" * 60)

    for steering, throttle, brake, expected in scenarios:
        caption = generator.generate_caption(
            steering, throttle, brake,
            camera_stats={'brightness': 0.5, 'contrast': 0.5, 'saturation': 0.5}
        )

        print(f"\n예상: {expected}")
        print(f"메인: {caption['main']}")
        print(f"상세: {caption['detail']}")
        print("-" * 60)
