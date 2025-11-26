#!/usr/bin/env python3
"""
Simple Vision Pro API Client Example

이 예제는 Vision Pro API를 사용하는 가장 간단한 방법을 보여줍니다.
"""

import requests
import time
import json
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8080"
POLL_INTERVAL = 0.1  # 100ms = 10 Hz


def fetch_vision_data():
    """비전 데이터 조회"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/monitor/data", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data: {e}")
        return None


def print_detection_info(data):
    """검출 정보 출력"""
    if not data:
        return

    # 성능 정보
    perf = data.get('performance', {})
    print(f"\n📊 Performance: FPS={perf.get('fps', 0):.1f}")

    # 분석 정보
    analytics = data.get('analytics', {})
    total = analytics.get('total_objects', 0)
    print(f"🎯 Total Objects: {total}")

    if total > 0:
        by_class = analytics.get('by_class', {})
        print(f"   By Class: {dict(by_class)}")

        # 검출 상세
        detections = data.get('detections', [])
        print(f"\n📍 Detections:")
        for i, det in enumerate(detections[:5], 1):  # 최대 5개만 표시
            class_name = det.get('class_name', 'Unknown')
            confidence = det.get('confidence', 0)
            depth = det.get('depth', 0)
            print(f"   {i}. {class_name}: {confidence:.2f} @ {depth:.1f}m")

        # 경고
        alerts = analytics.get('alerts', [])
        if alerts:
            print(f"\n⚠️  Alerts:")
            for alert in alerts:
                print(f"   - [{alert['severity'].upper()}] {alert['message']}")


def main():
    """메인 함수"""
    print("=" * 60)
    print("Vision Pro API Client - Simple Example")
    print("=" * 60)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Poll Interval: {POLL_INTERVAL}s ({1/POLL_INTERVAL:.0f} Hz)")
    print("=" * 60)
    print("\nPress Ctrl+C to stop\n")

    try:
        while True:
            # 현재 시간
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"\n[{timestamp}]", end="")

            # 데이터 조회
            data = fetch_vision_data()

            # 정보 출력
            print_detection_info(data)

            # 대기
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n👋 Client stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
