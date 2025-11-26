#!/usr/bin/env python3
"""
Vision Pro Data Logger Example

실시간 검출 데이터를 CSV 파일로 저장하는 예제
"""

import requests
import time
import csv
from datetime import datetime
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8080"
POLL_INTERVAL = 1.0  # 1 second
OUTPUT_DIR = Path("logs")
OUTPUT_FILE = OUTPUT_DIR / f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


def ensure_output_dir():
    """출력 디렉토리 생성"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR.absolute()}")


def fetch_vision_data():
    """비전 데이터 조회"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/monitor/data", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        return None


def log_to_csv(writer, data):
    """CSV에 데이터 기록"""
    if not data:
        return

    timestamp = datetime.now().isoformat()
    detections = data.get('detections', [])
    analytics = data.get('analytics', {})
    performance = data.get('performance', {})

    # 각 검출에 대해 행 기록
    if detections:
        for det in detections:
            row = [
                timestamp,
                det.get('class_id', ''),
                det.get('class_name', ''),
                f"{det.get('confidence', 0):.4f}",
                f"{det.get('depth', 0):.2f}",
                det.get('track_id', ''),
                analytics.get('total_objects', 0),
                f"{performance.get('fps', 0):.2f}"
            ]
            writer.writerow(row)
    else:
        # 검출 없음
        row = [timestamp, '', 'None', '', '', '', 0, f"{performance.get('fps', 0):.2f}"]
        writer.writerow(row)


def main():
    """메인 함수"""
    print("=" * 70)
    print("Vision Pro Data Logger")
    print("=" * 70)
    print(f"API: {API_BASE_URL}")
    print(f"Interval: {POLL_INTERVAL}s")

    ensure_output_dir()
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 70)
    print("\nPress Ctrl+C to stop\n")

    # CSV 파일 열기
    with open(OUTPUT_FILE, 'w', newline='') as csvfile:
        fieldnames = [
            'timestamp',
            'class_id',
            'class_name',
            'confidence',
            'depth_m',
            'track_id',
            'total_objects',
            'fps'
        ]
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)  # 헤더

        row_count = 0

        try:
            while True:
                # 데이터 조회
                data = fetch_vision_data()

                # CSV 기록
                log_to_csv(writer, data)
                csvfile.flush()  # 즉시 쓰기

                row_count += 1
                if row_count % 10 == 0:
                    print(f"✅ Logged {row_count} rows")

                # 대기
                time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print(f"\n\n📊 Total rows logged: {row_count}")
            print(f"💾 Saved to: {OUTPUT_FILE}")
            print("👋 Logger stopped")
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
