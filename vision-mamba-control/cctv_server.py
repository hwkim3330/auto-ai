#!/usr/bin/env python3
"""
CCTV Monitoring Server with Depth Anything V3

Features:
- Person detection and tracking
- Depth estimation (Depth Anything V3)
- Distance measurement
- Loitering detection
- Zone intrusion alerts
- Real-time analytics
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import time
import threading
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from capture.webcam import WebcamCapture
from vision.tesla_detector import TeslaDetector
from vision.depth_estimator import DepthAnythingV3, CCTVMonitor

app = Flask(__name__,
            template_folder='web/templates',
            static_folder='web/static')

# Global state
webcam = None
tesla_detector = None
depth_estimator = None
cctv_monitor = None
is_running = False
current_frame = None
current_data = {
    'fps': 0.0,
    'total_persons': 0,
    'loitering_count': 0,
    'close_persons': 0,
    'avg_distance': 0.0,
    'avg_height': 0.0,
    'alerts': []
}


def init_system():
    """시스템 초기화"""
    global tesla_detector, depth_estimator, cctv_monitor

    try:
        # Initialize YOLO detector
        tesla_detector = TeslaDetector(model_size='n', confidence_threshold=0.5)
        print("✅ YOLOv8 Person Detector initialized")

        # Initialize Depth Anything V3
        depth_estimator = DepthAnythingV3(model_size='small', device='cpu')
        print("✅ Depth Anything V3 initialized")

        # Initialize CCTV Monitor
        cctv_monitor = CCTVMonitor(depth_estimator)
        print("✅ CCTV Monitor initialized")

    except Exception as e:
        print(f"⚠️ System init failed: {e}")


def update_loop():
    """메인 업데이트 루프"""
    global webcam, current_frame, current_data, is_running

    last_time = time.time()

    while is_running:
        try:
            if webcam is None:
                time.sleep(0.1)
                continue

            # 프레임 읽기
            frame = webcam.read_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # FPS 계산
            current_time = time.time()
            fps = 1.0 / (current_time - last_time) if current_time > last_time else 30.0
            last_time = current_time

            # YOLO 감지
            detections = []
            if tesla_detector:
                detections = tesla_detector.detect_objects(frame)

            # CCTV 모니터링 처리
            analytics = {'total_persons': 0, 'loitering': [], 'close_persons': []}

            if cctv_monitor:
                frame_processed, analytics = cctv_monitor.process_frame(
                    frame, detections
                )
            else:
                frame_processed = frame

            # 데이터 업데이트
            persons = [d for d in detections if 'person' in d.get('class', '')]
            distances = [p.get('distance_depth', 0) for p in persons if p.get('distance_depth', 0) > 0]
            heights = [p.get('height', 0) for p in persons if p.get('height', 0) > 0]

            avg_distance = np.mean(distances) if distances else 0.0
            avg_height = np.mean(heights) if heights else 0.0

            # 알림 생성
            alerts = []
            for loiter in analytics.get('loitering', []):
                alerts.append(f"⚠️ Loitering detected: ID {loiter['id']} ({loiter['duration']:.0f}s)")

            for close in analytics.get('close_persons', []):
                alerts.append(f"🚨 Person too close: {close['distance']:.1f}m")

            current_data.update({
                'fps': float(fps),
                'total_persons': analytics.get('total_persons', 0),
                'loitering_count': len(analytics.get('loitering', [])),
                'close_persons': len(analytics.get('close_persons', [])),
                'avg_distance': float(avg_distance),
                'avg_height': float(avg_height),
                'alerts': alerts[:5]  # Last 5 alerts
            })

            # 프레임 저장
            current_frame = frame_processed

        except Exception as e:
            print(f"Error in update loop: {e}")
            time.sleep(0.1)


def generate_frames():
    """비디오 스트리밍을 위한 프레임 생성"""
    global current_frame

    while True:
        if current_frame is not None:
            # JPEG 인코딩
            ret, buffer = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()

            # MJPEG 스트림
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)


@app.route('/')
def index():
    """메인 페이지"""
    return render_template('cctv.html')


@app.route('/video_feed')
def video_feed():
    """비디오 스트림"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/data')
def get_data():
    """현재 데이터 API"""
    return jsonify(current_data)


@app.route('/api/start', methods=['POST'])
def start_capture():
    """웹캠 시작"""
    global webcam, is_running

    if is_running:
        return jsonify({'status': 'already_running'})

    webcam = WebcamCapture(camera_id=0, width=640, height=480, fps=30)

    if not webcam.start():
        return jsonify({'status': 'error', 'message': 'Failed to start webcam'})

    is_running = True

    # 백그라운드 스레드 시작
    thread = threading.Thread(target=update_loop, daemon=True)
    thread.start()

    return jsonify({'status': 'started'})


@app.route('/api/stop', methods=['POST'])
def stop_capture():
    """웹캠 중지"""
    global webcam, is_running

    is_running = False

    if webcam:
        webcam.stop()
        webcam = None

    return jsonify({'status': 'stopped'})


if __name__ == '__main__':
    print("=" * 60)
    print("CCTV Monitoring System - Depth Anything V3")
    print("=" * 60)
    print()
    print("🚀 Initializing system...")

    init_system()

    print()
    print("🌐 Starting CCTV server...")
    print("📍 Open your browser: http://localhost:8081")
    print()
    print("Features:")
    print("  - Person detection & tracking")
    print("  - Depth estimation (Depth Anything V3)")
    print("  - Distance measurement")
    print("  - Loitering detection")
    print("  - Real-time alerts")
    print()
    print("=" * 60)
    print()

    # Flask 서버 실행
    app.run(host='0.0.0.0', port=8081, debug=False, threaded=True)
