#!/usr/bin/env python3
"""
Vision Mamba Web Server - Tesla-style Web Interface

Flask 기반 웹 서버로 터미네이터/테슬라 스타일 UI 제공
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
from models.caption_generator import CaptionGenerator

# Try to load Vision Mamba model
try:
    import torch
    from models.control_model import create_control_model_tiny
    from vision.terminator_vision import TerminatorVision

    MODEL_AVAILABLE = True
    print("Vision Mamba model loaded")
except Exception as e:
    MODEL_AVAILABLE = False
    print(f"Model not available: {e}")
    print("Running in demo mode")

app = Flask(__name__,
            template_folder='web/templates',
            static_folder='web/static')

# Global state
webcam = None
model = None
caption_gen = CaptionGenerator()
vision_system = None
is_running = False
current_frame = None
current_data = {
    'steering': 0.0,
    'throttle': 0.0,
    'brake': 0.0,
    'fps': 0.0,
    'caption_main': '대기 중...',
    'caption_detail': '웹캠을 시작하세요',
    'objects': 0,
    'lanes': 0
}

def init_system():
    """시스템 초기화"""
    global model, vision_system

    if MODEL_AVAILABLE:
        try:
            model = create_control_model_tiny(use_film=True)
            model.eval()
            print("✅ Model initialized")
        except:
            print("⚠️ Model init failed - using demo mode")

    try:
        from vision.terminator_vision import TerminatorVision
        vision_system = TerminatorVision(use_yolo=False)
        print("✅ Vision system initialized")
    except Exception as e:
        print(f"⚠️ Vision system init failed: {e}")

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

            # 카메라 통계
            brightness, contrast, saturation = webcam.compute_camera_stats(frame)

            # 제어 신호 예측
            if MODEL_AVAILABLE and model is not None:
                # AI mode
                steering, throttle, brake = model.predict_from_webcam(
                    frame, brightness, contrast, saturation
                )
            else:
                # Demo mode
                t = time.time()
                steering = np.sin(t * 0.5) * 0.7
                throttle = (np.sin(t * 0.3) + 1) / 2
                brake = max(0, -steering * 0.3)

            # 설명 생성
            caption = caption_gen.generate_caption(
                steering, throttle, brake,
                camera_stats={
                    'brightness': brightness,
                    'contrast': contrast,
                    'saturation': saturation
                }
            )

            # Vision system 처리
            if vision_system:
                frame_processed = vision_system.process_frame(
                    frame, steering, throttle, brake, fps
                )
            else:
                frame_processed = webcam.draw_overlay(
                    frame, steering, throttle, brake, fps
                )

            # 데이터 업데이트
            current_data.update({
                'steering': float(steering),
                'throttle': float(throttle),
                'brake': float(brake),
                'fps': float(fps),
                'caption_main': caption['main'],
                'caption_detail': caption['detail'],
                'brightness': float(brightness),
                'contrast': float(contrast),
                'saturation': float(saturation),
                'objects': 0,  # vision_system에서 가져올 수 있음
                'lanes': 0
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
    return render_template('index.html')

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
    print("Vision Mamba Web Server - Tesla Style")
    print("=" * 60)
    print()
    print("🚀 Initializing system...")

    init_system()

    print()
    print("🌐 Starting web server...")
    print("📍 Open your browser: http://localhost:5000")
    print()
    print("Controls:")
    print("  - Start: 웹캠 시작")
    print("  - Stop: 웹캠 중지")
    print("  - Ctrl+C: 서버 종료")
    print()
    print("=" * 60)
    print()

    # Flask 서버 실행
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
