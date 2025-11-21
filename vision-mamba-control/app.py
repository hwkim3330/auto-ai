#!/usr/bin/env python3
"""
Vision Mamba Control - Unified Server

통합 서버:
- /driving - 자율주행 (Vision Mamba + RL)
- /cctv - CCTV 모니터링 (Depth Anything V3 + Person Detection)
"""

from flask import Flask, render_template, Response, jsonify, request
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
from vision.liquidai_describer import LiquidAIDescriber, get_describer

app = Flask(__name__,
            template_folder='web/templates',
            static_folder='web/static')

# ============================================================================
# CCTV Mode
# ============================================================================
cctv_webcam = None
tesla_detector = None
depth_estimator = None
cctv_monitor = None
cctv_running = False
cctv_frame = None
cctv_data = {
    'fps': 0.0,
    'total_persons': 0,
    'loitering_count': 0,
    'close_persons': 0,
    'avg_distance': 0.0,
    'avg_height': 0.0,
    'alerts': [],
    'ai_description': ''
}

# LiquidAI
liquidai_describer = None
liquidai_enabled = False
current_detections = []
current_analytics = {}


def init_cctv_system():
    """Initialize CCTV system"""
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
        print(f"⚠️ CCTV init failed: {e}")


def cctv_update_loop():
    """CCTV update loop"""
    global cctv_webcam, cctv_frame, cctv_data, cctv_running

    last_time = time.time()

    while cctv_running:
        try:
            if cctv_webcam is None:
                time.sleep(0.1)
                continue

            frame = cctv_webcam.read_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # FPS
            current_time = time.time()
            fps = 1.0 / (current_time - last_time) if current_time > last_time else 30.0
            last_time = current_time

            # YOLO detection
            detections = []
            if tesla_detector:
                detections = tesla_detector.detect_objects(frame)

            # CCTV monitoring
            analytics = {'total_persons': 0, 'loitering': [], 'close_persons': []}

            if cctv_monitor:
                frame_processed, analytics = cctv_monitor.process_frame(frame, detections)
            else:
                frame_processed = frame

            # Update data
            persons = [d for d in detections if 'person' in d.get('class', '')]
            distances = [p.get('distance_depth', 0) for p in persons if p.get('distance_depth', 0) > 0]
            heights = [p.get('height', 0) for p in persons if p.get('height', 0) > 0]

            avg_distance = np.mean(distances) if distances else 0.0
            avg_height = np.mean(heights) if heights else 0.0

            # Alerts
            alerts = []
            for loiter in analytics.get('loitering', []):
                alerts.append(f"⚠️ Loitering: ID {loiter['id']} ({loiter['duration']:.0f}s)")

            for close in analytics.get('close_persons', []):
                alerts.append(f"🚨 Too close: {close['distance']:.1f}m")

            # Update current detections for AI description
            global current_detections, current_analytics
            current_detections = detections
            current_analytics = analytics

            # Generate AI description if enabled
            ai_desc = ''
            if liquidai_enabled and liquidai_describer:
                try:
                    ai_desc = liquidai_describer.describe_scene(detections, analytics)
                except Exception as e:
                    ai_desc = f"AI unavailable: {str(e)[:50]}"

            cctv_data.update({
                'fps': float(fps),
                'total_persons': analytics.get('total_persons', 0),
                'loitering_count': len(analytics.get('loitering', [])),
                'close_persons': len(analytics.get('close_persons', [])),
                'avg_distance': float(avg_distance),
                'avg_height': float(avg_height),
                'alerts': alerts[:5],
                'ai_description': ai_desc
            })

            cctv_frame = frame_processed

        except Exception as e:
            print(f"CCTV error: {e}")
            time.sleep(0.1)


def generate_cctv_frames():
    """Generate CCTV video stream"""
    global cctv_frame

    while True:
        if cctv_frame is not None:
            ret, buffer = cv2.imencode('.jpg', cctv_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)


# ============================================================================
# Routes
# ============================================================================

@app.route('/')
def index():
    """메인 페이지 - 모드 선택"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vision Mamba Control</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
                background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
                color: #f0f0f0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
            .container {
                text-align: center;
            }
            h1 {
                font-size: 3rem;
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 2rem;
            }
            .buttons {
                display: flex;
                gap: 2rem;
                justify-content: center;
            }
            .btn {
                padding: 1.5rem 3rem;
                font-size: 1.2rem;
                border: none;
                border-radius: 12px;
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
                transition: transform 0.3s;
            }
            .btn:hover {
                transform: translateY(-5px);
            }
            .btn-cctv {
                background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                color: white;
            }
            .btn-drive {
                background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
                color: white;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Vision Mamba Control</h1>
            <div class="buttons">
                <a href="/cctv" class="btn btn-cctv">📹 CCTV Monitoring</a>
            </div>
        </div>
    </body>
    </html>
    """


@app.route('/cctv')
def cctv_page():
    """CCTV 모니터링 페이지"""
    return render_template('cctv.html')


@app.route('/cctv/video_feed')
def cctv_video_feed():
    """CCTV 비디오 스트림"""
    return Response(generate_cctv_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/cctv/data')
def cctv_get_data():
    """CCTV 데이터 API"""
    return jsonify(cctv_data)


@app.route('/api/cctv/start', methods=['POST'])
def cctv_start():
    """CCTV 시작"""
    global cctv_webcam, cctv_running

    if cctv_running:
        return jsonify({'status': 'already_running'})

    cctv_webcam = WebcamCapture(camera_id=0, width=640, height=480, fps=30)

    if not cctv_webcam.start():
        return jsonify({'status': 'error', 'message': 'Failed to start webcam'})

    cctv_running = True

    # Start background thread
    thread = threading.Thread(target=cctv_update_loop, daemon=True)
    thread.start()

    return jsonify({'status': 'started'})


@app.route('/api/cctv/stop', methods=['POST'])
def cctv_stop():
    """CCTV 중지"""
    global cctv_webcam, cctv_running

    cctv_running = False

    if cctv_webcam:
        cctv_webcam.stop()
        cctv_webcam = None

    return jsonify({'status': 'stopped'})


# ============================================================================
# LiquidAI API Endpoints
# ============================================================================

@app.route('/api/liquidai/enable', methods=['POST'])
def liquidai_enable():
    """Enable LiquidAI scene description"""
    global liquidai_describer, liquidai_enabled

    try:
        liquidai_describer = get_describer()
        liquidai_enabled = True

        # Start loading model in background
        def load_model_bg():
            liquidai_describer.load_model()

        thread = threading.Thread(target=load_model_bg, daemon=True)
        thread.start()

        return jsonify({
            'status': 'enabled',
            'message': 'LiquidAI enabled. Model loading in background...'
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/liquidai/disable', methods=['POST'])
def liquidai_disable():
    """Disable LiquidAI scene description"""
    global liquidai_enabled

    liquidai_enabled = False

    return jsonify({'status': 'disabled'})


@app.route('/api/liquidai/status')
def liquidai_status():
    """Get LiquidAI status"""
    status = {
        'enabled': liquidai_enabled,
        'loaded': False,
        'loading': False,
        'model_id': 'LiquidAI/LFM2-1.2B',
        'description': cctv_data.get('ai_description', '')
    }

    if liquidai_describer:
        describer_status = liquidai_describer.get_status()
        status['loaded'] = describer_status.get('loaded', False)
        status['loading'] = describer_status.get('loading', False)

    return jsonify(status)


@app.route('/api/liquidai/describe', methods=['POST'])
def liquidai_describe_now():
    """Generate description immediately"""
    global liquidai_describer

    if not liquidai_enabled:
        return jsonify({
            'status': 'error',
            'message': 'LiquidAI not enabled'
        })

    if not liquidai_describer:
        liquidai_describer = get_describer()

    try:
        # Force immediate description
        liquidai_describer.last_description_time = 0
        description = liquidai_describer.describe_scene(
            current_detections,
            current_analytics
        )

        return jsonify({
            'status': 'success',
            'description': description
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })


if __name__ == '__main__':
    print("=" * 60)
    print("Vision Mamba Control - Unified Server")
    print("=" * 60)
    print()
    print("🚀 Initializing systems...")
    print()

    # Initialize CCTV
    init_cctv_system()

    print()
    print("🌐 Starting unified server...")
    print("📍 Open your browser:")
    print("   - Main: http://localhost:8080")
    print("   - CCTV: http://localhost:8080/cctv")
    print()
    print("=" * 60)
    print()

    # Run Flask server
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
