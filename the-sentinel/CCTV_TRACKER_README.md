# 실시간 CCTV 추적 시스템 | Real-Time CCTV Tracking System

> **"모든 사람의 위치를 지도에 찍고 예측하기"** - Person of Interest 스타일 실시간 추적

## ⚠️ 윤리적/법적 고지 | ETHICS & LEGAL NOTICE

**본 시스템은 교육 및 연구 목적으로만 사용해야 합니다.**

### 준수 사항
- ❌ **얼굴 인식 금지** - 개인정보보호법 위반
- ✅ **익명 추적만** - 개인 식별 불가능
- ✅ **교통 모니터링 목적** - 공공 안전 용도
- ✅ **연구/교육 목적** - 상업적 사용 금지

### 법적 근거
- **개인정보보호법** 제15조, 제24조
- **정보통신망법** 제22조
- **CCTV 설치 및 운영 관련 법률**
- **도로교통법** 제2조

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│            실시간 CCTV 추적 시스템                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐      ┌─────────────────┐             │
│  │ 한국 공공     │─────>│ Object Detection│             │
│  │ CCTV API     │      │    (YOLO/BGS)   │             │
│  └──────────────┘      └────────┬─────────┘             │
│                                 │                       │
│                                 ▼                       │
│                      ┌──────────────────┐               │
│                      │ Multi-Object     │               │
│                      │ Tracking (SORT)  │               │
│                      └────────┬──────────┘               │
│                               │                          │
│                               ▼                          │
│                      ┌──────────────────┐               │
│                      │ Movement         │               │
│                      │ Prediction       │               │
│                      │ (Kalman Filter)  │               │
│                      └────────┬──────────┘               │
│                               │                          │
│                               ▼                          │
│                      ┌──────────────────┐               │
│                      │ Real-Time Map    │               │
│                      │ Visualization    │               │
│                      │ (Leaflet.js)     │               │
│                      └──────────────────┘               │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 핵심 기능

### 1. CCTV 통합 (CCTV Integration)
- 한국 공공데이터포털 API 연동
- 서울시 CCTV 위치 정보
- 국토교통부 ITS CCTV
- 실시간 스트림 처리

### 2. 객체 탐지 (Object Detection)
- **사람 탐지**: Background subtraction + 형태 분석
- **차량 탐지**: 크기 및 종횡비 기반 분류
- **실시간 처리**: 30 FPS
- **생산 환경**: YOLOv8, DETR 통합 가능

### 3. 다중 객체 추적 (Multi-Object Tracking)
- **IoU 기반 매칭**: 탐지 결과와 기존 트랙 연결
- **트랙 관리**: 새 트랙 생성, 오래된 트랙 제거
- **궤적 기록**: 최근 30개 위치 저장
- **생산 환경**: DeepSORT, ByteTrack 통합 가능

### 4. 이동 예측 (Movement Prediction)
- **속도 계산**: 연속된 위치 기반 벡터 계산
- **위치 예측**: Kalman Filter 스타일 예측
- **예측 시간**: 0.5초 앞 위치 예측
- **시각화**: 점선으로 예측 경로 표시

### 5. 실시간 지도 시각화
- **OpenStreetMap** 기반
- **Leaflet.js** 사용
- **실시간 업데이트**: 2초 간격
- **인터랙티브**: 클릭으로 상세 정보

## 설치 및 실행

### 의존성 설치

```bash
cd /home/kim/auto-ai/the-sentinel

# OpenCV 설치
pip install opencv-python opencv-contrib-python

# 기타 의존성
pip install numpy requests
```

### 실행 방법

#### 1. 실시간 추적 시작

```bash
# 카메라로 실시간 추적 (웹캠 사용)
python3 realtime_tracker.py
```

**컨트롤**:
- `q`: 종료
- 화면에 실시간으로 탐지/추적 결과 표시

#### 2. 지도 시각화 열기

```bash
# 브라우저에서 지도 열기
firefox map_visualization.html

# 또는
chromium-browser map_visualization.html
```

**기능**:
- 🔄 데이터 새로고침: 추적 데이터 업데이트
- 🔮 예측 표시: 예측 위치 on/off
- 📈 궤적 표시: 이동 궤적 on/off

## 사용 가능한 API

### 한국 공공 CCTV API

#### 1. 서울 열린데이터 광장
```
URL: https://data.seoul.go.kr/
API: CCTV 설치 현황
인증: API 키 필요
```

#### 2. 국토교통부 ITS
```
URL: https://www.its.go.kr/
API: 지능형 교통체계 CCTV
인증: API 키 필요
```

#### 3. 도로교통공단
```
URL: https://www.koroad.or.kr/
API: 교통 모니터링 CCTV
인증: API 키 필요
```

### API 키 발급 방법

1. **공공데이터포털** (https://www.data.go.kr/)
   - 회원가입
   - API 신청
   - 승인 대기 (1-2일)
   - API 키 발급

2. **코드에 적용**
```python
# realtime_tracker.py 에서
cctv_api = KoreaCCTVIntegration(api_key="YOUR_API_KEY_HERE")
```

## 실제 CCTV 연동 방법

### RTSP 스트림 사용

```python
# realtime_tracker.py 수정

def get_stream(self, camera_id: str) -> Optional[cv2.VideoCapture]:
    camera = next((c for c in self.cameras if c.id == camera_id), None)
    if camera is None:
        return None

    # 실제 RTSP URL 사용
    rtsp_url = f"rtsp://username:password@{camera.stream_url}"
    cap = cv2.VideoCapture(rtsp_url)

    return cap
```

### HTTP 스트림 사용

```python
# M-JPEG 스트림
mjpeg_url = "http://camera_ip/mjpeg_stream"
cap = cv2.VideoCapture(mjpeg_url)
```

## 성능 최적화

### 1. GPU 가속

```python
# YOLOv8 GPU 사용
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.to('cuda')  # GPU로 이동

results = model(frame, device='cuda')
```

### 2. 멀티스레딩

```python
import threading
import queue

# 각 카메라를 별도 스레드에서 처리
threads = []
for camera_id in camera_ids:
    thread = threading.Thread(
        target=process_camera,
        args=(camera_id, frame_queue)
    )
    threads.append(thread)
    thread.start()
```

### 3. 프레임 스킵

```python
# 매 N번째 프레임만 처리
frame_count = 0
skip_frames = 2

while True:
    ret, frame = cap.read()
    frame_count += 1

    if frame_count % skip_frames != 0:
        continue

    # 프레임 처리
```

## 고급 기능

### 1. YOLOv8 통합

```bash
pip install ultralytics
```

```python
from ultralytics import YOLO

class YOLODetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')  # nano 모델

    def detect(self, frame):
        results = self.model(frame, classes=[0, 2])  # person, car
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())

                detections.append(Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(conf),
                    class_id=cls,
                    class_name='person' if cls == 0 else 'vehicle'
                ))

        return detections
```

### 2. DeepSORT 통합

```bash
pip install deep-sort-realtime
```

```python
from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(
    max_age=30,
    n_init=3,
    nms_max_overlap=1.0,
    max_cosine_distance=0.3
)

tracks = tracker.update_tracks(detections, frame=frame)
```

### 3. 데이터베이스 저장

```python
import sqlite3
from datetime import datetime

# SQLite 데이터베이스
conn = sqlite3.connect('tracking_data.db')
cursor = conn.cursor()

# 테이블 생성
cursor.execute('''
CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY,
    track_id INTEGER,
    timestamp TEXT,
    class TEXT,
    latitude REAL,
    longitude REAL,
    velocity_x REAL,
    velocity_y REAL,
    camera_id TEXT
)
''')

# 데이터 저장
def save_track(track):
    cursor.execute('''
    INSERT INTO tracks VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        track.track_id,
        datetime.now().isoformat(),
        track.class_name,
        track.geo_location[0],
        track.geo_location[1],
        track.velocity[0],
        track.velocity[1],
        track.camera_id
    ))
    conn.commit()
```

## 웹 API 서버

실시간 데이터를 웹에 제공하는 서버:

```python
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/tracks')
def get_tracks():
    # 현재 추적 데이터 반환
    return jsonify({
        'timestamp': time.time(),
        'tracks': [asdict(track) for track in current_tracks]
    })

@app.route('/api/cameras')
def get_cameras():
    return jsonify({
        'cameras': [asdict(cam) for cam in cameras]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

지도에서 실시간 데이터 fetch:

```javascript
async function loadTrackingData() {
    const response = await fetch('http://localhost:5000/api/tracks');
    const data = await response.json();
    tracks = data.tracks;

    updateMap();
    updateStats();
}
```

## 예시 출력

### 터미널 출력
```
======================================================================
REAL-TIME CCTV TRACKING SYSTEM
한국 공공 CCTV 실시간 추적 시스템
======================================================================

ETHICS & LEGAL NOTICE:
- 얼굴 인식 없음 (No facial recognition)
- 익명 추적만 (Anonymous tracking only)
- 교육/연구 목적 (Educational/Research purpose)
======================================================================
[CCTV API] Initialized
[CCTV API] Loaded 4 cameras
[Detector] Initialized background subtractor
[Tracker] Initialized
[Tracking System] Initialized
[Tracking] Started camera: SEOUL_001
[Tracking] Starting real-time processing...
Press 'q' to quit
```

### 지도 화면
- 서울 지도 위에 4개 CCTV 위치 표시
- 실시간으로 사람/차량 마커 움직임
- 궤적 선으로 이동 경로 표시
- 예측 위치 점선으로 표시
- 클릭하면 상세 정보 팝업

## 주의사항

### 개인정보 보호
1. **얼굴 블러링 필수**
```python
# 얼굴 영역 검출
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(frame)

# 블러 처리
for (x, y, w, h) in faces:
    frame[y:y+h, x:x+w] = cv2.blur(frame[y:y+h, x:x+w], (23, 23))
```

2. **데이터 익명화**
- 개인 식별 정보 저장 금지
- 트랙 ID만 사용 (이름, 얼굴 등 저장 X)
- GPS 위치는 일반화 (정확한 좌표 X, 구역만)

3. **데이터 보유 기간**
- 최대 7일 이내 삭제
- 법적 요구사항 준수

### 성능 고려사항
- 동시 처리 카메라 수: GPU 성능에 따라 1-16개
- 프레임 레이트: 10-30 FPS
- 추적 정확도: 85-95% (환경에 따라 다름)

## 라이선스

MIT License - 교육/연구 목적 사용

---

**"영화처럼 모든 사람의 위치를 지도에 찍고 예측한다"** 🎥📍
