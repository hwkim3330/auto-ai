# API Reference - Vision Pro v1.3

REST API 완전한 레퍼런스

---

## 📡 Base URL

```
Development: http://localhost:8080
Production:  https://your-domain.com
```

---

## 🔑 Authentication

**현재 버전**: 인증 없음 (로컬 개발용)

**향후 버전** (v2.0+):
```http
Authorization: Bearer <JWT_TOKEN>
```

---

## 📋 Endpoints

### 1. 홈페이지

```http
GET /
```

**설명**: 메인 랜딩 페이지

**응답**:
```html
<!DOCTYPE html>
<html>
  ...
</html>
```

---

### 2. 모니터링 UI

```http
GET /monitor
```

**설명**: 실시간 AI 비전 모니터링 대시보드

**응답**:
```html
<!DOCTYPE html>
<html>
  <!-- monitor.html 템플릿 -->
</html>
```

**Features**:
- Real-time video feed
- Bird's Eye View
- Detection log
- Performance charts
- Settings panel
- Screenshot/Recording/Notifications

---

### 3. 실시간 데이터 API

```http
GET /api/monitor/data
```

**설명**: 현재 비전 시스템 데이터를 JSON으로 반환

**Query Parameters**: 없음

**응답 예시**:
```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.87,
      "bbox": [120, 150, 320, 480],
      "track_id": 1,
      "depth": 2.5,
      "position_3d": {
        "x": 0.3,
        "y": -0.1,
        "z": 2.5
      },
      "height": 1.75
    },
    {
      "class_id": 2,
      "class_name": "car",
      "confidence": 0.92,
      "bbox": [400, 200, 600, 400],
      "track_id": 2,
      "depth": 5.8,
      "position_3d": {
        "x": 1.2,
        "y": 0.0,
        "z": 5.8
      },
      "height": 1.5
    }
  ],
  "analytics": {
    "total_objects": 2,
    "by_class": {
      "person": 1,
      "car": 1
    },
    "closest_distance": 2.5,
    "alerts": [
      {
        "type": "proximity",
        "message": "Person detected within 3m",
        "severity": "warning"
      }
    ]
  },
  "performance": {
    "fps": 28.5,
    "detection_fps": 5.7,
    "depth_fps": 0.57
  },
  "timestamp": "2025-11-20T16:45:23.123456"
}
```

**응답 필드 설명**:

| Field | Type | Description |
|-------|------|-------------|
| `detections` | Array | 검출된 객체 목록 |
| `detections[].class_id` | Integer | COCO 클래스 ID (0-79) |
| `detections[].class_name` | String | 클래스 이름 (person, car, ...) |
| `detections[].confidence` | Float | 신뢰도 (0.0-1.0) |
| `detections[].bbox` | Array[4] | [x1, y1, x2, y2] 픽셀 좌표 |
| `detections[].track_id` | Integer | 추적 ID |
| `detections[].depth` | Float | 거리 (미터) |
| `detections[].position_3d` | Object | 3D 좌표 |
| `detections[].height` | Float | 추정 높이 (미터) |
| `analytics` | Object | 분석 데이터 |
| `analytics.total_objects` | Integer | 총 객체 수 |
| `analytics.by_class` | Object | 클래스별 카운트 |
| `analytics.closest_distance` | Float | 가장 가까운 객체 거리 |
| `analytics.alerts` | Array | 경고 목록 |
| `performance` | Object | 성능 메트릭 |
| `performance.fps` | Float | 전체 FPS |
| `performance.detection_fps` | Float | 검출 FPS |
| `performance.depth_fps` | Float | 깊이 추정 FPS |
| `timestamp` | String | ISO 8601 타임스탬프 |

**Status Codes**:
- `200 OK`: 성공
- `500 Internal Server Error`: 비전 시스템 오류

**Polling 권장**: 100ms (10 Hz) - 너무 빠르면 서버 부하

---

### 4. 비디오 스트림 (MJPEG)

```http
GET /api/stream/video
```

**설명**: 실시간 비디오 스트림 (MJPEG over HTTP)

**응답 헤더**:
```http
Content-Type: multipart/x-mixed-replace; boundary=frame
```

**응답 본문**: MJPEG 스트림 (연속된 JPEG 이미지)

**사용 예시** (HTML):
```html
<img src="/api/stream/video" alt="Vision Feed">
```

**Features**:
- YOLOv8 바운딩 박스 오버레이
- 신뢰도 & 클래스 이름 라벨
- 추적 ID 표시
- 30 FPS

---

### 5. BEV 스트림 (MJPEG)

```http
GET /api/stream/bev
```

**설명**: Bird's Eye View 실시간 스트림 (MJPEG over HTTP)

**응답 헤더**:
```http
Content-Type: multipart/x-mixed-replace; boundary=frame
```

**응답 본문**: MJPEG 스트림 (연속된 JPEG 이미지)

**사용 예시** (HTML):
```html
<img src="/api/stream/bev" alt="Bird's Eye View">
```

**Features**:
- 탑다운 뷰 (5m x 5m)
- Multi-class 렌더링 (원/사각형/삼각형)
- 거리 색상 코딩 (녹색/노란색/빨간색)
- 카메라 위치 표시

---

### 6. 시스템 상태 (TODO - v1.4)

```http
GET /api/status
```

**설명**: 시스템 상태 및 헬스 체크

**응답 예시**:
```json
{
  "status": "ok",
  "version": "1.3.0",
  "uptime": 3600,
  "vision_system": {
    "active": true,
    "models_loaded": ["yolov8n", "depth_anything_v3"],
    "webcam": {
      "connected": true,
      "resolution": [640, 480],
      "fps": 30
    }
  },
  "performance": {
    "cpu_usage": 65.2,
    "memory_usage": 2048,
    "gpu_usage": 0
  }
}
```

---

### 7. 설정 조회 (TODO - v1.4)

```http
GET /api/settings
```

**설명**: 현재 설정 조회

**응답 예시**:
```json
{
  "vision": {
    "device": "cpu",
    "yolo": {
      "size": "n",
      "confidence": 0.35
    },
    "temporal": {
      "enabled": true,
      "history_size": 5,
      "confidence_alpha": 0.7,
      "bbox_alpha": 0.6
    }
  },
  "performance": {
    "detection_interval": 5,
    "depth_interval": 50,
    "target_fps": 30
  }
}
```

---

### 8. 설정 업데이트 (TODO - v1.4)

```http
POST /api/settings
Content-Type: application/json
```

**요청 본문**:
```json
{
  "vision": {
    "yolo": {
      "confidence": 0.4
    },
    "temporal": {
      "enabled": false
    }
  },
  "performance": {
    "detection_interval": 3
  }
}
```

**응답**:
```json
{
  "success": true,
  "message": "Settings updated successfully",
  "updated_fields": [
    "vision.yolo.confidence",
    "vision.temporal.enabled",
    "performance.detection_interval"
  ]
}
```

**Status Codes**:
- `200 OK`: 설정 업데이트 성공
- `400 Bad Request`: 잘못된 설정 값
- `500 Internal Server Error`: 설정 적용 실패

---

### 9. 스크린샷 생성 (TODO - v1.4)

```http
POST /api/screenshot
Content-Type: application/json
```

**요청 본문**:
```json
{
  "include_bev": true,
  "include_annotations": true
}
```

**응답**:
```json
{
  "success": true,
  "filename": "screenshot_2025-11-20T16-45-23.png",
  "url": "/static/screenshots/screenshot_2025-11-20T16-45-23.png",
  "size": 245632
}
```

---

### 10. 녹화 시작 (TODO - v1.4)

```http
POST /api/recording/start
Content-Type: application/json
```

**요청 본문**:
```json
{
  "duration": 60,
  "format": "mp4",
  "include_bev": false
}
```

**응답**:
```json
{
  "success": true,
  "recording_id": "rec_1234567890",
  "message": "Recording started",
  "duration": 60
}
```

---

### 11. 녹화 중지 (TODO - v1.4)

```http
POST /api/recording/stop
Content-Type: application/json
```

**요청 본문**:
```json
{
  "recording_id": "rec_1234567890"
}
```

**응답**:
```json
{
  "success": true,
  "filename": "recording_2025-11-20T16-45-23.mp4",
  "url": "/static/recordings/recording_2025-11-20T16-45-23.mp4",
  "duration": 45.2,
  "size": 15728640
}
```

---

## 📊 COCO Classes Reference

Vision Pro는 COCO 데이터셋의 80개 클래스를 검출합니다:

```python
COCO_CLASSES = [
    # 사람 (0)
    'person',

    # 차량 (1-8)
    'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',

    # 교통 관련 (9-13)
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',

    # 동물 (14-23)
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',

    # 액세서리 (24-28)
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',

    # 스포츠 (29-38)
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',

    # 주방 (39-50)
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',

    # 음식 (51-60)
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet',

    # 가전/가구 (61-70)
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink',

    # 기타 (71-79)
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]
```

---

## 🔧 Error Handling

### 에러 응답 포맷

```json
{
  "error": {
    "code": "VISION_SYSTEM_ERROR",
    "message": "Failed to process frame",
    "details": "YOLOv8 inference failed: CUDA out of memory",
    "timestamp": "2025-11-20T16:45:23.123456"
  }
}
```

### 에러 코드

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VISION_SYSTEM_ERROR` | 500 | 비전 시스템 처리 오류 |
| `WEBCAM_ERROR` | 500 | 웹캠 접근 오류 |
| `MODEL_LOAD_ERROR` | 500 | AI 모델 로드 실패 |
| `INVALID_SETTINGS` | 400 | 잘못된 설정 값 |
| `NOT_FOUND` | 404 | 리소스 없음 |
| `RATE_LIMIT_EXCEEDED` | 429 | 요청 제한 초과 |

---

## 📡 WebSocket API (TODO - v2.0)

### 연결

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/vision');

ws.onopen = () => {
  console.log('Connected to Vision Pro WebSocket');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected from Vision Pro WebSocket');
};
```

### 메시지 포맷

**Server → Client**:
```json
{
  "type": "detection_update",
  "data": {
    "detections": [...],
    "analytics": {...},
    "performance": {...}
  },
  "timestamp": "2025-11-20T16:45:23.123456"
}
```

**Client → Server**:
```json
{
  "action": "update_settings",
  "data": {
    "vision.yolo.confidence": 0.5
  }
}
```

---

## 🔍 사용 예제

### Python

```python
import requests
import time

# 데이터 조회
response = requests.get('http://localhost:8080/api/monitor/data')
data = response.json()

print(f"FPS: {data['performance']['fps']}")
print(f"Total objects: {data['analytics']['total_objects']}")

for det in data['detections']:
    print(f"- {det['class_name']}: {det['confidence']:.2f} @ {det['depth']:.1f}m")

# 폴링 루프 (10 Hz)
while True:
    response = requests.get('http://localhost:8080/api/monitor/data')
    data = response.json()
    # ... 데이터 처리 ...
    time.sleep(0.1)  # 100ms
```

### JavaScript (Fetch API)

```javascript
// 데이터 조회
async function fetchVisionData() {
  const response = await fetch('http://localhost:8080/api/monitor/data');
  const data = await response.json();

  console.log(`FPS: ${data.performance.fps}`);
  console.log(`Total objects: ${data.analytics.total_objects}`);

  data.detections.forEach(det => {
    console.log(`- ${det.class_name}: ${det.confidence.toFixed(2)} @ ${det.depth.toFixed(1)}m`);
  });
}

// 폴링 루프 (10 Hz)
setInterval(fetchVisionData, 100);
```

### cURL

```bash
# 데이터 조회
curl http://localhost:8080/api/monitor/data

# 예쁘게 출력 (jq 사용)
curl -s http://localhost:8080/api/monitor/data | jq .

# FPS만 추출
curl -s http://localhost:8080/api/monitor/data | jq '.performance.fps'

# 설정 업데이트 (TODO)
curl -X POST http://localhost:8080/api/settings \
  -H "Content-Type: application/json" \
  -d '{"vision": {"yolo": {"confidence": 0.4}}}'
```

---

## 📚 Rate Limiting (TODO - v2.0)

**제한**:
- `/api/monitor/data`: 100 requests/minute
- `/api/settings`: 10 requests/minute
- `/api/screenshot`: 5 requests/minute

**헤더**:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1637512345
```

---

## 🔒 CORS (TODO - v2.0)

**현재**: 모든 도메인 허용 (개발용)

**프로덕션**:
```python
CORS(app, resources={
    r"/api/*": {"origins": ["https://yourdomain.com"]}
})
```

---

**API 버전**: v1.3
**마지막 업데이트**: 2025-11-20
**상태**: Production Ready (일부 엔드포인트 TODO)
