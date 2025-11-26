# Examples - Vision Pro API

Vision Pro API 사용 예제 모음

---

## 📋 예제 목록

### 1. `simple_client.py` - 간단한 클라이언트

가장 기본적인 API 사용 예제입니다.

**기능**:
- `/api/monitor/data` 엔드포인트 폴링
- 실시간 검출 정보 출력
- FPS, 객체 카운트, 경고 표시

**사용법**:
```bash
# 서버 실행 (별도 터미널)
python app.py

# 클라이언트 실행
python examples/simple_client.py
```

**출력 예시**:
```
[16:45:23.123]
📊 Performance: FPS=28.5
🎯 Total Objects: 2
   By Class: {'person': 1, 'car': 1}

📍 Detections:
   1. person: 0.87 @ 2.5m
   2. car: 0.92 @ 5.8m

⚠️  Alerts:
   - [WARNING] Person detected within 3m
```

---

### 2. `data_logger.py` - CSV 데이터 로거

실시간 검출 데이터를 CSV 파일로 저장합니다.

**기능**:
- 1초마다 데이터 조회
- CSV 파일로 자동 저장 (`logs/detections_*.csv`)
- 타임스탬프, 클래스, 신뢰도, 거리 기록

**사용법**:
```bash
python examples/data_logger.py
```

**출력 파일**:
```csv
timestamp,class_id,class_name,confidence,depth_m,track_id,total_objects,fps
2025-11-20T16:45:23.123,0,person,0.8700,2.50,1,2,28.50
2025-11-20T16:45:23.123,2,car,0.9200,5.80,2,2,28.50
2025-11-20T16:45:24.456,0,person,0.8550,2.48,1,1,29.10
```

---

## 🛠️ 추가 예제 (TODO)

### 3. `alert_monitor.py` - 경고 모니터

특정 조건에서 알림을 발생시킵니다.

**조건**:
- 사람이 2m 이내 접근
- 특정 객체(차량, 동물) 검출
- 객체 수가 임계값 초과

### 4. `video_recorder.py` - 비디오 레코더

조건에 따라 자동으로 녹화를 시작합니다.

**트리거**:
- 특정 객체 검출
- 경고 발생
- 수동 시작/중지

### 5. `mqtt_bridge.py` - MQTT 브릿지

Vision Pro 데이터를 MQTT로 전송합니다.

**사용 케이스**:
- IoT 통합
- Home Assistant 연동
- 원격 모니터링

---

## 📚 API 참조

자세한 API 사양은 [API Reference](../docs/api/API_REFERENCE.md)를 참조하세요.

**주요 엔드포인트**:
- `GET /api/monitor/data` - 실시간 데이터
- `GET /api/stream/video` - 비디오 스트림 (MJPEG)
- `GET /api/stream/bev` - BEV 스트림 (MJPEG)

---

## 🔧 요구 사항

**Python 패키지**:
```bash
pip install requests  # HTTP 클라이언트
```

**서버 실행**:
```bash
python app.py
```

---

## 💡 팁

### 폴링 주기

API 폴링 주기는 용도에 따라 조정하세요:

- **Real-time UI**: 100ms (10 Hz) - `simple_client.py`
- **Data logging**: 1s (1 Hz) - `data_logger.py`
- **Analytics**: 5s (0.2 Hz) - 통계 분석용
- **Monitoring**: 10s (0.1 Hz) - 감시용

너무 빠른 폴링은 서버 부하를 증가시킵니다!

### 에러 처리

API 호출 시 항상 에러 처리를 포함하세요:

```python
import requests

try:
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    data = response.json()
except requests.exceptions.Timeout:
    print("서버 응답 시간 초과")
except requests.exceptions.ConnectionError:
    print("서버 연결 실패")
except requests.exceptions.HTTPError as e:
    print(f"HTTP 에러: {e}")
```

### 멀티스레딩

대량의 요청을 병렬로 처리하려면 멀티스레딩을 사용하세요:

```python
from concurrent.futures import ThreadPoolExecutor
import requests

def fetch_data():
    response = requests.get('http://localhost:8080/api/monitor/data')
    return response.json()

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(fetch_data) for _ in range(10)]
    results = [f.result() for f in futures]
```

---

## 🤝 기여

새로운 예제를 추가하려면:

1. 이 폴더에 Python 스크립트 생성
2. Shebang (`#!/usr/bin/env python3`) 추가
3. Docstring으로 설명 작성
4. 이 README에 예제 추가
5. Pull Request 제출

---

**마지막 업데이트**: 2025-11-20
**버전**: v1.3
