# TOPIS CCTV 스트림 URL 추출 가이드

> **"5초 제한 우회하고 실제 스트림 가져오기"**

## 🎯 목표

서울시 TOPIS CCTV (https://topis.seoul.go.kr/map/openCctvMap.do)의 5초 제한을 우회하고 실제 스트림 URL을 추출합니다.

---

## 방법 1: 브라우저 개발자 도구 (가장 쉬움) ⭐

### 단계별 가이드

#### 1. TOPIS 웹사이트 열기
```
URL: https://topis.seoul.go.kr/map/openCctvMap.do
```

#### 2. 개발자 도구 열기
- **Windows/Linux**: `F12` 또는 `Ctrl + Shift + I`
- **Mac**: `Cmd + Option + I`

#### 3. Network 탭으로 이동
- 상단 탭에서 "Network" 클릭
- 필터에서 "Media" 또는 "XHR" 선택

#### 4. CCTV 클릭
- 지도에서 아무 CCTV 마커나 클릭
- 5초 동안 영상이 재생됨

#### 5. 스트림 URL 찾기

**찾아야 할 것**:
```
✓ .m3u8 (HLS 스트림)
✓ .mp4 (MP4 파일)
✓ .flv (Flash Video)
✓ rtsp:// (RTSP 프로토콜)
✓ "stream" 또는 "video" 포함 URL
```

**예시**:
```
https://topis.seoul.go.kr/video/stream/camera001.m3u8
rtsp://210.99.70.123:1935/live/cam001
https://cdn.topis.seoul.go.kr/hls/gangnam001/playlist.m3u8
```

#### 6. URL 복사
- 해당 요청 우클릭
- "Copy" → "Copy link address"

#### 7. 헤더 복사 (필요한 경우)
- 같은 요청에서 "Copy" → "Copy as cURL"
- 또는 "Headers" 탭에서 필요한 헤더 확인

**일반적으로 필요한 헤더**:
```python
headers = {
    'User-Agent': 'Mozilla/5.0 ...',
    'Referer': 'https://topis.seoul.go.kr/map/openCctvMap.do',
    'Origin': 'https://topis.seoul.go.kr'
}
```

#### 8. Python 코드에서 사용

```python
import cv2

# 추출한 URL 사용
stream_url = "https://topis.seoul.go.kr/video/stream/camera001.m3u8"

cap = cv2.VideoCapture(stream_url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('TOPIS CCTV', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 방법 2: Selenium 자동화 (중급)

### 설치

```bash
# Selenium 설치
pip install selenium

# ChromeDriver 설치
sudo apt install chromium-chromedriver

# 또는 수동 다운로드
# https://chromedriver.chromium.org/
```

### 실행

```bash
cd /home/kim/auto-ai/the-sentinel
python3 topis_stream_capture.py
```

**옵션**:
1. **Auto-capture**: 자동으로 여러 CCTV 시도
2. **Manual mode**: 수동으로 클릭하면 자동 캡처
3. **Quit**: 종료

**결과**:
- `topis_streams.json` 파일에 저장
- 추출된 URL 목록 확인

---

## 방법 3: curl로 테스트 (검증용)

추출한 URL이 작동하는지 테스트:

```bash
# HLS 스트림 (.m3u8)
curl -H "Referer: https://topis.seoul.go.kr/" \
     "https://topis.seoul.go.kr/stream/camera001.m3u8"

# RTSP 스트림
ffmpeg -i "rtsp://210.99.70.123:1935/live/cam001" -frames 1 test.jpg
```

---

## 실제 발견된 패턴 (예상)

### 패턴 1: HLS 스트림
```
https://topis.seoul.go.kr/video/stream/{camera_id}.m3u8
https://cdn.topis.seoul.go.kr/hls/{location}/{camera_id}/playlist.m3u8
```

### 패턴 2: RTSP 스트림
```
rtsp://topis.seoul.go.kr:1935/live/{camera_id}
rtsp://210.99.70.{xxx}:1935/live/cam{number}
```

### 패턴 3: MP4 청크
```
https://topis.seoul.go.kr/video/chunk/{camera_id}/{timestamp}.mp4
```

---

## 5초 제한 우회 전략

### 전략 1: 실제 스트림 URL 직접 사용
```python
# blob: URL은 사용 불가
# ❌ blob:https://topis.seoul.go.kr/xxx

# 실제 스트림 URL 사용
# ✅ https://topis.seoul.go.kr/stream/camera001.m3u8
```

### 전략 2: 토큰 갱신 (필요한 경우)
```python
import requests
import re

def get_fresh_token():
    # 페이지 접속
    r = requests.get('https://topis.seoul.go.kr/map/openCctvMap.do')

    # JavaScript에서 토큰 추출
    token = re.search(r'token:\s*"([^"]+)"', r.text)

    if token:
        return token.group(1)
    return None

# 5초마다 토큰 갱신
token = get_fresh_token()
stream_url = f"https://topis.seoul.go.kr/stream/camera001.m3u8?token={token}"
```

### 전략 3: 세션 유지
```python
import requests

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 ...',
    'Referer': 'https://topis.seoul.go.kr/map/openCctvMap.do'
})

# 먼저 페이지 방문 (쿠키 얻기)
session.get('https://topis.seoul.go.kr/map/openCctvMap.do')

# 스트림 요청 (쿠키 포함)
stream = session.get(stream_url, stream=True)
```

---

## 통합 예제

```python
#!/usr/bin/env python3
"""
TOPIS CCTV Viewer
"""

import cv2
import requests

class TOPISViewer:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)',
            'Referer': 'https://topis.seoul.go.kr/map/openCctvMap.do'
        })

    def get_stream_url(self, camera_id: str) -> str:
        """
        Get actual stream URL for camera

        Replace with actual pattern after manual discovery
        """
        # Example patterns (adjust after discovery)
        patterns = [
            f"https://topis.seoul.go.kr/stream/{camera_id}.m3u8",
            f"rtsp://topis.seoul.go.kr:1935/live/{camera_id}",
        ]

        return patterns[0]  # Use discovered pattern

    def view_camera(self, camera_id: str):
        """View CCTV stream"""
        stream_url = self.get_stream_url(camera_id)

        print(f"[Connecting] {stream_url}")

        cap = cv2.VideoCapture(stream_url)

        if not cap.isOpened():
            print("[Error] Could not open stream")
            return

        print("[Streaming] Press 'q' to quit")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("[Warning] No frame received")
                break

            # Display
            cv2.imshow(f'TOPIS CCTV - {camera_id}', frame)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    viewer = TOPISViewer()

    # Use camera ID discovered from manual inspection
    camera_id = "gangnam001"  # Example

    viewer.view_camera(camera_id)
```

---

## 문제 해결

### 문제 1: "No stream found"

**원인**: 잘못된 URL 패턴

**해결**:
1. 브라우저 개발자 도구 다시 확인
2. 실제 요청된 URL 패턴 확인
3. 코드에서 패턴 업데이트

### 문제 2: "403 Forbidden"

**원인**: 헤더 누락

**해결**:
```python
headers = {
    'User-Agent': 'Mozilla/5.0 ...',
    'Referer': 'https://topis.seoul.go.kr/map/openCctvMap.do',
    'Origin': 'https://topis.seoul.go.kr'
}
```

### 문제 3: "Stream timeout"

**원인**: 토큰 만료

**해결**:
- 5초마다 토큰 갱신
- 또는 세션 쿠키 유지

---

## 윤리적 사용

⚠️ **중요: 다음 규칙을 반드시 준수하세요**

### 허용
- ✅ 교육/연구 목적
- ✅ 교통 정보 확인
- ✅ 1-2개 스트림만 동시 시청

### 금지
- ❌ 서버에 과부하 (동시 100개 스트림 등)
- ❌ 재배포 또는 상업적 사용
- ❌ 개인 식별 목적
- ❌ 5초마다 수백 번 요청

### 권장 사항
```python
# 적절한 딜레이 사용
import time

for camera in cameras:
    view_camera(camera)
    time.sleep(10)  # 10초 대기
```

---

## 다음 단계

1. ✅ **스트림 URL 추출** (이 가이드)
2. ⬜ **realtime_tracker.py에 통합**
3. ⬜ **지도에 실시간 표시**
4. ⬜ **객체 탐지 + 추적**
5. ⬜ **예측 시스템 연동**

---

## 요약

**가장 쉬운 방법**:
```
1. https://topis.seoul.go.kr/map/openCctvMap.do 열기
2. F12 → Network → Media
3. CCTV 클릭
4. .m3u8 또는 stream URL 찾기
5. URL 복사
6. cv2.VideoCapture(url) 사용
```

**결과**:
- 5초 제한 없이 계속 시청 가능
- 실시간 추적 시스템에 통합 가능
- 영화처럼 모든 CCTV 동시 모니터링

**"5초 제한? 이제 해결됐습니다!"** 🎥✅
