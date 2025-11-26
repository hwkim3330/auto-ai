# Project Structure - Vision Pro

프로젝트 디렉토리 구조 상세 설명

---

## 📁 전체 구조

```
vision-mamba-control/
│
├── 📄 README.md                    # 프로젝트 개요
├── 📄 CHANGELOG.md                 # 버전 히스토리
├── 📄 PROJECT_SUMMARY.md           # 프로젝트 완료 보고서
├── 📄 IMPROVEMENTS.md              # 성능 개선 상세
├── 📄 TEMPORAL_SMOOTHING.md        # Temporal Smoothing 기술 문서
├── 📄 UI_ENHANCEMENTS.md           # UI/UX 개선 가이드
├── 📄 ADVANCED_FEATURES.md         # 고급 기능 가이드
├── 📄 DEPLOYMENT.md                # 배포 가이드 (보안 포함)
│
├── 📄 requirements.txt             # Python 의존성
├── 📄 config.yaml                  # 시스템 설정
├── 📄 .env.example                 # 환경 변수 템플릿
├── 📄 .gitignore                   # Git 제외 파일
│
├── 📜 app.py                       # 메인 Flask 서버
│
├── 📁 src/                         # 소스 코드
│   ├── 📁 vision/                  # AI 비전 시스템
│   │   ├── __init__.py
│   │   ├── vision_system.py        # 통합 비전 시스템
│   │   ├── object_detector.py      # YOLOv8 래퍼
│   │   ├── depth_estimator.py      # Depth Anything V3 래퍼
│   │   ├── temporal_smoother.py    # 프레임 일관성 시스템
│   │   └── bev_renderer.py         # Bird's Eye View 렌더러
│   │
│   ├── 📁 models/                  # AI 모델 (자동 다운로드)
│   │   ├── yolov8n.pt              # YOLOv8n weights (~6 MB)
│   │   └── depth_anything_v3.pth   # Depth model (~100 MB)
│   │
│   └── 📁 utils/                   # 유틸리티
│       ├── __init__.py
│       ├── config_loader.py        # YAML 설정 로더
│       └── logger.py               # 로깅 유틸리티
│
├── 📁 web/                         # 웹 인터페이스
│   ├── 📁 templates/               # HTML 템플릿
│   │   ├── index.html              # 홈페이지
│   │   └── monitor.html            # 메인 모니터링 UI
│   │
│   ├── 📁 static/                  # 정적 파일
│   │   ├── 📁 css/
│   │   ├── 📁 js/
│   │   ├── 📁 images/
│   │   ├── 📁 screenshots/         # 스크린샷 저장
│   │   └── 📁 recordings/          # 녹화 파일 저장
│   │
│   └── 📁 api/                     # API 엔드포인트 (app.py에 통합)
│
├── 📁 docs/                        # 문서
│   ├── 📄 README.md                # 문서 인덱스
│   ├── 📄 ARCHITECTURE.md          # 시스템 아키텍처
│   ├── 📄 PROJECT_STRUCTURE.md     # 현재 문서
│   │
│   ├── 📁 api/                     # API 문서
│   │   ├── 📄 API_REFERENCE.md     # 완전한 API 레퍼런스
│   │   └── 📄 WEBSOCKET.md         # WebSocket API (TODO)
│   │
│   ├── 📁 guides/                  # 사용자 가이드
│   │   ├── 📄 QUICK_START.md       # 빠른 시작 가이드
│   │   ├── 📄 CONFIGURATION.md     # 설정 가이드
│   │   └── 📄 TROUBLESHOOTING.md   # 문제 해결
│   │
│   └── 📁 images/                  # 문서용 이미지
│       ├── architecture.png
│       ├── screenshot.png
│       └── bev.png
│
├── 📁 examples/                    # 사용 예제
│   ├── 📄 README.md                # 예제 설명
│   ├── 📜 simple_client.py         # 간단한 API 클라이언트
│   ├── 📜 data_logger.py           # CSV 데이터 로거
│   ├── 📜 alert_monitor.py         # 경고 모니터 (TODO)
│   └── 📜 video_recorder.py        # 비디오 레코더 (TODO)
│
├── 📁 scripts/                     # 유틸리티 스크립트
│   ├── 📜 install.sh               # 자동 설치 스크립트
│   ├── 📜 start.sh                 # 서버 시작 스크립트 (TODO)
│   ├── 📜 stop.sh                  # 서버 중지 스크립트 (TODO)
│   ├── 📜 backup.sh                # 데이터 백업 스크립트 (TODO)
│   └── 📜 benchmark.py             # 성능 벤치마크 (TODO)
│
├── 📁 tests/                       # 테스트 코드
│   ├── __init__.py
│   ├── test_object_detector.py     # YOLOv8 테스트
│   ├── test_depth_estimator.py     # Depth 모델 테스트
│   ├── test_temporal_smoother.py   # Smoothing 테스트
│   ├── test_bev_renderer.py        # BEV 테스트
│   ├── test_api.py                 # API 엔드포인트 테스트
│   └── test_integration.py         # 통합 테스트
│
├── 📁 logs/                        # 로그 파일 (자동 생성)
│   ├── vision-pro.log
│   ├── depth_log.json
│   └── error.log
│
└── 📁 venv/                        # Python 가상 환경
    ├── bin/
    ├── lib/
    └── ...
```

---

## 📂 주요 디렉토리 설명

### `/src/vision/` - AI 비전 시스템

**핵심 컴포넌트**:

1. **`vision_system.py`** - 통합 비전 시스템
   - 모든 AI 모델 조율
   - 프레임 처리 파이프라인
   - 결과 병합 및 반환

2. **`object_detector.py`** - YOLOv8 래퍼
   - 객체 검출
   - 바운딩 박스 생성
   - 클래스 분류

3. **`depth_estimator.py`** - Depth Anything V3 래퍼
   - 단안 깊이 추정
   - 3D 좌표 계산
   - 높이 추정

4. **`temporal_smoother.py`** - 프레임 일관성
   - Exponential Moving Average
   - Confidence Hysteresis
   - Object Persistence

5. **`bev_renderer.py`** - Bird's Eye View
   - 탑다운 뷰 생성
   - Multi-class 렌더링
   - 거리 색상 코딩

---

### `/web/` - 웹 인터페이스

**구조**:

1. **`templates/monitor.html`** - 메인 UI
   - **1,800+ lines** of HTML/CSS/JS
   - 실시간 비디오 피드
   - Chart.js 그래프
   - Settings Panel
   - Action Buttons

2. **`static/`** - 정적 리소스
   - CSS 스타일시트
   - JavaScript 모듈
   - 이미지 & 아이콘
   - 스크린샷 & 녹화 파일

---

### `/docs/` - 문서

**문서 카테고리**:

1. **루트 문서** (프로젝트 루트)
   - README.md: 프로젝트 개요
   - CHANGELOG.md: 버전 히스토리
   - DEPLOYMENT.md: 배포 가이드
   - IMPROVEMENTS.md: 개선 사항
   - etc.

2. **`docs/api/`** - API 문서
   - API_REFERENCE.md: 완전한 API 레퍼런스
   - 모든 엔드포인트 설명
   - 요청/응답 예시

3. **`docs/guides/`** - 사용자 가이드
   - 빠른 시작
   - 설정 가이드
   - 문제 해결

4. **`docs/`** - 기술 문서
   - ARCHITECTURE.md: 시스템 구조
   - PROJECT_STRUCTURE.md: 현재 문서

---

### `/examples/` - 사용 예제

**예제 스크립트**:

1. **`simple_client.py`** - 기본 API 클라이언트
   - `/api/monitor/data` 폴링
   - 실시간 정보 출력

2. **`data_logger.py`** - 데이터 로거
   - CSV 파일로 저장
   - 타임스탬프, 클래스, 거리 기록

**향후 예제** (TODO):
- `alert_monitor.py`: 조건부 알림
- `video_recorder.py`: 자동 녹화
- `mqtt_bridge.py`: MQTT 통합

---

### `/scripts/` - 유틸리티 스크립트

**스크립트**:

1. **`install.sh`** - 자동 설치
   - 가상 환경 생성
   - 의존성 설치
   - 디렉토리 생성
   - .env 파일 복사

**향후 스크립트** (TODO):
- `start.sh`: 서버 시작
- `stop.sh`: 서버 중지
- `backup.sh`: 데이터 백업
- `benchmark.py`: 성능 테스트

---

### `/tests/` - 테스트

**테스트 구조**:

1. **Unit Tests**
   - `test_object_detector.py`
   - `test_depth_estimator.py`
   - `test_temporal_smoother.py`
   - `test_bev_renderer.py`

2. **Integration Tests**
   - `test_integration.py`: 전체 파이프라인
   - `test_api.py`: REST API 엔드포인트

**실행**:
```bash
pytest tests/
pytest tests/test_object_detector.py -v
```

---

## 📜 루트 파일 설명

### `app.py` - 메인 서버

**구조**:
- Flask 앱 초기화
- API 라우트 정의
- 비전 시스템 관리
- 백그라운드 스레드

**주요 라우트**:
- `/` - 홈페이지
- `/monitor` - 모니터링 UI
- `/api/monitor/data` - 실시간 데이터
- `/api/stream/video` - 비디오 스트림
- `/api/stream/bev` - BEV 스트림

---

### `requirements.txt` - Python 의존성

**주요 패키지**:
```txt
Flask==3.0.0
ultralytics==8.0.0       # YOLOv8
opencv-python==4.8.0
numpy==1.24.0
torch==2.0.0
torchvision==0.15.0
pyyaml==6.0.0
python-dotenv==1.0.0
```

---

### `config.yaml` - 시스템 설정

**섹션**:
- `server`: 포트, 호스트
- `vision`: AI 모델 설정
- `webcam`: 카메라 설정
- `logging`: 로그 레벨
- `performance`: 최적화
- `ui`: UI 설정
- `alerts`: 알림 임계값

---

### `.env.example` - 환경 변수

**내용**:
```bash
# API Keys
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here

# Server
FLASK_SECRET_KEY=random_secret
PORT=8080
```

---

### `.gitignore` - Git 제외

**제외 항목**:
- `venv/` - 가상 환경
- `.env` - 환경 변수 (비밀 키)
- `*.pyc` - Python 바이트코드
- `logs/` - 로그 파일
- `*.pt`, `*.pth` - AI 모델 weights

---

## 🔄 데이터 흐름

```
Webcam
  │
  ▼
app.py (vision_update_loop)
  │
  ▼
vision_system.py
  ├──> object_detector.py (YOLOv8)
  ├──> depth_estimator.py (Depth Anything V3)
  ├──> temporal_smoother.py (EMA)
  └──> bev_renderer.py (Top-down view)
  │
  ▼
Results (JSON)
  │
  ▼
Flask API (/api/monitor/data)
  │
  ▼
Web UI (monitor.html)
  ├──> Chart.js (Graphs)
  ├──> Video Feed
  ├──> BEV Feed
  └──> Detection Log
```

---

## 📈 파일 통계

**코드**:
- Python: ~5,000 lines
- HTML/CSS/JS: ~1,800 lines (monitor.html)
- 총 코드: ~7,000 lines

**문서**:
- Markdown: ~2,500 lines (8개 주요 문서)

**파일 수**:
- Python 모듈: 15개
- 문서: 15개
- 예제: 2개
- 스크립트: 1개

---

## 🔧 개발 가이드

### 새 기능 추가

1. **백엔드**: `src/vision/` 또는 `app.py`
2. **프론트엔드**: `web/templates/monitor.html`
3. **API**: `app.py`의 라우트 추가
4. **문서**: 해당 문서 업데이트
5. **테스트**: `tests/` 에 테스트 추가

### 폴더 추가 규칙

- `/src/`: 핵심 로직
- `/web/`: UI 코드
- `/docs/`: 문서
- `/examples/`: 사용 예제
- `/scripts/`: 자동화 스크립트
- `/tests/`: 테스트 코드

---

**문서 버전**: v1.3
**마지막 업데이트**: 2025-11-20
**상태**: Production Ready
