# Documentation - Vision Pro v1.3

Vision Pro 완전한 문서 모음

---

## 📚 문서 인덱스

### 🚀 시작하기

1. **[README.md](../README.md)** - 프로젝트 개요
   - Vision Pro 소개
   - 주요 기능
   - Quick Start
   - 산업별 use cases

2. **[DEPLOYMENT.md](../DEPLOYMENT.md)** - 배포 가이드
   - 로컬 실행
   - 프로덕션 배포
   - Docker 배포
   - 보안 설정
   - API 키 관리 ⭐

---

### 📖 사용자 가이드

프로젝트 사용 방법을 단계별로 설명합니다.

**향후 작성 예정**:
- Quick Start Guide - 5분 안에 시작하기
- Configuration Guide - 설정 상세 설명
- Troubleshooting Guide - 문제 해결

---

### 🔧 기술 문서

시스템의 기술적 세부사항을 설명합니다.

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** - 시스템 아키텍처 ⭐
   - 시스템 구조도 (ASCII art)
   - 데이터 플로우
   - 컴포넌트 상세
   - AI 모델 파이프라인
   - 성능 프로파일
   - 네트워크 아키텍처
   - 배포 아키텍처
   - 확장성 고려사항

2. **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - 프로젝트 구조 ⭐
   - 전체 디렉토리 트리
   - 주요 디렉토리 설명
   - 루트 파일 설명
   - 데이터 흐름
   - 개발 가이드

3. **[IMPROVEMENTS.md](../IMPROVEMENTS.md)** - 성능 개선
   - v1.1 Stability & Performance
   - 에러 핸들링 강화
   - JSON 로깅 최적화
   - Temporal Smoothing
   - 설정 시스템
   - 성능 비교 테이블

4. **[TEMPORAL_SMOOTHING.md](../TEMPORAL_SMOOTHING.md)** - Temporal Smoothing
   - EMA 알고리즘 상세
   - Confidence Hysteresis
   - Object Persistence
   - 코드 설명
   - 성능 벤치마크

---

### 🎨 UI/UX 문서

사용자 인터페이스 관련 문서입니다.

1. **[UI_ENHANCEMENTS.md](../UI_ENHANCEMENTS.md)** - UI/UX 개선 (v1.2)
   - Chart.js 실시간 그래프
   - Settings Panel (9개 파라미터)
   - 사용 가이드
   - 성능 영향 분석

2. **[ADVANCED_FEATURES.md](../ADVANCED_FEATURES.md)** - 고급 기능 (v1.3) ⭐
   - 📸 스크린샷 캡처
   - ⏺ 비디오 녹화
   - 🔔 웹 알림 시스템
   - Action Buttons UI
   - 사용 시나리오
   - 문제 해결

---

### 📡 API 문서

REST API 완전한 레퍼런스입니다.

1. **[api/API_REFERENCE.md](api/API_REFERENCE.md)** - API 레퍼런스 ⭐
   - Base URL
   - 모든 엔드포인트 설명
   - 요청/응답 예시
   - COCO Classes 참조
   - 에러 핸들링
   - 사용 예제 (Python, JavaScript, cURL)

2. **WebSocket API** (TODO - v2.0)
   - 실시간 양방향 통신
   - 메시지 포맷
   - 연결 관리

---

### 📝 버전 관리

프로젝트 변경사항을 기록합니다.

1. **[CHANGELOG.md](../CHANGELOG.md)** - 버전 히스토리 ⭐
   - v1.3: Advanced Features
   - v1.2: UI/UX Enhancement
   - v1.1: Stability & Performance
   - v1.0: Initial Release

2. **[PROJECT_SUMMARY.md](../PROJECT_SUMMARY.md)** - 프로젝트 요약 ⭐
   - 프로젝트 개요
   - 목표 달성 현황
   - 주요 성과 (성능 개선 테이블)
   - 생성된 파일 목록
   - 기술 스택
   - 성능 지표

---

## 📊 문서 구조 (시각화)

```
docs/
│
├── 📄 README.md                    # 현재 문서 (문서 인덱스)
│
├── 📄 ARCHITECTURE.md              # 시스템 아키텍처 (시각화 포함)
├── 📄 PROJECT_STRUCTURE.md         # 프로젝트 구조 (트리 구조)
│
├── 📁 api/                         # API 문서
│   ├── 📄 API_REFERENCE.md         # 완전한 API 레퍼런스
│   └── 📄 WEBSOCKET.md             # WebSocket API (TODO)
│
├── 📁 guides/                      # 사용자 가이드 (TODO)
│   ├── 📄 QUICK_START.md
│   ├── 📄 CONFIGURATION.md
│   └── 📄 TROUBLESHOOTING.md
│
└── 📁 images/                      # 문서용 이미지 (TODO)
    ├── architecture.png
    ├── screenshot.png
    └── bev.png
```

---

## 📁 루트 문서 (프로젝트 루트)

프로젝트 루트에 있는 주요 문서들:

```
vision-mamba-control/
│
├── 📄 README.md                    # 프로젝트 메인 README
│
├── 📄 CHANGELOG.md                 # 버전 히스토리
├── 📄 PROJECT_SUMMARY.md           # 프로젝트 완료 보고서
│
├── 📄 IMPROVEMENTS.md              # 성능 개선 상세
├── 📄 TEMPORAL_SMOOTHING.md        # 기술 문서
├── 📄 UI_ENHANCEMENTS.md           # UI/UX 개선
├── 📄 ADVANCED_FEATURES.md         # 고급 기능 가이드
│
├── 📄 DEPLOYMENT.md                # 배포 가이드
│
└── 📁 docs/                        # 이 폴더 (추가 문서)
```

---

## 🔍 문서 검색 가이드

### 목적별 문서 찾기

**"어떻게 시작하나요?"**
→ [README.md](../README.md) → Quick Start

**"API는 어떻게 사용하나요?"**
→ [api/API_REFERENCE.md](api/API_REFERENCE.md)

**"설정을 변경하려면?"**
→ [DEPLOYMENT.md](../DEPLOYMENT.md) → Configuration

**"성능이 낮은데 최적화하려면?"**
→ [IMPROVEMENTS.md](../IMPROVEMENTS.md) → Performance

**"시스템 구조를 이해하고 싶어요"**
→ [ARCHITECTURE.md](ARCHITECTURE.md)

**"프로젝트 파일 구조는?"**
→ [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

**"스크린샷/녹화는 어떻게?"**
→ [ADVANCED_FEATURES.md](../ADVANCED_FEATURES.md)

**"배포는 어떻게 하나요?"**
→ [DEPLOYMENT.md](../DEPLOYMENT.md)

**"API 키는 어떻게 관리하나요?"**
→ [DEPLOYMENT.md](../DEPLOYMENT.md) → Security

**"버전별 변경사항은?"**
→ [CHANGELOG.md](../CHANGELOG.md)

---

## 📈 문서 통계

**총 문서**: 15개 (8개 루트 + 3개 docs/ + 4개 examples/scripts)

**문서 라인 수**:
- 루트 문서: ~1,800 lines
- docs/ 문서: ~700 lines
- 총: ~2,500 lines

**문서 유형**:
- README: 4개
- 기술 문서: 5개
- API 문서: 1개
- 가이드: 4개
- 버전 관리: 2개

---

## ✨ 문서 작성 가이드라인

새 문서를 추가할 때:

1. **명확한 제목**: 문서 목적을 한눈에 파악
2. **목차 추가**: 긴 문서는 목차 필수
3. **코드 예제**: 실제 사용 가능한 예제 포함
4. **시각화**: ASCII art, 다이어그램, 표 사용
5. **버전 명시**: 문서 하단에 버전 & 날짜 추가
6. **링크**: 관련 문서 링크 추가

**예시**:
```markdown
# Document Title - Vision Pro v1.3

간단한 설명

---

## 목차

1. [Section 1](#section-1)
2. [Section 2](#section-2)

---

## Section 1

내용...

## Section 2

내용...

---

**문서 버전**: v1.3
**마지막 업데이트**: 2025-11-20
**상태**: Production Ready
```

---

## 🔗 외부 링크

### AI 모델 문서

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Depth Anything V2/V3](https://github.com/DepthAnything/Depth-Anything-V2)
- [PyTorch Documentation](https://pytorch.org/docs/)

### 웹 기술

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Chart.js Documentation](https://www.chartjs.org/docs/)
- [MDN Web APIs](https://developer.mozilla.org/en-US/docs/Web/API)

---

## 🤝 기여

문서 개선 제안:

1. 오타, 에러 발견 → Issue 생성
2. 새 문서 제안 → Pull Request
3. 번역 기여 → Translation 폴더

---

**문서 인덱스 버전**: v1.3
**마지막 업데이트**: 2025-11-20
**총 문서 수**: 15개
**상태**: Complete
