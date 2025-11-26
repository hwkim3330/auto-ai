# 🚀 Vision Pro → AI Vision Platform
## 압도적인 산업용 AI 비전 플랫폼 구축 계획

**목표**: 단순 CCTV를 넘어, 모든 산업의 "눈"이 되는 범용 AI 비전 플랫폼

---

## 🎯 1. 핵심 전략: "One Vision, All Industries"

### 1.1 플랫폼 컨셉
```
Vision Pro Platform
├─ 1️⃣ Core Vision Engine (범용 비전 AI)
│  ├─ YOLOv8 / YOLOv11 (객체 탐지)
│  ├─ Depth Anything V3 (깊이 추정)
│  ├─ SAM 2 (Segment Anything Model 2)
│  ├─ GroundingDINO (Open-vocabulary Detection)
│  └─ CLIP / SigLIP (Vision-Language)
│
├─ 2️⃣ Multi-AI Integration
│  ├─ Gemini 2.0 Flash (Google - 멀티모달)
│  ├─ GPT-4V / GPT-4o (OpenAI - 비전+추론)
│  ├─ Claude 3.5 Sonnet (Anthropic - 분석)
│  ├─ LLaVA / InternVL (오픈소스 VLM)
│  └─ HyperCLOVA X (Naver - 한국어)
│
├─ 3️⃣ Industry-Specific Modules
│  ├─ 🏭 Manufacturing (제조업)
│  ├─ 🏥 Healthcare (의료)
│  ├─ 🏪 Retail (리테일)
│  ├─ 🚗 Automotive (자동차)
│  ├─ 🏗️ Construction (건설)
│  ├─ 🌾 Agriculture (농업)
│  ├─ 🏙️ Smart City (스마트시티)
│  └─ 🔒 Security (보안)
│
└─ 4️⃣ Deployment Options
   ├─ 🌐 Web App (GitHub Pages)
   ├─ 📱 Mobile App (Flutter)
   ├─ 🖥️ Desktop App (Electron)
   ├─ 🔌 Edge Device (Jetson, RaspberryPi)
   └─ ☁️ Cloud API (AWS/GCP/Azure)
```

---

## 🏭 2. 산업별 특화 기능 (Industry Modules)

### 2.1 제조업 (Manufacturing) - "Smart Factory AI"

**문제점**:
- 수작업 품질 검사 (느리고 부정확)
- 작업자 안전사고
- 생산 라인 비효율

**Vision Pro 솔루션**:
```python
Manufacturing Module:
├─ Quality Inspection (품질 검사)
│  ├─ Defect Detection (불량 검출)
│  │  ├─ 표면 흠집, 균열, 변색
│  │  ├─ 치수 측정 (±0.1mm 정밀도)
│  │  └─ OCR (시리얼 번호, 로트 번호)
│  ├─ Assembly Verification (조립 검증)
│  │  ├─ 부품 누락 감지
│  │  └─ 조립 순서 확인
│  └─ Real-time Dashboard
│     ├─ 불량률 실시간 그래프
│     └─ 자동 알림 (불량률 급증 시)
│
├─ Worker Safety (작업자 안전)
│  ├─ PPE Detection (안전장구 착용 감지)
│  │  ├─ 헬멧, 안전화, 장갑
│  │  └─ 미착용 시 자동 경고
│  ├─ Dangerous Zone Monitoring
│  │  ├─ 위험 구역 침입 감지
│  │  └─ 기계 가동 중 접근 경고
│  └─ Fall Detection (낙상 감지)
│
└─ Process Optimization (공정 최적화)
   ├─ Cycle Time Analysis (사이클 타임 분석)
   ├─ Bottleneck Detection (병목 구간 파악)
   └─ Predictive Maintenance (예지 보전)
      └─ 장비 이상 징후 조기 감지
```

**ROI**:
- 품질 검사 시간: 90% 감소 (10분 → 1분)
- 불량률: 50% 감소
- 안전사고: 80% 감소
- 연간 절감액: **대기업 기준 10억원+**

### 2.2 의료 (Healthcare) - "Medical Vision AI"

**문제점**:
- 의료 영상 판독 시간 오래 걸림
- 의사 부족 (특히 지방)
- 환자 모니터링 인력 부족

**Vision Pro 솔루션**:
```python
Healthcare Module:
├─ Medical Imaging (의료 영상 분석)
│  ├─ X-Ray Analysis (흉부 X-ray)
│  │  ├─ 폐렴, 결핵, 폐암 의심 소견
│  │  └─ 심비대, 흉수 감지
│  ├─ CT/MRI Analysis
│  │  ├─ 종양 세그멘테이션
│  │  └─ 크기/위치 자동 측정
│  └─ Pathology (병리 슬라이드)
│     ├─ 암세포 검출
│     └─ 등급 분류
│
├─ Patient Monitoring (환자 모니터링)
│  ├─ Fall Detection (낙상 감지)
│  │  └─ 응급 호출 자동 발송
│  ├─ Activity Recognition
│  │  ├─ 침상 이탈 감지
│  │  └─ 이상 행동 감지
│  └─ Vital Signs Estimation (비접촉 생체 신호)
│     ├─ 호흡수 추정 (카메라만으로)
│     └─ 움직임 패턴 분석
│
└─ Triage Support (응급실 분류 지원)
   ├─ Trauma Detection (외상 감지)
   └─ Priority Recommendation (우선순위 제안)
```

**협력 대상**:
- 루닛 (Lunit): 폐암/유방암 AI
- 뷰노 (Vuno): 의료 영상 AI
- 서울대병원, 삼성서울병원

**규제**:
- 식약처 의료기기 인허가 필요 (Class II/III)
- 임상시험 요구
- → Phase 1: 의사 보조용 (인허가 불요)
- → Phase 2: 진단 보조 (인허가 필요)

### 2.3 리테일 (Retail) - "Smart Store AI"

**문제점**:
- 무인 매장 도난
- 재고 관리 어려움
- 고객 동선 분석 부족

**Vision Pro 솔루션**:
```python
Retail Module:
├─ Loss Prevention (도난 방지)
│  ├─ Suspicious Behavior Detection
│  │  ├─ 제품 주머니에 넣기
│  │  ├─ 긴 시간 배회
│  │  └─ 태그 제거 시도
│  ├─ Self-checkout Monitoring
│  │  ├─ 제품 스캔 누락 감지
│  │  └─ 바코드 조작 감지
│  └─ Real-time Alert
│
├─ Inventory Management (재고 관리)
│  ├─ Shelf Monitoring (진열대 모니터링)
│  │  ├─ 품절 감지 (자동 알림)
│  │  ├─ 잘못된 진열 감지
│  │  └─ 가격표 오류 감지
│  ├─ Stock Level Estimation
│  │  └─ 재고 수량 자동 추정
│  └─ Expiry Date Check (유통기한 확인)
│
└─ Customer Analytics (고객 분석)
   ├─ Heatmap (고객 동선 히트맵)
   ├─ Dwell Time (체류 시간 분석)
   ├─ Demographics (연령/성별 추정)
   └─ Conversion Rate (구매 전환율)
      └─ 들어온 사람 vs 구매한 사람
```

**타겟 고객**:
- GS25, CU, 세븐일레븐 (무인 편의점)
- 이마트24, 위드미 (무인 슈퍼)
- 롯데마트, 이마트 (대형마트)

### 2.4 자동차 (Automotive) - "Vehicle Intelligence AI"

**Vision Pro 솔루션**:
```python
Automotive Module:
├─ Manufacturing (제조)
│  ├─ Weld Inspection (용접 검사)
│  ├─ Paint Defect Detection (도장 불량)
│  └─ Assembly Quality Check
│
├─ Dealership (판매점)
│  ├─ Damage Assessment (차량 손상 평가)
│  │  ├─ 스크래치, 찌그러짐 자동 감지
│  │  └─ 수리 비용 자동 산정
│  └─ Vehicle Inspection (차량 검사)
│
└─ Infrastructure (인프라)
   ├─ Traffic Monitoring (교통 모니터링)
   ├─ Parking Management (주차 관리)
   │  ├─ 빈 자리 감지
   │  ├─ 불법 주차 감지
   │  └─ 차량 번호 인식
   └─ Toll System (통행료 시스템)
```

### 2.5 건설 (Construction) - "Construction Safety AI"

```python
Construction Module:
├─ Safety Monitoring
│  ├─ PPE Compliance (안전장구 착용)
│  ├─ Unsafe Behavior Detection
│  │  ├─ 추락 위험 행동
│  │  └─ 안전 펜스 침입
│  └─ Equipment Safety
│     └─ 중장비 작업 반경 모니터링
│
├─ Progress Tracking (공정 진행률)
│  ├─ 3D Reconstruction (현장 3D 복원)
│  ├─ Progress vs Plan (계획 대비 실적)
│  └─ Material Tracking (자재 추적)
│
└─ Quality Control (품질 관리)
   ├─ Crack Detection (균열 감지)
   └─ Dimension Verification (치수 검증)
```

### 2.6 농업 (Agriculture) - "Smart Farm AI"

```python
Agriculture Module:
├─ Crop Monitoring (작물 모니터링)
│  ├─ Growth Stage Detection (생육 단계)
│  ├─ Disease Detection (병해 감지)
│  │  ├─ 잎 변색, 시들음
│  │  └─ 해충 감지
│  └─ Yield Estimation (수확량 예측)
│
├─ Livestock Monitoring (가축 모니터링)
│  ├─ Health Monitoring (건강 상태)
│  │  ├─ 행동 패턴 이상
│  │  └─ 질병 조기 감지
│  ├─ Counting (개체 수 카운팅)
│  └─ Behavior Analysis (행동 분석)
│
└─ Automation (자동화)
   ├─ Weed Detection (잡초 감지)
   └─ Ripeness Detection (성숙도 감지)
```

### 2.7 스마트시티 (Smart City) - "Urban Intelligence AI"

```python
Smart City Module:
├─ Traffic Management
│  ├─ Traffic Flow Optimization
│  ├─ Accident Detection
│  └─ Congestion Prediction
│
├─ Public Safety
│  ├─ Crowd Monitoring (군중 모니터링)
│  ├─ Fight/Violence Detection
│  └─ Emergency Response
│
├─ Environmental Monitoring
│  ├─ Illegal Dumping Detection
│  ├─ Fire/Smoke Detection
│  └─ Flood Monitoring
│
└─ Infrastructure Management
   ├─ Pothole Detection (도로 파손)
   ├─ Streetlight Monitoring
   └─ Public Facility Maintenance
```

### 2.8 보안 (Security) - "Advanced Security AI"

```python
Security Module:
├─ Perimeter Security (경계 보안)
│  ├─ Intrusion Detection (침입 감지)
│  ├─ Loitering Detection (배회 감지)
│  └─ Tailgating Detection (동반 출입)
│
├─ Access Control (출입 통제)
│  ├─ Face Recognition (얼굴 인식)
│  ├─ License Plate Recognition
│  └─ Unauthorized Access Alert
│
├─ Threat Detection (위협 감지)
│  ├─ Weapon Detection (무기 감지)
│  ├─ Suspicious Object Detection
│  └─ Aggressive Behavior Detection
│
└─ Forensics (포렌식)
   ├─ Person Re-identification
   ├─ Object Search (특정 객체 검색)
   └─ Timeline Reconstruction
```

---

## 🤖 3. Multi-AI Integration 전략

### 3.1 AI 모델 역할 분담

```python
AI Orchestra:
├─ 🎯 Detection Layer (탐지)
│  ├─ YOLOv11: 실시간 객체 탐지
│  ├─ SAM 2: 정밀 세그멘테이션
│  └─ GroundingDINO: 텍스트 기반 탐지
│
├─ 🧠 Understanding Layer (이해)
│  ├─ Gemini 2.0 Flash:
│  │  ├─ 영상 설명 생성
│  │  ├─ 실시간 스트리밍 분석
│  │  └─ 멀티모달 추론
│  ├─ GPT-4o:
│  │  ├─ 복잡한 시나리오 분석
│  │  ├─ 의사결정 지원
│  │  └─ 보고서 생성
│  ├─ Claude 3.5:
│  │  ├─ 장문 영상 분석
│  │  ├─ 패턴 인식
│  │  └─ 이상 징후 설명
│  └─ HyperCLOVA X:
│     └─ 한국어 자연어 처리
│
├─ 🎬 Action Layer (행동)
│  ├─ LLaVA / InternVL:
│  │  ├─ 로컬 VLM (프라이버시)
│  │  └─ 빠른 응답
│  └─ Fine-tuned Models:
│     └─ 산업별 특화 모델
│
└─ 📊 Analytics Layer (분석)
   ├─ Time-series Analysis
   ├─ Anomaly Detection (AutoEncoder)
   └─ Predictive Analytics
```

### 3.2 실시간 AI 파이프라인

```
카메라 입력
    ↓
[Edge Processing]
├─ YOLOv11 (객체 탐지) - 30 FPS
├─ Depth Anything V3 (깊이) - 6 FPS
└─ Tracking (추적) - 30 FPS
    ↓
[Event Trigger]
이상 감지 or 특정 조건 만족 시
    ↓
[Cloud AI - On-demand]
├─ Gemini 2.0: "이 상황 설명해줘"
├─ GPT-4o: "다음 행동 예측"
└─ Claude: "위험도 평가"
    ↓
[Decision & Action]
├─ 알림 발송
├─ 자동 제어
└─ 로그 기록
```

### 3.3 Cost Optimization

**문제**: 모든 프레임을 Cloud AI로 보내면 비용 폭발

**해결**:
```python
Intelligent Routing:
├─ Edge AI (무료):
│  ├─ 일반 모니터링 (95% 케이스)
│  └─ 간단한 탐지
│
├─ Cloud AI (유료 - 필요시만):
│  ├─ 이상 상황 분석 (3% 케이스)
│  ├─ 복잡한 추론 (1% 케이스)
│  └─ 주기적 품질 검사 (1% 케이스)
│
└─ Cost Estimate:
   ├─ Edge only: $0/month
   ├─ Edge + Cloud (smart): $50/month
   └─ vs. Cloud only: $5,000/month
   
   💰 비용 절감: 99%
```

---

## 🌐 4. GitHub Pages 자동화 전략

### 4.1 Repository 구조

```
vision-pro-platform/
├─ .github/
│  └─ workflows/
│     ├─ deploy.yml (자동 배포)
│     ├─ test.yml (자동 테스트)
│     └─ docs.yml (문서 자동 생성)
│
├─ docs/ (GitHub Pages)
│  ├─ index.html (랜딩 페이지)
│  ├─ demo/ (라이브 데모)
│  ├─ industries/ (산업별 소개)
│  │  ├─ manufacturing.html
│  │  ├─ healthcare.html
│  │  ├─ retail.html
│  │  └─ ...
│  └─ api-docs/ (API 문서)
│
├─ src/ (소스 코드)
│  ├─ core/ (핵심 엔진)
│  ├─ industries/ (산업별 모듈)
│  ├─ ai/ (AI 통합)
│  └─ web/ (웹 인터페이스)
│
├─ models/ (AI 모델)
│  ├─ yolov8n.pt
│  ├─ depth_anything_v3_small.pth
│  └─ industry_specific/
│
├─ examples/ (예제)
│  ├─ manufacturing_demo.py
│  ├─ retail_demo.py
│  └─ ...
│
└─ README.md
```

### 4.2 CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Build demo
        run: |
          python scripts/build_demo.py
      
      - name: Generate docs
        run: |
          python scripts/generate_docs.py
      
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs
```

### 4.3 Interactive Demo on GitHub Pages

```html
<!-- docs/demo/index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Vision Pro Platform - Live Demo</title>
    <style>
        /* Apple/Tesla 스타일 디자인 */
    </style>
</head>
<body>
    <div class="demo-container">
        <!-- 1. 비디오 업로드 or 웹캠 -->
        <div class="video-input">
            <input type="file" accept="video/*" id="videoUpload">
            <button onclick="startWebcam()">Use Webcam</button>
        </div>
        
        <!-- 2. 산업 선택 -->
        <div class="industry-selector">
            <select id="industry">
                <option value="manufacturing">Manufacturing</option>
                <option value="retail">Retail</option>
                <option value="security">Security</option>
                <option value="healthcare">Healthcare</option>
            </select>
        </div>
        
        <!-- 3. AI 모델 선택 -->
        <div class="ai-selector">
            <label><input type="checkbox" value="yolo" checked> YOLO Detection</label>
            <label><input type="checkbox" value="depth"> Depth Estimation</label>
            <label><input type="checkbox" value="segment"> Segmentation</label>
            <label><input type="checkbox" value="vlm"> AI Description (Gemini)</label>
        </div>
        
        <!-- 4. 결과 표시 -->
        <div class="results">
            <canvas id="outputCanvas"></canvas>
            <div id="aiInsights"></div>
        </div>
    </div>
    
    <script>
        // TensorFlow.js로 브라우저에서 직접 실행
        // YOLO 모델을 TFJS로 변환하여 배포
    </script>
</body>
</html>
```

---

## 🎨 5. 압도적인 차별화 요소

### 5.1 기술적 차별화

| 비교 항목 | 기존 CCTV | Vision Pro Platform |
|----------|-----------|---------------------|
| **AI 모델** | 단일 모델 | 멀티 AI 오케스트라 |
| **산업 특화** | 없음 | 8개 산업별 모듈 |
| **확장성** | 하드코딩 | 플러그인 아키텍처 |
| **언어** | 영어 | 한국어 최적화 |
| **배포** | 클라우드 전용 | Edge/Cloud 하이브리드 |
| **비용** | $5,000/월 | $50/월 (99% 절감) |
| **오픈소스** | ❌ | ✅ 코어 오픈소스 |
| **자동화** | 수동 설정 | GitHub Actions 자동 |

### 5.2 비즈니스 차별화

**경쟁사**:
1. **Amazon Rekognition**: 범용, 비쌈, 미국 중심
2. **Google Cloud Vision**: API만 제공, 커스터마이징 어려움
3. **Microsoft Azure Vision**: 엔터프라이즈 중심, 복잡
4. **Verkada**: 하드웨어 종속

**Vision Pro 우위**:
1. ✅ **산업 특화**: 제조/의료/리테일 등 맞춤 솔루션
2. ✅ **한국 시장**: 법규, 언어, 문화 최적화
3. ✅ **비용 효율**: Edge AI로 99% 비용 절감
4. ✅ **오픈소스**: 커뮤니티 기여, 투명성
5. ✅ **하이브리드**: 클라우드 + Edge 선택 가능
6. ✅ **Multi-AI**: 최고의 AI들을 상황별로 활용

### 5.3 기술 스택 비교

```
전통적인 CCTV AI:
OpenCV → 단일 detection 모델 → 알림

Vision Pro Platform:
카메라 입력
  ↓
[Pre-processing] (Edge)
  ├─ 해상도 최적화
  ├─ 프레임 선택
  └─ ROI 추출
  ↓
[Detection Ensemble] (Edge)
  ├─ YOLOv11 (범용)
  ├─ SAM 2 (세그멘테이션)
  └─ GroundingDINO (텍스트 기반)
  ↓
[Tracking & Memory] (Edge)
  ├─ Multi-object tracking
  ├─ Re-identification
  └─ Trajectory prediction
  ↓
[Event Analysis] (Edge→Cloud)
  IF 이상 감지:
    ├─ Gemini 2.0: "뭐가 일어났어?"
    ├─ GPT-4o: "위험해?"
    └─ Claude: "어떻게 대응?"
  ↓
[Action] (Edge)
  ├─ 알림 (SMS, 푸시, 이메일)
  ├─ 제어 (문 잠금, 조명 켜기)
  └─ 로그 (데이터베이스, 클라우드)
  ↓
[Analytics] (Cloud - 배치)
  ├─ 트렌드 분석
  ├─ 예측 모델링
  └─ 대시보드 업데이트
```

---

## 📈 6. Go-to-Market 전략

### 6.1 3단계 론칭 전략

**Phase 1: GitHub 오픈소스 (1개월)**
- 목표: 개발자 커뮤니티 확보
- 전략:
  - ✅ 코어 엔진 오픈소스 공개
  - ✅ GitHub Pages 데모 사이트
  - ✅ 상세한 문서 + 튜토리얼
  - ✅ Reddit, HN, 유튜브 홍보
- KPI:
  - GitHub Star: 1,000+
  - 기여자: 10+

**Phase 2: 산업별 파일럿 (3개월)**
- 목표: 실제 고객 확보
- 전략:
  - 🏭 제조업: KETI + 삼성전자
  - 🏪 리테일: GS25 무인점포
  - 🔒 보안: 아파트 관리사무소
- KPI:
  - 파일럿: 3개 산업 × 3개 고객 = 9곳
  - 매출: 각 500만원/월 = 4,500만원/월

**Phase 3: 스케일업 (6-12개월)**
- 목표: 시장 리더
- 전략:
  - 💰 시리즈A 투자 유치 (30억원)
  - 🏢 법인 설립
  - 👥 팀 확장 (개발 10명, 영업 5명)
  - 🌏 해외 진출 (일본, 동남아)
- KPI:
  - 고객: 100+ 기업
  - 매출: 5억원/월
  - ARR: 60억원

### 6.2 수익 모델

```
Revenue Streams:
├─ 1️⃣ SaaS Subscription (월 구독)
│  ├─ Starter: $99/월 (1 camera, basic features)
│  ├─ Professional: $299/월 (10 cameras, all features)
│  ├─ Enterprise: $999/월 (unlimited, custom)
│  └─ Industry Package: Custom pricing
│
├─ 2️⃣ Hardware Bundle
│  ├─ Edge AI Box (Jetson Orin): $2,000
│  └─ Camera + Edge Box: $3,000
│
├─ 3️⃣ Professional Services
│  ├─ Consulting: $200/hour
│  ├─ Custom Development: $10,000+
│  └─ Training: $5,000/day
│
└─ 4️⃣ API Usage (Pay-as-you-go)
   ├─ Cloud AI Calls: $0.01/request
   └─ Storage: $0.10/GB/month

예상 수익 (Phase 2 종료 시점):
- SaaS: 30 customers × $500/월 = $15,000/월
- Hardware: 10 units × $3,000 = $30,000 (one-time)
- Services: $20,000/월
────────────────────────────────────────
Total: $35,000/월 + $30K one-time
ARR: $420,000 (약 5억원)
```

---

## 🛠️ 7. 즉시 실행 계획 (Next 7 Days)

### Day 1-2: GitHub 리포지토리 설정
- [ ] 리포지토리 생성: `vision-pro-platform`
- [ ] README.md 작성 (압도적으로)
- [ ] 디렉토리 구조 설정
- [ ] GitHub Pages 활성화
- [ ] GitHub Actions CI/CD 설정

### Day 3-4: 코어 리팩토링
- [ ] Vision Pro v1.0 코드 모듈화
- [ ] 플러그인 아키텍처 구현
- [ ] 산업별 모듈 인터페이스 정의
- [ ] AI 통합 레이어 추가

### Day 5-6: 데모 사이트 구축
- [ ] 랜딩 페이지 (docs/index.html)
- [ ] 인터랙티브 데모 (TensorFlow.js)
- [ ] 산업별 소개 페이지
- [ ] API 문서 자동 생성

### Day 7: 론칭
- [ ] Reddit r/computervision 포스팅
- [ ] Hacker News 제출
- [ ] 유튜브 데모 영상 (10분)
- [ ] Twitter/X 홍보
- [ ] LinkedIn 공유

---

## 🎯 성공 지표

### 3개월 목표 (Phase 1 완료)
- ✅ GitHub Stars: 1,000+
- ✅ 문서 페이지뷰: 10,000+
- ✅ 데모 사용자: 500+
- ✅ 파일럿 고객: 3곳
- ✅ 미디어 노출: 5+ 기사

### 6개월 목표 (Phase 2 완료)
- ✅ 유료 고객: 30+
- ✅ MRR: $15,000 (월간 반복 수익)
- ✅ 팀 규모: 5명
- ✅ 투자 유치: 시드 10억원

### 12개월 목표 (Phase 3 완료)
- ✅ 고객: 100+
- ✅ ARR: $420K (5억원)
- ✅ 시리즈A: 30억원
- ✅ 팀: 15명
- ✅ 해외 진출: 일본, 싱가포르

---

**다음 단계**: GitHub 리포지토리 설정 시작!
