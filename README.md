# Auto-AI: Autonomous Company Platform

**완전 자동으로 굴러가는 회사** - AI 기반 전사 자동화 플랫폼

## 🎯 개요

Auto-AI는 회사의 **모든 레벨(전략/전술/운영)**을 자동화하여 사람의 최소 개입으로 자율 운영되는 완전 자동화 기업을 만드는 플랫폼입니다.

### 핵심 차별점

- ✅ **3계층 AI 시스템**: CEO AI → 부서 AI → 태스크 봇
- ✅ **완전한 회사 구조 정의**: 조직도, 부서, 역할, 프로세스 전체 모델링
- ✅ **의사결정 자동화**: 전략적/전술적/운영적 의사결정 자동화
- ✅ **워크플로우 자동화**: 승인, 태스크 할당, 프로세스 자동 실행
- ✅ **자가 치유 & 최적화**: 시스템이 스스로 문제를 감지하고 해결

### 핵심 기능

#### 1. 문서 자동화
- 보고서 자동 생성 (일일, 주간, 월간)
- 이메일 자동 작성 및 발송
- 계약서, 제안서, 회의록 자동 생성
- 문서 번역 및 요약

#### 2. 업무 플로우 자동화
- 프로젝트 관리 자동화
- 태스크 자동 할당 및 추적
- 승인 프로세스 자동화
- 마감일 관리 및 알림

#### 3. 데이터 분석 자동화
- 매출/실적 데이터 자동 분석
- 대시보드 자동 생성
- 트렌드 분석 및 예측
- 이상 탐지 및 알림

#### 4. 커뮤니케이션 자동화
- 이메일 자동 응답 및 분류
- Slack/Teams 채팅 자동화
- 회의 일정 자동 조율
- 회의록 자동 생성 및 배포

#### 5. 의사결정 지원
- AI 기반 의사결정 제안
- 리스크 분석 및 평가
- 시나리오 시뮬레이션
- 최적화 솔루션 제공

## 🏗️ 시스템 아키텍처

### 3계층 자동화 구조

```
┌─────────────────────────────────────────────┐
│         전략 레벨 (Strategic)                │
│    CEO AI - 전략적 의사결정 및 분석           │
│  • 시장 분석 • 전략 수립 • 시나리오 시뮬레이션  │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────┴───────────────────────────┐
│         전술 레벨 (Tactical)                 │
│   부서 AI - 부서별 운영 최적화                │
│ • 리소스 배분 • 프로젝트 관리 • 성과 분석      │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────┴───────────────────────────┐
│         운영 레벨 (Operational)              │
│    태스크 봇 - 일상 업무 자동화               │
│  • 문서 생성 • 이메일 처리 • 데이터 수집       │
└─────────────────────────────────────────────┘
```

### 디렉토리 구조

```
auto-ai/
├── src/
│   ├── core/                    # 핵심 엔진
│   │   ├── ai_engine.py         # Gemini AI 엔진
│   │   ├── scheduler.py         # 작업 스케줄러
│   │   └── workflow_engine.py   # 워크플로우 자동화
│   ├── models/                  # 데이터 모델
│   │   └── company.py           # 회사 구조 전체 모델
│   ├── agents/                  # AI 에이전트 계층
│   │   ├── ceo_agent.py         # CEO AI (전략)
│   │   ├── department_agent.py  # 부서 AI (전술)
│   │   └── task_bot.py          # 태스크 봇 (운영)
│   ├── automation/              # 자동화 모듈
│   ├── integrations/            # 외부 연동
│   └── api/                     # REST API
├── web/                         # 웹 대시보드
├── docs/                        # 문서
│   └── COMPANY_STRUCTURE.md     # 회사 구조 상세 정의
├── tests/                       # 테스트
└── config/                      # 설정
```

### 데이터 모델

완전한 회사 구조를 모델링:

- **조직 구조**: Company → Departments → Teams → Employees
- **업무 데이터**: Projects → Milestones → Tasks
- **재무 데이터**: Revenue, Expenses, Budget, Forecasts
- **성과 데이터**: KPIs, Performance Reviews
- **의사결정**: Decisions, Approvals, Workflows

## 기술 스택

- **AI/ML**: Google Gemini API
- **Backend**: Python 3.10+, FastAPI
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Database**: SQLite (개발), PostgreSQL (프로덕션)
- **Task Queue**: APScheduler
- **Deployment**: Docker, Docker Compose

## 설치 및 실행

### 요구사항
- Python 3.10 이상
- pip 또는 uv 패키지 매니저
- Git

### 설치

```bash
# 저장소 클론
git clone https://github.com/hwkim3330/auto-ai.git
cd auto-ai

# 의존성 설치
pip install -r requirements.txt

# 설정 파일 생성
cp config/config.example.yaml config/config.yaml
# config.yaml 파일을 열어 Gemini API 키 등 설정

# 실행
python src/main.py
```

### Docker 실행

```bash
docker-compose up -d
```

## 사용법

### 웹 대시보드
브라우저에서 `http://localhost:8000` 접속

### CLI 명령어

```bash
# 문서 자동 생성
python cli.py generate-report --type daily

# 워크플로우 실행
python cli.py run-workflow --name weekly-review

# 데이터 분석
python cli.py analyze-data --source sales.csv

# 이메일 자동 응답 시작
python cli.py start-email-bot
```

## 주요 워크플로우

### 1. 일일 업무 자동화
- 오전 9시: 이메일 확인 및 분류
- 오전 9:30: 일일 보고서 생성
- 오전 10시: 팀 회의 준비
- 오후 6시: 업무 진행 상황 요약

### 2. 주간 업무 자동화
- 월요일: 주간 계획 수립
- 수요일: 중간 점검 보고서
- 금요일: 주간 성과 리포트

### 3. 월간 업무 자동화
- 매월 1일: 월간 목표 설정
- 매월 15일: 중간 점검
- 매월 말: 월간 결산 및 분석

## 보안

- API 키는 환경 변수 또는 안전한 설정 파일에 저장
- 모든 통신은 HTTPS 사용
- 민감한 데이터는 암호화하여 저장
- 접근 권한 관리 및 감사 로그

## 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 라이선스

MIT License

## 연락처

- GitHub: [@hwkim3330](https://github.com/hwkim3330)
- Repository: [auto-ai](https://github.com/hwkim3330/auto-ai)

---

**🤖 Powered by AI - Auto-AI는 AI가 AI를 만들어 사람을 자유롭게 합니다**
