# 📰 Auto Daily Report

**매일 아침 자동으로 생성되는 AI 뉴스 & 기상 리포트**

뉴스, 기상정보, 환율을 AI로 요약하여 매일 아침 7시에 자동으로 배송합니다!

---

## 🎯 Features

### 📰 뉴스 수집 & 요약
- **소스**: NewsAPI, Google News RSS
- **카테고리**: 정치, 경제, IT, 세계
- **AI 요약**: GPT-4 / Claude로 핵심만 추출
- **개인화**: 관심 키워드 자동 필터링

### 🌤️ 기상 정보
- **상세 날씨**: 오늘/내일/주간 예보
- **알림**: 비/눈 예상 시 자동 알림
- **미세먼지**: AQI 지수 포함
- **옷차림 추천**: 기온별 자동 추천

### 💰 금융 정보 (선택)
- **환율**: USD, JPY, CNY
- **주요 지수**: 코스피, 나스닥
- **가상화폐**: 비트코인, 이더리움

### 🎙️ 음성 리포트
- **TTS 생성**: Microsoft Edge-TTS
- **자연스러운 음성**: 한국어 자연 음성
- **파일 저장**: MP3 다운로드 가능

---

## 🚀 Quick Start

### 1. 설치

```bash
# Clone repository
git clone https://github.com/yourusername/auto-daily-report.git
cd auto-daily-report

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API 키 설정

```bash
# Copy .env.example
cp .env.example .env

# Edit .env file
nano .env
```

```.env
# Required
OPENAI_API_KEY=sk-...           # OpenAI API (GPT-4)
NEWS_API_KEY=your_key            # NewsAPI.org
OPENWEATHER_API_KEY=your_key     # OpenWeatherMap

# Optional
TELEGRAM_BOT_TOKEN=your_token    # Telegram 발송
TELEGRAM_CHAT_ID=your_id
EMAIL_USER=your@email.com
EMAIL_PASSWORD=your_password
```

### 3. 실행

```bash
# 수동 실행 (즉시 리포트 생성)
python main.py

# 결과 확인
cat reports/daily/report_2025-11-20.md
```

---

## ⏰ 자동화 (GitHub Actions)

### GitHub Actions로 매일 자동 실행

**설정된 스케줄**: 매일 오전 7시 (KST)

```yaml
# .github/workflows/daily-report.yml
on:
  schedule:
    - cron: '0 22 * * *'  # UTC 22:00 = KST 07:00
  workflow_dispatch:      # 수동 실행 가능
```

**설정 방법**:
1. GitHub Repository → Settings → Secrets and variables → Actions
2. 환경 변수 추가:
   - `OPENAI_API_KEY`
   - `NEWS_API_KEY`
   - `OPENWEATHER_API_KEY`
   - `TELEGRAM_BOT_TOKEN` (선택)
   - `TELEGRAM_CHAT_ID` (선택)

3. Actions 탭에서 자동 실행 확인

---

## 📊 출력 형식

### 1. Markdown Report

```markdown
# 📰 Daily Report - 2025년 11월 20일 수요일

## 🌤️ 날씨
- 서울: 맑음, 12°C (체감 10°C)
- 강수 확률: 10%
- 미세먼지: 좋음 (25 µg/m³)
- **옷차림**: 가을 자켓 추천

## 📰 주요 뉴스

### 정치
- [요약] ...

### 경제
- [요약] ...

### IT & 과학
- [요약] ...

## 💰 금융
- 환율: $1 = ₩1,320
- 코스피: 2,450 (+1.2%)
```

### 2. 음성 파일

`reports/daily/report_2025-11-20.mp3`

### 3. HTML 웹페이지

`reports/daily/report_2025-11-20.html` (GitHub Pages 배포)

---

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌──────────────┐
│   Collectors │─────▶│  Processors  │─────▶│  Generators  │
│  (API 수집)  │      │  (AI 요약)    │      │ (텍스트/음성) │
└─────────────┘      └──────────────┘      └──────────────┘
      │                                              │
      │                                              ▼
      │                                     ┌──────────────┐
      │                                     │  Publishers  │
      │                                     │ (배포/알림)   │
      │                                     └──────────────┘
      │                                              │
      ▼                                              ▼
┌─────────────┐                              ┌─────────────┐
│  External   │                              │   Users     │
│    APIs     │                              │ (Email/TG)  │
└─────────────┘                              └─────────────┘
```

---

## 📂 Project Structure

```
auto-daily-report/
│
├── src/
│   ├── collectors/              # 데이터 수집
│   │   ├── news_collector.py    # 뉴스 수집 (NewsAPI)
│   │   ├── weather_collector.py # 기상 수집 (OpenWeather)
│   │   └── finance_collector.py # 금융 수집 (선택)
│   │
│   ├── processors/              # 데이터 처리
│   │   ├── summarizer.py        # AI 요약 (GPT-4)
│   │   └── formatter.py         # 포맷팅
│   │
│   ├── generators/              # 콘텐츠 생성
│   │   ├── text_generator.py   # Markdown/HTML 생성
│   │   └── audio_generator.py  # 음성 생성 (TTS)
│   │
│   └── publishers/              # 배포
│       ├── telegram_publisher.py
│       ├── email_publisher.py
│       └── github_publisher.py
│
├── reports/                     # 생성된 리포트
│   ├── daily/                   # 매일 리포트
│   └── archive/                 # 아카이브
│
├── .github/
│   └── workflows/
│       └── daily-report.yml     # GitHub Actions
│
├── config/
│   └── config.yaml              # 설정 파일
│
├── main.py                      # 메인 실행 파일
├── requirements.txt             # Python 의존성
├── .env.example                 # 환경 변수 예시
└── README.md                    # 현재 문서
```

---

## 🛠️ API & Services

### 필수 API

1. **NewsAPI** (무료)
   - URL: https://newsapi.org/
   - 무료 플랜: 100 requests/day
   - 등록: 이메일만 필요

2. **OpenWeatherMap** (무료)
   - URL: https://openweathermap.org/api
   - 무료 플랜: 1,000 calls/day
   - 등록: 무료

3. **OpenAI API** (유료)
   - URL: https://platform.openai.com/
   - GPT-4 사용 (또는 GPT-3.5)
   - 비용: ~$0.01/요약

### 선택 API

4. **Telegram Bot** (무료)
   - BotFather로 봇 생성
   - 푸시 알림용

5. **Exchange Rate API** (무료)
   - URL: https://exchangerate-api.com/
   - 1,500 requests/month

---

## ⚙️ Configuration

### config.yaml

```yaml
report:
  language: ko
  timezone: Asia/Seoul
  categories:
    - politics
    - economy
    - technology
    - world

news:
  sources:
    - bbc-news
    - techcrunch
    - the-verge
  keywords:
    - AI
    - 우주
    - 혁신
  max_articles: 10

weather:
  city: Seoul
  units: metric
  forecast_days: 3

ai:
  model: gpt-4
  temperature: 0.3
  max_tokens: 500

publishing:
  telegram: true
  email: false
  github_pages: true

tts:
  enabled: true
  voice: ko-KR-SunHiNeural
  rate: +0%
```

---

## 🎨 Customization

### 뉴스 키워드 필터링

`config/config.yaml` 수정:

```yaml
news:
  keywords:
    - "인공지능"
    - "우주"
    - "양자컴퓨팅"
  exclude:
    - "연예"
    - "스포츠"
```

### AI 모델 변경

```python
# src/processors/summarizer.py
# GPT-4 → GPT-3.5 (저렴)
model = "gpt-3.5-turbo"

# OpenAI → Claude
from anthropic import Anthropic
client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
```

---

## 💡 Use Cases

### 1. 개인 사용
- 매일 아침 이메일/텔레그램 수신
- 출근길 음성으로 청취

### 2. 팀/회사
- 팀 채널에 자동 공유
- 주요 뉴스 브리핑

### 3. 블로그/유튜브
- 자동 콘텐츠 생성
- 소스로 활용

---

## 📈 Roadmap

### v1.0 (Current)
- [x] 뉴스/기상 수집
- [x] AI 요약
- [x] Markdown 리포트
- [x] GitHub Actions 자동화

### v1.1 (Next)
- [ ] 음성 리포트 (TTS)
- [ ] HTML 웹페이지
- [ ] Telegram 발송

### v1.2
- [ ] 이메일 발송
- [ ] GitHub Pages 자동 배포
- [ ] 모바일 앱 (React Native)

### v2.0
- [ ] 개인화 AI
- [ ] 사용자 피드백 학습
- [ ] 멀티 언어 지원
- [ ] 비디오 리포트

---

## 🤝 Contributing

Pull requests are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License - 자유롭게 사용하세요!

---

## 🙏 Acknowledgments

- NewsAPI for news data
- OpenWeatherMap for weather data
- OpenAI for GPT-4
- Microsoft Edge-TTS for voice synthesis

---

**Made with ❤️ by Auto-AI Team**

[GitHub](https://github.com/yourusername/auto-daily-report) · [Issues](https://github.com/yourusername/auto-daily-report/issues)

⭐ Star us on GitHub if you find this useful!
