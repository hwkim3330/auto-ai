# Deployment Guide - Vision Pro

이 가이드는 Vision Pro를 안전하게 배포하는 방법을 설명합니다.

---

## 🔐 비밀 키 관리 (중요!)

### ⚠️ 절대 Git에 올리지 말 것

다음 파일들은 **절대 Git에 커밋하면 안 됩니다**:
- `.env` - 환경 변수 파일
- API 키 (OpenAI, Google, Anthropic 등)
- 데이터베이스 비밀번호
- SSH 키, SSL 인증서
- `credentials.json`, `secrets.yaml` 등

### ✅ 안전한 방법

#### 1. .env 파일 사용

```bash
# 1. .env.example을 복사
cp .env.example .env

# 2. .env 파일 편집 (실제 키 입력)
nano .env

# 3. .gitignore 확인 (.env가 포함되어 있는지)
cat .gitignore | grep .env
```

**.env 파일 예시**:
```bash
# AI API Keys
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxx
GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxx

# Server
FLASK_SECRET_KEY=your_random_32_character_string_here
PORT=8080
```

#### 2. 환경 변수로 로드

**app.py 수정**:
```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'default-dev-key')
```

**필요한 패키지**:
```bash
pip install python-dotenv
```

---

## 🚀 로컬 배포

### 기본 실행

```bash
# 1. 가상 환경 활성화
source venv/bin/activate

# 2. 서버 실행
python app.py

# 3. 브라우저에서 접속
# http://localhost:8080
```

### systemd 서비스 등록 (자동 시작)

```bash
# 1. 서비스 파일 생성
sudo nano /etc/systemd/system/vision-pro.service
```

**vision-pro.service**:
```ini
[Unit]
Description=Vision Pro AI Surveillance System
After=network.target

[Service]
Type=simple
User=kim
WorkingDirectory=/home/kim/auto-ai/vision-mamba-control
Environment="PATH=/home/kim/auto-ai/vision-mamba-control/venv/bin"
ExecStart=/home/kim/auto-ai/vision-mamba-control/venv/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# 2. 서비스 활성화
sudo systemctl daemon-reload
sudo systemctl enable vision-pro.service
sudo systemctl start vision-pro.service

# 3. 상태 확인
sudo systemctl status vision-pro.service

# 4. 로그 확인
sudo journalctl -u vision-pro.service -f
```

---

## 🌐 외부 접속 (공개 배포)

### 옵션 1: Ngrok (간단, 테스트용)

```bash
# 1. Ngrok 설치
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | \
  sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && \
  echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | \
  sudo tee /etc/apt/sources.list.d/ngrok.list && \
  sudo apt update && sudo apt install ngrok

# 2. Ngrok 인증
ngrok config add-authtoken YOUR_AUTHTOKEN

# 3. 터널 시작
ngrok http 8080

# 4. 제공된 URL로 접속
# https://xxxx-xx-xxx-xxx-xx.ngrok-free.app
```

**장점**: 설정 간단, 즉시 사용
**단점**: 무료 플랜 제약, 보안 제한적

### 옵션 2: 클라우드 서버 (프로덕션)

#### AWS EC2

```bash
# 1. EC2 인스턴스 생성
# - Ubuntu 22.04 LTS
# - t3.medium (CPU) 또는 g4dn.xlarge (GPU)
# - 보안 그룹: 8080 포트 열기

# 2. 서버 접속
ssh -i your-key.pem ubuntu@your-ec2-ip

# 3. 프로젝트 클론
git clone https://github.com/yourusername/vision-pro-platform.git
cd vision-pro-platform

# 4. 설치
./install.sh

# 5. .env 설정
cp .env.example .env
nano .env  # API 키 입력

# 6. 실행
python app.py
```

#### Nginx 리버스 프록시 설정

```bash
# 1. Nginx 설치
sudo apt install nginx

# 2. 설정 파일 생성
sudo nano /etc/nginx/sites-available/vision-pro
```

**/etc/nginx/sites-available/vision-pro**:
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

```bash
# 3. 심볼릭 링크 생성
sudo ln -s /etc/nginx/sites-available/vision-pro /etc/nginx/sites-enabled/

# 4. Nginx 재시작
sudo nginx -t
sudo systemctl restart nginx
```

#### SSL 인증서 (HTTPS)

```bash
# 1. Certbot 설치
sudo apt install certbot python3-certbot-nginx

# 2. SSL 인증서 발급
sudo certbot --nginx -d yourdomain.com

# 3. 자동 갱신 확인
sudo certbot renew --dry-run
```

---

## 🐳 Docker 배포

```bash
# 1. Dockerfile 생성
nano Dockerfile
```

**Dockerfile**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8080

# Run
CMD ["python", "app.py"]
```

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  vision-pro:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./logs:/app/logs
      - ./config.yaml:/app/config.yaml:ro
    environment:
      - FLASK_ENV=production
    env_file:
      - .env
    restart: unless-stopped
```

```bash
# 빌드 및 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 중지
docker-compose down
```

---

## 🔒 보안 체크리스트

### 배포 전 확인사항

- [ ] `.env` 파일이 `.gitignore`에 포함되어 있는지 확인
- [ ] 모든 API 키가 `.env` 파일에 저장되어 있는지 확인
- [ ] `FLASK_SECRET_KEY`를 랜덤 문자열로 변경
- [ ] `FLASK_ENV=production` 설정
- [ ] HTTPS 인증서 설정 (프로덕션)
- [ ] 방화벽 규칙 확인 (필요한 포트만 열기)
- [ ] CORS 설정 확인
- [ ] Rate limiting 고려
- [ ] 로그 파일 권한 확인
- [ ] 데이터베이스 백업 설정 (향후)

### 권장 보안 설정

**app.py 수정**:
```python
from flask import Flask
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')

# CORS 설정 (필요한 도메인만 허용)
CORS(app, resources={r"/api/*": {"origins": ["https://yourdomain.com"]}})

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)
```

---

## 📊 모니터링

### 서버 상태 확인

```bash
# CPU/메모리 사용량
htop

# 디스크 사용량
df -h

# 네트워크 사용량
vnstat

# 프로세스 확인
ps aux | grep python
```

### 로그 확인

```bash
# Vision Pro 로그
tail -f logs/vision-pro.log

# Systemd 로그
sudo journalctl -u vision-pro.service -f

# Nginx 로그
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

---

## 🛠️ 문제 해결

### 포트가 이미 사용 중

```bash
# 8080 포트 사용 프로세스 찾기
sudo lsof -i :8080

# 프로세스 종료
sudo kill -9 <PID>
```

### 웹캠 접근 권한 오류

```bash
# 사용자를 video 그룹에 추가
sudo usermod -a -G video $USER

# 재로그인 필요
```

### 메모리 부족

```bash
# 스왑 메모리 추가
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## 📈 성능 최적화

### 프로덕션 설정

**config.yaml**:
```yaml
performance:
  target_fps: 30
  max_consecutive_errors: 10
  error_sleep_time: 0.1

vision:
  detection_interval: 5  # 프레임마다 검출 (높은 정확도)
  depth_interval: 50     # 50프레임마다 깊이 추정

logging:
  enabled: true
  buffer_size: 100  # 100개 엔트리마다 저장
```

### GPU 가속

```bash
# CUDA 설치 확인
nvidia-smi

# config.yaml에서 GPU 활성화
vision:
  device: 'cuda'  # CPU에서 GPU로 변경
```

**예상 성능**:
- CPU: 25-30 FPS
- GPU (NVIDIA GTX 1060+): 60+ FPS
- Jetson Orin Nano: 30 FPS

---

## 📞 지원

문제가 발생하면:
1. GitHub Issues: https://github.com/yourusername/vision-pro-platform/issues
2. Discord: https://discord.gg/visionpro
3. Email: contact@visionpro.ai

---

**업데이트**: 2025-11-20
**버전**: v1.3
**상태**: Production Ready
