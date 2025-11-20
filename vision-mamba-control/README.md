# Vision Mamba Control - CCTV Monitoring System

**Depth Anything V3 기반 지능형 CCTV 모니터링 시스템**

## 🚀 빠른 시작

```bash
# 통합 서버 실행
python app.py
```

**브라우저에서 접속:**
- 메인: http://localhost:8080
- CCTV: http://localhost:8080/cctv

## ✨ 기능

- ✅ **실시간 사람 감지 & 추적** - YOLOv8n + Depth Anything V3
- ✅ **거리 측정** - 정확한 metric depth estimation
- ✅ **키(신장) 측정** - 깊이 + bbox로 실제 키 계산  
- ✅ **3D 위치 추적** - (x, y, z) 실시간 좌표
- ✅ **배회 감지** - 30초 이상 체류 시 경고
- ✅ **근접 경고** - 2m 이내 접근 알림
- ✅ **자동 데이터 로깅** - CSV & JSON 저장

## 📊 데이터 로깅

```
cctv_logs/
├── cctv_log_YYYYMMDD_HHMMSS.csv   # 실시간 스트리밍
└── cctv_log_YYYYMMDD_HHMMSS.json  # 배치 저장
```

**CSV 포맷:**
```csv
timestamp,person_id,bbox_x,bbox_y,bbox_w,bbox_h,distance_m,height_m,pos_x,pos_y,pos_z,confidence,is_loitering,is_close_alert
```

## 🧠 기술 스택

- **Depth Anything V3** (ByteDance) - 최신 깊이 추정
- **YOLOv8n** (Ultralytics) - 실시간 사람 감지
- **Flask** - 웹 서버
- **OpenCV** - 영상 처리

## 📦 설치

```bash
pip install torch torchvision opencv-python flask numpy ultralytics
pip install huggingface_hub safetensors omegaconf
```

## 📝 라이선스

MIT License

---
**현재 실행 중: http://localhost:8080/cctv** 🚀
