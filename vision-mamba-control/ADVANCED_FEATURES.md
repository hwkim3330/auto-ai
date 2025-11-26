# Advanced Features - Vision Pro v1.3

## 🚀 새로운 고급 기능

Vision Pro에 스크린샷, 녹화, 알림 시스템이 추가되었습니다!

---

## ✅ 구현된 기능

### 1. 📸 스크린샷 캡처

#### 기능
- **즉시 캡처**: 현재 비전 피드를 PNG로 저장
- **자동 파일명**: `vision-pro-YYYY-MM-DDTHH-mm-ss.png`
- **고해상도**: 원본 해상도 유지 (640x480 또는 더 높음)
- **시각적 피드백**: 버튼 애니메이션
- **다운로드 자동 시작**: 클릭 즉시 다운로드

#### 사용법
1. Vision System 활성화
2. 오른쪽 하단의 📸 버튼 클릭
3. 브라우저 다운로드 폴더에 자동 저장

#### 기술 구현
```javascript
function takeScreenshot() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
        // Download as PNG
    }, 'image/png');
}
```

**Location**: monitor.html:1193-1231

---

### 2. ⏺ 비디오 녹화

#### 기능
- **실시간 녹화**: 비전 피드를 WebM 형식으로 녹화
- **고품질**: 2.5 Mbps 비트레이트, VP9 코덱
- **토글 버튼**: 시작/중지 한 번에
- **녹화 상태 표시**: 빨간 버튼 애니메이션 (pulse)
- **자동 저장**: 녹화 중지 시 즉시 다운로드

#### 사용법
1. Vision System 활성화
2. 오른쪽 하단의 ⏺ 버튼 클릭 (녹화 시작)
3. 다시 클릭하여 녹화 중지
4. 브라우저 다운로드 폴더에 WebM 파일 저장

#### 녹화 파일
- **형식**: WebM (video/webm)
- **코덱**: VP9
- **프레임레이트**: 30 FPS
- **비트레이트**: 2.5 Mbps
- **파일명**: `vision-pro-recording-YYYY-MM-DDTHH-mm-ss.webm`

#### 기술 구현
```javascript
const stream = videoFeed.captureStream(30);
const mediaRecorder = new MediaRecorder(stream, {
    mimeType: 'video/webm;codecs=vp9',
    videoBitsPerSecond: 2500000
});

mediaRecorder.ondataavailable = (event) => {
    recordedChunks.push(event.data);
};

mediaRecorder.onstop = () => {
    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    // Download video
};
```

**Location**: monitor.html:1233-1297

---

### 3. 🔔 웹 알림 시스템

#### 기능
- **브라우저 알림**: 시스템 트레이 알림
- **알림 종류**:
  - 📸 스크린샷 저장됨
  - ⏺ 녹화 시작/종료
  - ⚠️ Warning 알림 (loitering 등)
  - 🚨 Alert 알림 (proximity 등)
- **권한 관리**: 사용자 승인 필요
- **토글 가능**: 언제든 활성화/비활성화

#### 사용법
1. 오른쪽 하단의 🔔 버튼 클릭
2. 브라우저에서 알림 권한 허용
3. 버튼이 녹색으로 변경되면 활성화 완료
4. 알림 발생 시 시스템 트레이에 표시

#### 알림 트리거
- **스크린샷 저장**: 즉시 알림
- **녹화 시작/종료**: 즉시 알림
- **감시 알림**: loitering, proximity alert 발생 시

#### 기술 구현
```javascript
if ('Notification' in window) {
    const permission = await Notification.requestPermission();
    if (permission === 'granted') {
        new Notification(title, {
            body: message,
            icon: '/static/icon.png',
            badge: '/static/badge.png'
        });
    }
}
```

**Location**: monitor.html:1299-1336

---

## 🎨 UI 디자인

### Action Buttons (오른쪽 하단)
```
🔔 알림        (Orange → Green)
⏺ 녹화        (Red, pulse animation)
📸 스크린샷     (Green)
⚙ 설정        (Blue)
```

### 버튼 상태
- **Normal**: 일반 상태
- **Hover**: 크기 확대 (1.05x)
- **Active**: 크기 축소 (0.95x)
- **Recording**: 빨간색 pulse 애니메이션
- **Notification Enabled**: 녹색 배경

### CSS 구현
```css
.action-btn {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
}

.record-btn.recording {
    animation: pulse 1.5s infinite;
}
```

**Location**: monitor.html:459-533

---

## 📊 브라우저 호환성

### Screenshot
- ✅ Chrome/Edge: Full support
- ✅ Firefox: Full support
- ✅ Safari: Full support

### Recording
- ✅ Chrome/Edge: VP9 support
- ⚠️ Firefox: VP8 fallback
- ⚠️ Safari: H.264 fallback (requires MIME type change)

### Web Notifications
- ✅ Chrome/Edge: Full support
- ✅ Firefox: Full support
- ✅ Safari 16+: Full support

---

## 🔧 문제 해결

### 스크린샷이 다운로드되지 않음
- Vision System이 활성화되어 있는지 확인
- 비디오 피드가 로드되었는지 확인
- 브라우저 팝업 차단 확인

### 녹화가 시작되지 않음
- Vision System이 활성화되어 있는지 확인
- 브라우저가 MediaRecorder API를 지원하는지 확인
- 콘솔에서 에러 메시지 확인

### 알림이 표시되지 않음
- 브라우저 알림 권한이 허용되었는지 확인
- 시스템 알림 설정 확인 (Windows/Mac)
- 브라우저 집중 모드(DND) 확인

---

## 📂 파일 저장 위치

### 기본 저장 위치
- **Windows**: `C:\Users\[Username]\Downloads\`
- **macOS**: `/Users/[Username]/Downloads/`
- **Linux**: `~/Downloads/`

### 파일명 형식
```
vision-pro-2025-11-20T16-51-49.png          # 스크린샷
vision-pro-recording-2025-11-20T16-51-49.webm  # 녹화
```

---

## 🚀 향후 계획

### Phase 3.1: 고급 녹화 기능
- [ ] 타임랩스 모드 (10x 속도)
- [ ] BEV 피드 녹화 옵션
- [ ] 양쪽 피드 동시 녹화 (side-by-side)
- [ ] 녹화 품질 설정 (저/중/고)
- [ ] MP4 형식 지원 (FFmpeg)

### Phase 3.2: 스크린샷 고급 기능
- [ ] 전체 화면 캡처 (비전 피드 + BEV + 통계)
- [ ] 자동 스크린샷 (이벤트 발생 시)
- [ ] 스크린샷 히스토리
- [ ] 주석 추가 기능

### Phase 3.3: 알림 고급 기능
- [ ] 알림 음성 (Text-to-Speech)
- [ ] 커스텀 알림 음성
- [ ] 이메일 알림
- [ ] Webhook 통합 (Slack, Discord, Teams)
- [ ] SMS 알림 (Twilio)

### Phase 3.4: ROI (Region of Interest)
- [ ] 마우스로 영역 그리기
- [ ] 다중 ROI 설정
- [ ] ROI별 알림 임계값
- [ ] ROI 분석 통계

---

## 💡 사용 시나리오

### 시나리오 1: 보안 감시
1. Vision System 활성화
2. 알림 활성화 (🔔)
3. 녹화 시작 (⏺)
4. 이상 행동 발생 시 알림 수신
5. 스크린샷 캡처 (📸)
6. 녹화 중지 및 증거 저장

### 시나리오 2: 성능 테스트
1. Vision System 활성화
2. Chart 모니터링
3. 특정 시점에 스크린샷 캡처
4. 성능 데이터 분석

### 시나리오 3: 프레젠테이션
1. Vision System 활성화
2. 녹화 시작
3. 데모 진행
4. 녹화 중지
5. 비디오를 프레젠테이션 자료로 사용

---

## 🧪 테스트 체크리스트

- [x] 스크린샷 버튼 클릭 시 PNG 다운로드
- [x] 녹화 버튼 클릭 시 녹화 시작/중지
- [x] 녹화 중 버튼 애니메이션 표시
- [x] 녹화 중지 시 WebM 다운로드
- [x] 알림 버튼 클릭 시 권한 요청
- [x] 알림 활성화 시 버튼 녹색 변경
- [x] 알림 발생 시 시스템 알림 표시
- [x] 스크린샷/녹화 시 알림 표시
- [x] 모든 버튼 hover 효과 작동
- [x] 모바일 반응형 확인

---

## 📝 코드 변경 사항

### monitor.html 수정 사항
1. **CSS 추가** (lines 459-533): Action buttons 스타일
2. **HTML 추가** (lines 873-879): 4개 액션 버튼
3. **JavaScript 추가** (lines 1193-1336):
   - `takeScreenshot()`: 스크린샷 기능
   - `toggleRecording()`: 녹화 기능
   - `toggleNotifications()`: 알림 토글
   - `showNotification()`: 알림 표시
4. **알림 통합** (lines 1368-1372): 알림 발생 시 자동 알림

### 총 추가 코드
- **CSS**: ~75 lines
- **HTML**: ~7 lines
- **JavaScript**: ~150 lines
- **Total**: ~232 lines

---

**Date**: 2025-11-20
**Version**: v1.3
**Status**: ✅ Complete - Production Ready
**Server**: http://localhost:8080

## 🎉 요약

Vision Pro v1.3에서는 다음 고급 기능이 추가되었습니다:

✅ **스크린샷 캡처**: 즉시 PNG로 저장
✅ **비디오 녹화**: WebM 형식으로 고품질 녹화
✅ **웹 알림**: 브라우저 시스템 알림

모든 기능은 직관적인 버튼 인터페이스로 접근 가능하며, 시각적 피드백과 함께 작동합니다!
