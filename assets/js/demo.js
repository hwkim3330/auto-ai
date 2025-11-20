// Auto-AI Demo JavaScript

// Scroll to demo section
function scrollToDemo() {
    document.getElementById('demo').scrollIntoView({ behavior: 'smooth' });
}

// Setup chat input on load
document.addEventListener('DOMContentLoaded', () => {
    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendDemoMessage();
            }
        });
    }
});

// Demo AI responses
const demoResponses = {
    "안녕": "안녕하세요! Auto-AI입니다. 어떤 업무를 자동화하고 싶으신가요?",
    "보고서": "보고서 자동 생성 기능을 사용하시려면 왼쪽의 '자동 보고서 생성' 패널을 이용해주세요. 일일, 주간, 월간 보고서를 자동으로 생성할 수 있습니다.",
    "이메일": "이메일 자동 작성 기능을 사용하시려면 '이메일 자동 작성' 패널에서 목적과 수신자를 입력해주세요. AI가 자동으로 적절한 내용을 생성합니다.",
    "도움": "Auto-AI는 다음 기능을 제공합니다:\n\n1. 📊 문서 자동화 (보고서, 이메일, 회의록)\n2. ⚙️ 업무 플로우 자동화\n3. 📈 데이터 분석\n4. 💬 커뮤니케이션 자동화\n5. 🤖 AI 의사결정 지원\n\n어떤 기능이 궁금하신가요?",
    "가격": "Auto-AI는 오픈소스 프로젝트입니다! GitHub에서 무료로 사용하실 수 있습니다. Gemini API 키만 있으면 바로 시작할 수 있어요.",
    "설치": "설치는 간단합니다:\n\n1. git clone https://github.com/hwkim3330/auto-ai.git\n2. pip install -r requirements.txt\n3. config/config.yaml에 Gemini API 키 설정\n4. python src/main.py 실행\n\n자세한 내용은 README를 참고해주세요!",
    "default": "흥미로운 질문이네요! 실제 AI 기능을 사용하시려면 Auto-AI를 설치하여 Gemini API와 연동해주세요. GitHub에서 자세한 정보를 확인하실 수 있습니다."
};

// Send demo message
function sendDemoMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();

    if (!message) return;

    // Add user message
    addChatMessage(message, 'user');
    input.value = '';

    // Simulate AI thinking
    setTimeout(() => {
        const response = getAIResponse(message);
        addChatMessage(response, 'ai');
    }, 800);
}

// Get AI response (demo)
function getAIResponse(message) {
    const lowerMessage = message.toLowerCase();

    for (const [keyword, response] of Object.entries(demoResponses)) {
        if (keyword !== 'default' && lowerMessage.includes(keyword)) {
            return response;
        }
    }

    return demoResponses.default;
}

// Add message to chat
function addChatMessage(text, sender) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}`;
    messageDiv.textContent = text;
    chatMessages.appendChild(messageDiv);

    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Generate demo report
function generateDemoReport() {
    const sales = document.getElementById('sales').value;
    const customers = document.getElementById('customers').value;
    const tasks = document.getElementById('tasks').value;

    const today = new Date().toLocaleDateString('ko-KR', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });

    const report = `# 📊 일일 업무 보고서
${today}

## 요약 (Executive Summary)
오늘은 전반적으로 양호한 성과를 거두었습니다. 매출 목표 달성률은 105%를 기록했으며, 신규 고객 확보에서도 우수한 실적을 보였습니다.

## 주요 성과
✅ 매출액: ${parseInt(sales).toLocaleString()}원 (목표 대비 +5%)
✅ 신규 고객: ${customers}명 (전주 대비 +12%)
✅ 완료된 작업: ${tasks}건 (목표: 20건)
✅ 고객 만족도: 4.7/5.0

## 진행 중인 작업
🔄 Q1 신제품 출시 준비 (진행률: 75%)
🔄 마케팅 캠페인 기획 (진행률: 60%)
🔄 고객 데이터 분석 시스템 구축 (진행률: 85%)

## 이슈 및 위험 사항
⚠️ 공급망 지연으로 인한 재고 부족 우려 (대응 중)
⚠️ 경쟁사 신제품 출시 예정 (모니터링 필요)

## 다음 단계
1. 재고 확보를 위한 공급업체와 긴급 미팅
2. 경쟁 분석 및 대응 전략 수립
3. 신제품 출시 일정 최종 검토
4. 고객 피드백 분석 및 반영

---
🤖 Auto-AI가 자동 생성한 보고서입니다.`;

    const output = document.getElementById('report-output');
    output.textContent = report;
    output.classList.add('visible');

    // Scroll to output
    setTimeout(() => {
        output.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

// Generate demo email
function generateDemoEmail() {
    const purpose = document.getElementById('email-purpose').value;
    const recipient = document.getElementById('recipient').value;

    const emailTemplates = {
        "프로젝트 제안": {
            subject: `[제안] 신규 프로젝트 협업 제안의 건`,
            body: `${recipient}님, 안녕하세요.

항상 좋은 협업을 해주셔서 감사드립니다.

이번에 새로운 프로젝트를 기획하게 되어 협업을 제안드리고자 합니다.

[프로젝트 개요]
- 목적: 업무 자동화 시스템 구축
- 기간: 3개월 (2025.02 ~ 2025.04)
- 범위: AI 기반 문서 자동화 및 워크플로우 개선

다음 주 화요일(14:00)에 간단한 미팅을 통해 자세한 내용을 공유드리고 싶습니다.
편하신 시간 알려주시면 감사하겠습니다.

감사합니다.

---
🤖 Auto-AI가 자동 생성한 이메일입니다.`
        },
        "회의 요청": {
            subject: `[회의 요청] ${new Date().toLocaleDateString('ko-KR', { month: 'long', day: 'numeric' })} 미팅 일정 조율`,
            body: `${recipient}님, 안녕하세요.

주간 업무 진행 상황을 공유하고 다음 단계를 논의하기 위해 미팅을 요청드립니다.

[회의 안건]
1. 프로젝트 진행 상황 리뷰
2. 현안 이슈 논의 및 해결 방안
3. 다음 주 주요 작업 계획
4. 리소스 할당 검토

[제안 일정]
- 1순위: ${getDayAfterTomorrow()} 14:00~15:00
- 2순위: ${getDayAfterTomorrow()} 16:00~17:00

편하신 시간 회신 부탁드립니다.

감사합니다.

---
🤖 Auto-AI가 자동 생성한 이메일입니다.`
        },
        "진행 상황 보고": {
            subject: `[보고] ${new Date().toLocaleDateString('ko-KR', { month: 'long', day: 'numeric' })} 프로젝트 진행 상황`,
            body: `${recipient}님, 안녕하세요.

금주 프로젝트 진행 상황을 보고드립니다.

[주요 성과]
✅ 1단계 개발 완료 (100%)
✅ 사용자 테스트 진행 중 (75%)
✅ 문서화 작업 진행 중 (60%)

[현황]
- 전체 진행률: 82%
- 일정: 정상 진행 중
- 이슈: 경미한 버그 3건 (수정 중)

[다음 주 계획]
1. 사용자 테스트 완료
2. 발견된 버그 수정
3. 최종 검수 준비

상세한 내용은 첨부된 보고서를 참고해 주시기 바랍니다.

감사합니다.

---
🤖 Auto-AI가 자동 생성한 이메일입니다.`
        },
        "감사 인사": {
            subject: `감사의 말씀 전합니다`,
            body: `${recipient}님, 안녕하세요.

이번 프로젝트에서 보여주신 헌신적인 노력과 전문성에 깊은 감사를 표합니다.

${recipient}님의 적극적인 참여와 창의적인 아이디어 덕분에 목표를 성공적으로 달성할 수 있었습니다.

앞으로도 계속해서 좋은 협업을 이어가길 기대합니다.

다시 한번 감사드립니다.

---
🤖 Auto-AI가 자동 생성한 이메일입니다.`
        }
    };

    const email = emailTemplates[purpose] || emailTemplates["프로젝트 제안"];

    const emailHTML = `<strong>제목:</strong> ${email.subject}

<strong>본문:</strong>
${email.body}`;

    const output = document.getElementById('email-output');
    output.innerHTML = emailHTML;
    output.classList.add('visible');

    // Scroll to output
    setTimeout(() => {
        output.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

// Helper: Get day after tomorrow formatted
function getDayAfterTomorrow() {
    const date = new Date();
    date.setDate(date.getDate() + 2);
    return date.toLocaleDateString('ko-KR', {
        month: 'long',
        day: 'numeric',
        weekday: 'long'
    });
}

// Add some interactive effects
document.addEventListener('DOMContentLoaded', () => {
    // Animate feature cards on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    document.querySelectorAll('.feature-card, .demo-panel, .tech-item').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'all 0.6s ease-out';
        observer.observe(el);
    });
});

// Add welcome message to chat
setTimeout(() => {
    const welcomeMessages = [
        "💼 업무 자동화에 대해 궁금하신 점이 있으신가요?",
        "📊 보고서, 이메일 등을 자동으로 생성할 수 있습니다!",
        "'도움'이라고 입력하시면 사용 가능한 기능을 안내해드립니다."
    ];

    welcomeMessages.forEach((msg, index) => {
        setTimeout(() => {
            addChatMessage(msg, 'ai');
        }, (index + 1) * 1500);
    });
}, 1000);

console.log('🤖 Auto-AI Demo Ready!');
console.log('GitHub: https://github.com/hwkim3330/auto-ai');
