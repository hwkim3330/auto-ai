// Auto-AI Frontend Application

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    checkSystemHealth();
    loadTasks();
    setupChatInput();
});

// Check system health
async function checkSystemHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();

        // Update AI Engine status
        const aiStatus = document.getElementById('ai-status');
        if (data.services.ai_engine) {
            aiStatus.textContent = '온라인';
            aiStatus.className = 'status-value online';
        } else {
            aiStatus.textContent = '오프라인';
            aiStatus.className = 'status-value offline';
        }

        // Update Scheduler status
        const schedulerStatus = document.getElementById('scheduler-status');
        if (data.services.scheduler) {
            schedulerStatus.textContent = '실행 중';
            schedulerStatus.className = 'status-value online';
        } else {
            schedulerStatus.textContent = '중지됨';
            schedulerStatus.className = 'status-value offline';
        }

        // Update Document Agent status
        const documentStatus = document.getElementById('document-status');
        if (data.services.document_agent) {
            documentStatus.textContent = '활성화';
            documentStatus.className = 'status-value online';
        } else {
            documentStatus.textContent = '비활성화';
            documentStatus.className = 'status-value offline';
        }
    } catch (error) {
        console.error('Failed to check system health:', error);
    }
}

// Load scheduled tasks
async function loadTasks() {
    try {
        const response = await fetch('/api/tasks');
        const data = await response.json();

        const tasksList = document.getElementById('tasks-list');
        if (Object.keys(data.tasks).length === 0) {
            tasksList.innerHTML = '<p style="color: #a8b2d1;">예정된 작업이 없습니다.</p>';
            return;
        }

        let tasksHTML = '';
        for (const [taskId, taskInfo] of Object.entries(data.tasks)) {
            const nextRun = taskInfo.next_run ? new Date(taskInfo.next_run).toLocaleString('ko-KR') : '예정 없음';
            tasksHTML += `
                <div class="task-item">
                    <h3>${taskId}</h3>
                    <p>유형: ${taskInfo.type} | 일정: ${taskInfo.schedule}</p>
                    <div class="next-run">다음 실행: ${nextRun}</div>
                    <button onclick="runTaskNow('${taskId}')" class="btn-primary" style="margin-top: 10px; padding: 8px 16px; font-size: 0.9em;">
                        지금 실행
                    </button>
                </div>
            `;
        }

        tasksList.innerHTML = tasksHTML;
    } catch (error) {
        console.error('Failed to load tasks:', error);
        document.getElementById('tasks-list').innerHTML = '<p style="color: #dc3545;">작업을 불러오지 못했습니다.</p>';
    }
}

// Run task immediately
async function runTaskNow(taskId) {
    try {
        const response = await fetch(`/api/tasks/${taskId}/run`, {
            method: 'POST'
        });
        const data = await response.json();

        if (data.status === 'success') {
            alert(`작업 '${taskId}'이(가) 실행되었습니다.`);
        } else {
            alert(`오류: ${data.message}`);
        }
    } catch (error) {
        console.error('Failed to run task:', error);
        alert('작업 실행에 실패했습니다.');
    }
}

// Setup chat input (Enter key to send)
function setupChatInput() {
    const chatInput = document.getElementById('chat-input');
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
}

// Send chat message
async function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();

    if (!message) return;

    // Add user message to chat
    addChatMessage(message, 'user');
    input.value = '';

    try {
        const response = await fetch('/api/ai/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        });

        const data = await response.json();

        if (data.status === 'success') {
            addChatMessage(data.response, 'ai');
        } else {
            addChatMessage(`오류: ${data.message}`, 'ai');
        }
    } catch (error) {
        console.error('Failed to send message:', error);
        addChatMessage('메시지 전송에 실패했습니다.', 'ai');
    }
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

// Generate daily report
async function generateDailyReport() {
    const sampleData = {
        sales: 1500000,
        customers: 45,
        tasks_completed: 23,
        issues: 2
    };

    try {
        const response = await fetch('/api/generate/report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(sampleData)
        });

        const data = await response.json();

        if (data.status === 'success') {
            // Display report in a modal or new window
            const reportWindow = window.open('', '_blank');
            reportWindow.document.write(`
                <html>
                <head>
                    <title>일일 보고서</title>
                    <style>
                        body { font-family: sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
                        pre { background: #f5f5f5; padding: 20px; border-radius: 5px; white-space: pre-wrap; }
                    </style>
                </head>
                <body>
                    <h1>일일 보고서</h1>
                    <pre>${data.report}</pre>
                </body>
                </html>
            `);
        } else {
            alert(`오류: ${data.message}`);
        }
    } catch (error) {
        console.error('Failed to generate report:', error);
        alert('보고서 생성에 실패했습니다.');
    }
}

// Generate email
async function generateEmail() {
    const purpose = prompt('이메일 목적을 입력하세요 (예: 프로젝트 제안):');
    if (!purpose) return;

    const recipient = prompt('수신자를 입력하세요:');
    if (!recipient) return;

    try {
        const response = await fetch('/api/generate/email', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                purpose,
                recipient,
                context: {},
                tone: 'professional'
            })
        });

        const data = await response.json();

        if (data.status === 'success') {
            const emailWindow = window.open('', '_blank');
            emailWindow.document.write(`
                <html>
                <head>
                    <title>생성된 이메일</title>
                    <style>
                        body { font-family: sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
                        .email-field { margin-bottom: 20px; }
                        .label { font-weight: bold; color: #666; }
                        .content { background: #f5f5f5; padding: 15px; border-radius: 5px; margin-top: 5px; }
                    </style>
                </head>
                <body>
                    <h1>생성된 이메일</h1>
                    <div class="email-field">
                        <div class="label">제목:</div>
                        <div class="content">${data.email.subject}</div>
                    </div>
                    <div class="email-field">
                        <div class="label">본문:</div>
                        <div class="content">${data.email.body}</div>
                    </div>
                </body>
                </html>
            `);
        } else {
            alert(`오류: ${data.message}`);
        }
    } catch (error) {
        console.error('Failed to generate email:', error);
        alert('이메일 생성에 실패했습니다.');
    }
}

// Generate meeting minutes (placeholder)
function generateMeetingMinutes() {
    alert('회의록 생성 기능은 곧 추가됩니다.');
}

// Analyze data (placeholder)
function analyzeData() {
    alert('데이터 분석 기능은 곧 추가됩니다.');
}

// Refresh data every 30 seconds
setInterval(() => {
    checkSystemHealth();
    loadTasks();
}, 30000);
