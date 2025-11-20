"""
Auto-AI Main Application
"""

import asyncio
from fastapi import FastAPI, WebSocket, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from loguru import logger
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.ai_engine import AIEngine
from src.core.scheduler import WorkflowScheduler
from src.agents.document_agent import DocumentAgent

# Configuration
GEMINI_API_KEY = "AIzaSyCWHhF2ijGfG8NAc5OVyej6rDp_fvrRXX0"

# Initialize FastAPI app
app = FastAPI(title="Auto-AI", description="Business Automation Platform")

# Global instances
ai_engine: AIEngine = None
scheduler: WorkflowScheduler = None
document_agent: DocumentAgent = None

# Setup templates and static files
templates = Jinja2Templates(directory="web/templates")
app.mount("/static", StaticFiles(directory="web/static"), name="static")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global ai_engine, scheduler, document_agent

    logger.info("🚀 Starting Auto-AI Platform...")

    # Initialize AI Engine
    ai_engine = AIEngine(api_key=GEMINI_API_KEY)
    logger.info("✅ AI Engine initialized")

    # Initialize Agents
    document_agent = DocumentAgent(ai_engine)
    logger.info("✅ Document Agent initialized")

    # Initialize and start Scheduler
    scheduler = WorkflowScheduler()
    scheduler.start()
    logger.info("✅ Workflow Scheduler started")

    logger.info("🎉 Auto-AI Platform is ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global scheduler

    logger.info("🛑 Shutting down Auto-AI Platform...")

    if scheduler:
        scheduler.shutdown()
        logger.info("✅ Scheduler stopped")

    logger.info("👋 Auto-AI Platform stopped")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "ai_engine": ai_engine is not None,
            "scheduler": scheduler is not None and scheduler.scheduler.running,
            "document_agent": document_agent is not None
        }
    }


@app.get("/api/tasks")
async def list_tasks():
    """List all scheduled tasks"""
    if scheduler:
        tasks = scheduler.list_tasks()
        return {"tasks": tasks}
    return {"tasks": {}}


@app.post("/api/tasks/{task_id}/run")
async def run_task(task_id: str):
    """Run a specific task immediately"""
    if scheduler:
        await scheduler.run_task_now(task_id)
        return {"status": "success", "message": f"Task '{task_id}' executed"}
    return {"status": "error", "message": "Scheduler not available"}


@app.post("/api/generate/report")
async def generate_report(data: dict):
    """Generate a daily report"""
    if document_agent:
        try:
            report = await document_agent.generate_daily_report(data)
            return {"status": "success", "report": report}
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {"status": "error", "message": str(e)}
    return {"status": "error", "message": "Document agent not available"}


@app.post("/api/generate/email")
async def generate_email(request: dict):
    """Generate an email"""
    if document_agent:
        try:
            email = await document_agent.generate_email(
                purpose=request.get("purpose"),
                recipient=request.get("recipient"),
                context=request.get("context", {}),
                tone=request.get("tone", "professional")
            )
            return {"status": "success", "email": email}
        except Exception as e:
            logger.error(f"Error generating email: {e}")
            return {"status": "error", "message": str(e)}
    return {"status": "error", "message": "Document agent not available"}


@app.post("/api/ai/chat")
async def ai_chat(message: dict):
    """Chat with AI"""
    if ai_engine:
        try:
            response = await ai_engine.generate_text(
                prompt=message.get("message"),
                context=message.get("context"),
                temperature=message.get("temperature", 0.7)
            )
            return {"status": "success", "response": response}
        except Exception as e:
            logger.error(f"Error in AI chat: {e}")
            return {"status": "error", "message": str(e)}
    return {"status": "error", "message": "AI engine not available"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back for now
            await websocket.send_text(f"Received: {data}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


def main():
    """Main entry point"""
    import uvicorn

    logger.add(
        "logs/auto-ai.log",
        rotation="1 day",
        retention="7 days",
        level="INFO"
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
