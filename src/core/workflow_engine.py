"""
Workflow Automation Engine - 업무 워크플로우 자동화 엔진
"""

from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum
from loguru import logger

from ..models.company import (
    Task, TaskStatus, TaskPriority,
    Approval, Employee, Decision
)


class WorkflowStatus(str, Enum):
    """워크플로우 상태"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStep(BaseModel):
    """워크플로우 단계"""
    id: str
    name: str
    action: str  # Action type
    parameters: Dict[str, Any] = {}
    condition: Optional[str] = None  # Condition to execute
    next_step_id: Optional[str] = None
    on_failure_step_id: Optional[str] = None


class Workflow(BaseModel):
    """워크플로우"""
    id: str
    name: str
    description: str
    trigger_type: str  # manual, scheduled, event
    steps: List[WorkflowStep] = []
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    variables: Dict[str, Any] = {}  # Workflow variables


class WorkflowEngine:
    """워크플로우 자동화 엔진"""

    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.action_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()
        logger.info("Workflow Engine initialized")

    def _register_default_handlers(self):
        """기본 액션 핸들러 등록"""
        self.register_action("send_email", self._send_email)
        self.register_action("create_task", self._create_task)
        self.register_action("request_approval", self._request_approval)
        self.register_action("notify", self._notify)
        self.register_action("update_status", self._update_status)
        self.register_action("assign_task", self._assign_task)

    def register_action(self, action_type: str, handler: Callable):
        """커스텀 액션 핸들러 등록"""
        self.action_handlers[action_type] = handler
        logger.info(f"Registered action handler: {action_type}")

    def create_workflow(
        self,
        name: str,
        description: str,
        steps: List[Dict[str, Any]],
        trigger_type: str = "manual"
    ) -> Workflow:
        """
        워크플로우 생성

        Args:
            name: 워크플로우 이름
            description: 설명
            steps: 워크플로우 단계들
            trigger_type: 트리거 타입

        Returns:
            생성된 워크플로우
        """
        workflow_id = f"wf_{datetime.now().timestamp()}"

        workflow_steps = [
            WorkflowStep(
                id=f"step_{i}",
                name=step["name"],
                action=step["action"],
                parameters=step.get("parameters", {}),
                condition=step.get("condition"),
                next_step_id=step.get("next_step_id")
            )
            for i, step in enumerate(steps)
        ]

        workflow = Workflow(
            id=workflow_id,
            name=name,
            description=description,
            trigger_type=trigger_type,
            steps=workflow_steps
        )

        self.workflows[workflow_id] = workflow
        logger.info(f"Workflow created: {name} ({workflow_id})")

        return workflow

    async def execute_workflow(
        self,
        workflow_id: str,
        initial_variables: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        워크플로우 실행

        Args:
            workflow_id: 워크플로우 ID
            initial_variables: 초기 변수들

        Returns:
            실행 결과
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")

        workflow = self.workflows[workflow_id]
        workflow.status = WorkflowStatus.IN_PROGRESS
        workflow.started_at = datetime.now()

        if initial_variables:
            workflow.variables.update(initial_variables)

        logger.info(f"Starting workflow: {workflow.name}")

        try:
            current_step_id = workflow.steps[0].id if workflow.steps else None

            while current_step_id:
                step = next((s for s in workflow.steps if s.id == current_step_id), None)
                if not step:
                    break

                # 조건 체크
                if step.condition and not self._evaluate_condition(step.condition, workflow.variables):
                    logger.info(f"Step {step.name} skipped (condition not met)")
                    current_step_id = step.next_step_id
                    continue

                # 단계 실행
                logger.info(f"Executing step: {step.name}")
                result = await self._execute_step(step, workflow.variables)

                # 결과를 변수에 저장
                workflow.variables[f"step_{step.id}_result"] = result

                # 다음 단계로
                if result.get("success", True):
                    current_step_id = step.next_step_id
                else:
                    current_step_id = step.on_failure_step_id or None

            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.now()

            logger.info(f"Workflow completed: {workflow.name}")

            return {
                "workflow_id": workflow_id,
                "status": "success",
                "duration_seconds": (workflow.completed_at - workflow.started_at).total_seconds(),
                "variables": workflow.variables
            }

        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            logger.error(f"Workflow failed: {workflow.name} - {e}")

            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e)
            }

    async def _execute_step(
        self,
        step: WorkflowStep,
        variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """워크플로우 단계 실행"""
        handler = self.action_handlers.get(step.action)

        if not handler:
            raise ValueError(f"Unknown action: {step.action}")

        # 파라미터에서 변수 치환
        parameters = self._substitute_variables(step.parameters, variables)

        return await handler(parameters)

    def _substitute_variables(
        self,
        parameters: Dict[str, Any],
        variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """파라미터의 변수를 실제 값으로 치환"""
        result = {}

        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                var_name = value[2:-1]
                result[key] = variables.get(var_name, value)
            else:
                result[key] = value

        return result

    def _evaluate_condition(self, condition: str, variables: Dict[str, Any]) -> bool:
        """조건 평가 (간단한 구현)"""
        # 실제로는 더 robust한 평가 필요
        try:
            return eval(condition, {"__builtins__": {}}, variables)
        except:
            return False

    # ===== 기본 액션 핸들러들 =====

    async def _send_email(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """이메일 발송"""
        logger.info(f"Sending email to {params.get('to')}: {params.get('subject')}")
        # 실제 이메일 발송 로직
        return {"success": True, "message": "Email sent"}

    async def _create_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """태스크 생성"""
        logger.info(f"Creating task: {params.get('title')}")
        # 실제 태스크 생성 로직
        return {"success": True, "task_id": f"task_{datetime.now().timestamp()}"}

    async def _request_approval(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """승인 요청"""
        logger.info(f"Requesting approval from {params.get('approver_id')}")
        # 실제 승인 요청 로직
        return {"success": True, "approval_id": f"appr_{datetime.now().timestamp()}"}

    async def _notify(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """알림 발송"""
        logger.info(f"Sending notification: {params.get('message')}")
        # 실제 알림 발송 로직
        return {"success": True}

    async def _update_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """상태 업데이트"""
        logger.info(f"Updating status: {params.get('status')}")
        # 실제 상태 업데이트 로직
        return {"success": True}

    async def _assign_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """태스크 할당"""
        logger.info(f"Assigning task {params.get('task_id')} to {params.get('assignee_id')}")
        # 실제 태스크 할당 로직
        return {"success": True}

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """워크플로우 상태 조회"""
        if workflow_id not in self.workflows:
            return {"error": "Workflow not found"}

        workflow = self.workflows[workflow_id]

        return {
            "id": workflow.id,
            "name": workflow.name,
            "status": workflow.status,
            "started_at": workflow.started_at,
            "completed_at": workflow.completed_at,
            "variables": workflow.variables
        }


# ===== 사전 정의된 워크플로우 템플릿 =====

def create_task_approval_workflow() -> List[Dict[str, Any]]:
    """태스크 승인 워크플로우"""
    return [
        {
            "name": "Create Task",
            "action": "create_task",
            "parameters": {
                "title": "${task_title}",
                "description": "${task_description}"
            }
        },
        {
            "name": "Request Approval",
            "action": "request_approval",
            "parameters": {
                "approver_id": "${manager_id}",
                "task_id": "${step_step_0_result.task_id}"
            }
        },
        {
            "name": "Notify Requester",
            "action": "notify",
            "parameters": {
                "user_id": "${requester_id}",
                "message": "Your task has been approved"
            }
        }
    ]


def create_project_launch_workflow() -> List[Dict[str, Any]]:
    """프로젝트 런칭 워크플로우"""
    return [
        {
            "name": "Send Kickoff Email",
            "action": "send_email",
            "parameters": {
                "to": "${team_email}",
                "subject": "Project Kickoff: ${project_name}",
                "body": "We're launching ${project_name}!"
            }
        },
        {
            "name": "Create Initial Tasks",
            "action": "create_task",
            "parameters": {
                "title": "Project Setup",
                "project_id": "${project_id}"
            }
        },
        {
            "name": "Schedule First Meeting",
            "action": "notify",
            "parameters": {
                "message": "First meeting scheduled for ${meeting_date}"
            }
        }
    ]


from pydantic import BaseModel, Field
