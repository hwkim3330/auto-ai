"""
Company Automation System

AI 기반 회사 전반 자동화:
- 워크플로우 자동화
- 의사결정 지원
- 리소스 최적화
- 예측 및 시뮬레이션

목표: 인간을 반복 작업에서 해방
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import time


class TaskPriority(Enum):
    """작업 우선순위"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """작업 상태"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class Task:
    """자동화 작업"""
    id: str
    name: str
    description: str
    task_type: str
    priority: TaskPriority
    status: TaskStatus
    assigned_ai: Optional[str] = None
    result: Optional[Any] = None
    created_at: float = 0
    completed_at: Optional[float] = None
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at == 0:
            self.created_at = time.time()


class WorkflowEngine:
    """
    워크플로우 엔진

    Features:
    - 작업 큐 관리
    - 의존성 해결
    - 병렬 실행
    - 자동 재시도
    """

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.tasks: Dict[str, Task] = {}
        self.queue: List[str] = []
        self.running_tasks: Dict[str, asyncio.Task] = {}

        # Statistics
        self.stats = {
            'total_tasks': 0,
            'completed': 0,
            'failed': 0,
            'avg_completion_time': 0
        }

    def add_task(self, task: Task):
        """작업 추가"""
        self.tasks[task.id] = task
        self.queue.append(task.id)
        self.stats['total_tasks'] += 1

        print(f"📝 Task added: {task.name} (Priority: {task.priority.name})")

    def get_ready_tasks(self) -> List[Task]:
        """실행 가능한 작업 반환 (의존성 충족)"""
        ready = []

        for task_id in self.queue:
            task = self.tasks[task_id]

            if task.status != TaskStatus.PENDING:
                continue

            # Check dependencies
            deps_satisfied = all(
                self.tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
                if dep_id in self.tasks
            )

            if deps_satisfied:
                ready.append(task)

        # Sort by priority
        ready.sort(key=lambda t: t.priority.value, reverse=True)

        return ready

    async def execute_task(self, task: Task) -> Any:
        """작업 실행"""
        print(f"🚀 Executing: {task.name}")

        task.status = TaskStatus.IN_PROGRESS

        try:
            # AI에게 작업 위임
            result = await self.orchestrator.query(
                prompt=f"Task: {task.description}",
                task_type=task.task_type
            )

            if 'error' in result:
                raise Exception(result['error'])

            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()

            self.stats['completed'] += 1

            completion_time = task.completed_at - task.created_at
            self._update_avg_completion_time(completion_time)

            print(f"✅ Completed: {task.name} (took {completion_time:.2f}s)")

            return result

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.result = {'error': str(e)}

            self.stats['failed'] += 1

            print(f"❌ Failed: {task.name} - {e}")

            raise

    def _update_avg_completion_time(self, new_time: float):
        """평균 완료 시간 업데이트"""
        n = self.stats['completed']
        avg = self.stats['avg_completion_time']

        self.stats['avg_completion_time'] = (avg * (n - 1) + new_time) / n

    async def run(self, max_concurrent: int = 3):
        """워크플로우 실행"""
        print(f"\n🏃 Starting workflow engine (max {max_concurrent} concurrent tasks)...\n")

        while self.queue:
            # Get ready tasks
            ready_tasks = self.get_ready_tasks()

            if not ready_tasks:
                # Wait for running tasks
                if self.running_tasks:
                    await asyncio.sleep(0.5)
                    continue
                else:
                    # No ready tasks and no running tasks - deadlock or all done
                    break

            # Execute tasks (up to max_concurrent)
            while ready_tasks and len(self.running_tasks) < max_concurrent:
                task = ready_tasks.pop(0)

                # Start task
                async_task = asyncio.create_task(self.execute_task(task))
                self.running_tasks[task.id] = async_task

                # Remove from queue
                if task.id in self.queue:
                    self.queue.remove(task.id)

            # Wait for at least one task to complete
            if self.running_tasks:
                done, pending = await asyncio.wait(
                    self.running_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Remove completed tasks
                for task_id in list(self.running_tasks.keys()):
                    if self.running_tasks[task_id] in done:
                        del self.running_tasks[task_id]

        print(f"\n✅ Workflow completed!")
        print(f"   Total: {self.stats['total_tasks']}")
        print(f"   Completed: {self.stats['completed']}")
        print(f"   Failed: {self.stats['failed']}")
        print(f"   Avg completion time: {self.stats['avg_completion_time']:.2f}s")

    def get_status(self) -> Dict:
        """현재 상태 반환"""
        return {
            'total_tasks': self.stats['total_tasks'],
            'completed': self.stats['completed'],
            'failed': self.stats['failed'],
            'pending': len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
            'in_progress': len([t for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS]),
            'avg_completion_time': self.stats['avg_completion_time']
        }


class CompanyAutomation:
    """
    회사 자동화 시스템

    Features:
    - 반복 작업 자동화
    - 의사결정 지원
    - 리소스 최적화
    - 시뮬레이션 기반 예측
    """

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.workflow_engine = WorkflowEngine(orchestrator)

        # Automation templates
        self.templates = {
            'code_review': self._code_review_workflow,
            'data_analysis': self._data_analysis_workflow,
            'report_generation': self._report_generation_workflow,
            'optimization': self._optimization_workflow,
            'prediction': self._prediction_workflow
        }

    def _code_review_workflow(self, code: str) -> List[Task]:
        """코드 리뷰 워크플로우"""
        tasks = [
            Task(
                id='code_review_1',
                name='Static Analysis',
                description=f'Analyze this code for bugs and issues:\n{code}',
                task_type='code_generation',
                priority=TaskPriority.HIGH,
                status=TaskStatus.PENDING
            ),
            Task(
                id='code_review_2',
                name='Security Check',
                description=f'Check for security vulnerabilities:\n{code}',
                task_type='code_generation',
                priority=TaskPriority.CRITICAL,
                status=TaskStatus.PENDING
            ),
            Task(
                id='code_review_3',
                name='Performance Analysis',
                description=f'Suggest performance improvements:\n{code}',
                task_type='optimization',
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING
            ),
            Task(
                id='code_review_4',
                name='Generate Summary',
                description='Combine all review results into a summary report',
                task_type='planning',
                priority=TaskPriority.HIGH,
                status=TaskStatus.PENDING,
                dependencies=['code_review_1', 'code_review_2', 'code_review_3']
            )
        ]

        return tasks

    def _data_analysis_workflow(self, data_description: str) -> List[Task]:
        """데이터 분석 워크플로우"""
        tasks = [
            Task(
                id='data_1',
                name='Data Exploration',
                description=f'Explore and understand the data: {data_description}',
                task_type='reasoning',
                priority=TaskPriority.HIGH,
                status=TaskStatus.PENDING
            ),
            Task(
                id='data_2',
                name='Statistical Analysis',
                description=f'Perform statistical analysis on: {data_description}',
                task_type='reasoning',
                priority=TaskPriority.HIGH,
                status=TaskStatus.PENDING,
                dependencies=['data_1']
            ),
            Task(
                id='data_3',
                name='Visualization Plan',
                description='Create visualization recommendations',
                task_type='planning',
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING,
                dependencies=['data_2']
            )
        ]

        return tasks

    def _report_generation_workflow(self, topic: str) -> List[Task]:
        """보고서 생성 워크플로우"""
        tasks = [
            Task(
                id='report_1',
                name='Research',
                description=f'Research topic: {topic}',
                task_type='reasoning',
                priority=TaskPriority.HIGH,
                status=TaskStatus.PENDING
            ),
            Task(
                id='report_2',
                name='Outline',
                description=f'Create report outline for: {topic}',
                task_type='planning',
                priority=TaskPriority.HIGH,
                status=TaskStatus.PENDING,
                dependencies=['report_1']
            ),
            Task(
                id='report_3',
                name='Draft',
                description=f'Write report draft',
                task_type='reasoning',
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING,
                dependencies=['report_2']
            )
        ]

        return tasks

    def _optimization_workflow(self, system_description: str) -> List[Task]:
        """최적화 워크플로우"""
        tasks = [
            Task(
                id='opt_1',
                name='Analyze System',
                description=f'Analyze system for optimization opportunities: {system_description}',
                task_type='reasoning',
                priority=TaskPriority.HIGH,
                status=TaskStatus.PENDING
            ),
            Task(
                id='opt_2',
                name='Generate Solutions',
                description='Generate optimization solutions',
                task_type='optimization',
                priority=TaskPriority.HIGH,
                status=TaskStatus.PENDING,
                dependencies=['opt_1']
            ),
            Task(
                id='opt_3',
                name='Evaluate Tradeoffs',
                description='Evaluate solution tradeoffs',
                task_type='reasoning',
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING,
                dependencies=['opt_2']
            )
        ]

        return tasks

    def _prediction_workflow(self, scenario: str) -> List[Task]:
        """예측 워크플로우 (시뮬레이션 기반)"""
        tasks = [
            Task(
                id='pred_1',
                name='Model Current State',
                description=f'Model current state of: {scenario}',
                task_type='reasoning',
                priority=TaskPriority.HIGH,
                status=TaskStatus.PENDING
            ),
            Task(
                id='pred_2',
                name='Identify Variables',
                description='Identify key variables and parameters',
                task_type='reasoning',
                priority=TaskPriority.HIGH,
                status=TaskStatus.PENDING,
                dependencies=['pred_1']
            ),
            Task(
                id='pred_3',
                name='Simulate Scenarios',
                description='Simulate multiple future scenarios',
                task_type='prediction',
                priority=TaskPriority.CRITICAL,
                status=TaskStatus.PENDING,
                dependencies=['pred_2']
            ),
            Task(
                id='pred_4',
                name='Generate Recommendations',
                description='Generate actionable recommendations',
                task_type='planning',
                priority=TaskPriority.HIGH,
                status=TaskStatus.PENDING,
                dependencies=['pred_3']
            )
        ]

        return tasks

    async def automate(self, workflow_type: str, **kwargs) -> Dict:
        """
        자동화 실행

        Args:
            workflow_type: 워크플로우 유형
            **kwargs: 워크플로우별 파라미터

        Returns:
            실행 결과
        """
        if workflow_type not in self.templates:
            return {'error': f'Unknown workflow type: {workflow_type}'}

        print(f"\n{'='*60}")
        print(f"🤖 Starting {workflow_type} automation")
        print(f"{'='*60}\n")

        # Get workflow template
        workflow_fn = self.templates[workflow_type]
        tasks = workflow_fn(**kwargs)

        # Add tasks to engine
        for task in tasks:
            self.workflow_engine.add_task(task)

        # Run workflow
        await self.workflow_engine.run()

        # Collect results
        results = {}
        for task in tasks:
            results[task.id] = {
                'name': task.name,
                'status': task.status.value,
                'result': task.result
            }

        return {
            'workflow_type': workflow_type,
            'status': 'completed',
            'tasks': results,
            'stats': self.workflow_engine.get_status()
        }


if __name__ == "__main__":
    print("=== Company Automation System ===\n")

    print("Available Workflows:")
    print("  1. code_review - 코드 리뷰 자동화")
    print("  2. data_analysis - 데이터 분석 자동화")
    print("  3. report_generation - 보고서 생성 자동화")
    print("  4. optimization - 시스템 최적화")
    print("  5. prediction - 시뮬레이션 기반 예측")

    print("\n✅ Company Automation System ready")
    print("\nUsage example:")
    print("  automation = CompanyAutomation(orchestrator)")
    print("  result = await automation.automate('code_review', code='...')")
    print("  result = await automation.automate('prediction', scenario='...')")
