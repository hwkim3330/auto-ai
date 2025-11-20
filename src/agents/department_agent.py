"""
Department AI Agent - 부서별 운영 최적화 에이전트
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from loguru import logger

from ..core.ai_engine import AIEngine
from ..models.company import (
    Department, Employee, Team, Project, Task,
    TaskStatus, TaskPriority, Budget
)


class DepartmentAgent:
    """
    부서 AI - 부서별 운영을 최적화하는 AI

    역할:
    - 프로젝트 관리 자동화
    - 리소스 최적 배분
    - 팀 성과 분석
    - 워크플로우 최적화
    - 부서 간 협업 조율
    """

    def __init__(
        self,
        ai_engine: AIEngine,
        department: Department
    ):
        self.ai = ai_engine
        self.department = department
        logger.info(f"Department AI initialized for: {department.name}")

    async def optimize_resource_allocation(
        self,
        employees: List[Employee],
        projects: List[Project],
        tasks: List[Task]
    ) -> Dict[str, Any]:
        """
        리소스 최적 배분

        Args:
            employees: 부서 직원 목록
            projects: 진행중인 프로젝트
            tasks: 할당되지 않은 태스크 목록

        Returns:
            리소스 배분 계획
        """
        # 직원별 현재 workload 계산
        employee_workload = {}
        for emp in employees:
            assigned_tasks = [t for t in tasks if t.assignee_id == emp.id and t.status != TaskStatus.DONE]
            total_hours = sum([t.estimated_hours for t in assigned_tasks])
            employee_workload[emp.id] = {
                "name": emp.name,
                "current_hours": total_hours,
                "availability": emp.availability,
                "skills": [s.name for s in emp.skills],
                "performance": emp.performance_score
            }

        # 미할당 태스크 정보
        unassigned_tasks = [t for t in tasks if not t.assignee_id and t.status == TaskStatus.TODO]

        prompt = f"""당신은 {self.department.name} 부서의 운영 AI입니다.
다음 직원들과 태스크를 분석하여 최적의 리소스 배분 계획을 수립하세요:

직원 현황 ({len(employees)}명):
{self._format_employee_workload(employee_workload)}

미할당 태스크 ({len(unassigned_tasks)}개):
{self._format_unassigned_tasks(unassigned_tasks)}

다음을 고려하여 배분 계획을 수립하세요:
1. 직원의 스킬과 태스크 요구사항 매칭
2. 현재 workload 균형
3. 태스크 우선순위
4. 직원 성과 점수 (높은 성과자에게 중요 태스크)

각 태스크에 대해 추천 담당자와 근거를 제시하세요.
"""

        try:
            allocation_plan = await self.ai.generate_text(prompt, temperature=0.4)

            # 실제 할당 추천 생성
            assignments = self._generate_assignments(
                unassigned_tasks,
                employees,
                employee_workload
            )

            return {
                "department": self.department.name,
                "allocation_plan": allocation_plan,
                "assignments": assignments,
                "workload_balance": self._calculate_workload_balance(employee_workload),
                "generated_at": datetime.now()
            }

        except Exception as e:
            logger.error(f"Error optimizing resource allocation: {e}")
            return {"error": str(e), "status": "error"}

    def _generate_assignments(
        self,
        tasks: List[Task],
        employees: List[Employee],
        workload: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """태스크 할당 추천 생성"""
        assignments = []

        for task in tasks:
            # 간단한 할당 로직 (실제로는 더 복잡)
            # 가용성이 가장 높고, 성과가 좋은 직원에게 우선 할당
            best_employee = min(
                employees,
                key=lambda e: workload[e.id]["current_hours"] / (e.availability * e.performance_score)
            )

            assignments.append({
                "task_id": task.id,
                "task_title": task.title,
                "recommended_assignee": best_employee.id,
                "assignee_name": best_employee.name,
                "confidence": 0.85,
                "reasoning": f"낮은 workload와 높은 성과 점수 ({best_employee.performance_score})"
            })

        return assignments

    def _calculate_workload_balance(self, workload: Dict[str, Any]) -> float:
        """Workload 균형도 계산 (0-100, 100이 가장 균형)"""
        hours = [w["current_hours"] for w in workload.values()]
        if not hours:
            return 100.0

        avg_hours = sum(hours) / len(hours)
        if avg_hours == 0:
            return 100.0

        variance = sum([(h - avg_hours) ** 2 for h in hours]) / len(hours)
        std_dev = variance ** 0.5

        # 표준편차가 작을수록 균형이 좋음
        balance = max(0, 100 - (std_dev / avg_hours * 100))
        return round(balance, 2)

    async def analyze_team_performance(
        self,
        team: Team,
        employees: List[Employee],
        completed_tasks: List[Task],
        time_period_days: int = 30
    ) -> Dict[str, Any]:
        """
        팀 성과 분석

        Args:
            team: 분석할 팀
            employees: 팀원 목록
            completed_tasks: 완료된 태스크 목록
            time_period_days: 분석 기간 (일)

        Returns:
            성과 분석 결과
        """
        cutoff_date = datetime.now() - timedelta(days=time_period_days)
        recent_tasks = [
            t for t in completed_tasks
            if t.completed_at and t.completed_at >= cutoff_date
        ]

        # 성과 지표 계산
        total_completed = len(recent_tasks)
        total_estimated = sum([t.estimated_hours for t in recent_tasks])
        total_actual = sum([t.actual_hours for t in recent_tasks])
        efficiency = (total_estimated / total_actual * 100) if total_actual > 0 else 100

        avg_performance = sum([e.performance_score for e in employees]) / len(employees) if employees else 0

        prompt = f"""당신은 {self.department.name} 부서의 운영 AI입니다.
{team.name} 팀의 최근 {time_period_days}일간 성과를 분석하세요:

성과 지표:
- 완료된 태스크: {total_completed}개
- 예상 소요 시간: {total_estimated:.1f}시간
- 실제 소요 시간: {total_actual:.1f}시간
- 효율성: {efficiency:.1f}%
- 팀 평균 성과 점수: {avg_performance:.1f}/10

팀원 수: {len(employees)}명

다음을 분석하세요:
1. 전반적 성과 평가 (1-10 점수)
2. 주요 강점
3. 개선이 필요한 영역
4. 효율성 향상 방안
5. 팀원 개발 제안
6. 다음 달 목표 제안
"""

        try:
            analysis = await self.ai.generate_text(prompt, temperature=0.4)

            return {
                "team_name": team.name,
                "period_days": time_period_days,
                "metrics": {
                    "completed_tasks": total_completed,
                    "efficiency": efficiency,
                    "avg_performance": avg_performance,
                    "productivity": total_completed / len(employees) if employees else 0
                },
                "analysis": analysis,
                "recommendations": self._generate_team_recommendations(
                    efficiency,
                    avg_performance,
                    total_completed
                ),
                "generated_at": datetime.now()
            }

        except Exception as e:
            logger.error(f"Error analyzing team performance: {e}")
            return {"error": str(e), "status": "error"}

    def _generate_team_recommendations(
        self,
        efficiency: float,
        avg_performance: float,
        completed_tasks: int
    ) -> List[str]:
        """팀 개선 제안 생성"""
        recommendations = []

        if efficiency < 80:
            recommendations.append("효율성이 낮습니다. 프로세스 개선이 필요합니다.")

        if avg_performance < 6:
            recommendations.append("팀 성과가 낮습니다. 교육 및 코칭이 필요합니다.")

        if completed_tasks < 10:
            recommendations.append("생산성이 낮습니다. 리소스 또는 프로세스를 점검하세요.")

        if not recommendations:
            recommendations.append("팀이 잘 운영되고 있습니다. 현재 수준을 유지하세요.")

        return recommendations

    async def optimize_workflow(
        self,
        current_process: str,
        bottlenecks: List[str]
    ) -> Dict[str, Any]:
        """
        워크플로우 최적화

        Args:
            current_process: 현재 프로세스 설명
            bottlenecks: 병목 지점 목록

        Returns:
            최적화 제안
        """
        bottlenecks_text = "\n".join([f"- {b}" for b in bottlenecks])

        prompt = f"""당신은 {self.department.name} 부서의 운영 AI입니다.
현재 워크플로우를 분석하고 최적화 방안을 제시하세요:

현재 프로세스:
{current_process}

발견된 병목 지점:
{bottlenecks_text}

다음을 제안하세요:
1. 개선된 워크플로우 (단계별)
2. 자동화 가능한 부분
3. 병목 해소 방안
4. 예상 효율성 향상: <0-100>%
5. 구현 난이도: [쉬움/보통/어려움]
6. 예상 ROI
"""

        try:
            optimization = await self.ai.generate_text(prompt, temperature=0.5)

            return {
                "department": self.department.name,
                "optimization_plan": optimization,
                "automation_opportunities": self._identify_automation_opportunities(current_process),
                "generated_at": datetime.now()
            }

        except Exception as e:
            logger.error(f"Error optimizing workflow: {e}")
            return {"error": str(e), "status": "error"}

    def _identify_automation_opportunities(self, process: str) -> List[str]:
        """자동화 기회 식별 (간단한 패턴 매칭)"""
        opportunities = []

        keywords = {
            "수동": "수동 프로세스 자동화",
            "복사": "데이터 복사 자동화",
            "이메일": "이메일 자동 발송",
            "보고서": "보고서 자동 생성",
            "승인": "승인 워크플로우 자동화",
            "데이터 입력": "데이터 입력 자동화"
        }

        for keyword, opportunity in keywords.items():
            if keyword in process:
                opportunities.append(opportunity)

        return opportunities or ["추가 분석 필요"]

    async def plan_project(
        self,
        project_name: str,
        objectives: List[str],
        available_resources: Dict[str, Any],
        deadline: datetime
    ) -> Dict[str, Any]:
        """
        프로젝트 계획 수립

        Args:
            project_name: 프로젝트 이름
            objectives: 목표 리스트
            available_resources: 가용 리소스
            deadline: 마감일

        Returns:
            프로젝트 계획
        """
        objectives_text = "\n".join([f"- {obj}" for obj in objectives])
        days_until_deadline = (deadline - datetime.now()).days

        prompt = f"""당신은 {self.department.name} 부서의 운영 AI입니다.
다음 프로젝트의 상세 실행 계획을 수립하세요:

프로젝트명: {project_name}

목표:
{objectives_text}

가용 리소스:
- 인원: {available_resources.get('headcount', 0)}명
- 예산: ${available_resources.get('budget', 0):,.2f}
- 기간: {days_until_deadline}일

다음을 포함한 계획을 작성하세요:
1. 주요 마일스톤 (3-5개)
   - 마일스톤명
   - 예상 완료일
   - 산출물

2. 상세 태스크 목록 (각 마일스톤별)
   - 태스크명
   - 예상 소요 시간
   - 필요 스킬
   - 우선순위

3. 리스크 및 완화 방안

4. 리소스 배분 계획

5. 성공 기준 및 KPI
"""

        try:
            plan = await self.ai.generate_text(prompt, temperature=0.5, max_tokens=3000)

            return {
                "project_name": project_name,
                "plan": plan,
                "estimated_duration_days": days_until_deadline,
                "resource_requirements": available_resources,
                "created_at": datetime.now(),
                "status": "draft"
            }

        except Exception as e:
            logger.error(f"Error planning project: {e}")
            return {"error": str(e), "status": "error"}

    def _format_employee_workload(self, workload: Dict[str, Any]) -> str:
        """Workload를 읽기 쉬운 형식으로"""
        lines = []
        for emp_id, data in workload.items():
            lines.append(
                f"- {data['name']}: {data['current_hours']}시간 "
                f"(가용성: {data['availability']*100:.0f}%, "
                f"성과: {data['performance']}/10)"
            )
        return "\n".join(lines)

    def _format_unassigned_tasks(self, tasks: List[Task]) -> str:
        """미할당 태스크를 읽기 쉬운 형식으로"""
        lines = []
        for task in tasks[:10]:  # 최대 10개만 표시
            lines.append(
                f"- [{task.priority.value}] {task.title} "
                f"(예상: {task.estimated_hours}시간)"
            )
        if len(tasks) > 10:
            lines.append(f"... 외 {len(tasks) - 10}개")
        return "\n".join(lines)
