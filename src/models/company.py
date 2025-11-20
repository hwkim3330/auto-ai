"""
Company Data Models - 회사 구조 데이터 모델
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class EmployeeRole(str, Enum):
    """직원 역할"""
    CEO = "ceo"
    COO = "coo"
    CFO = "cfo"
    CTO = "cto"
    CMO = "cmo"
    DEPARTMENT_HEAD = "department_head"
    TEAM_LEAD = "team_lead"
    SENIOR_ENGINEER = "senior_engineer"
    ENGINEER = "engineer"
    JUNIOR_ENGINEER = "junior_engineer"
    MANAGER = "manager"
    SPECIALIST = "specialist"


class DepartmentType(str, Enum):
    """부서 유형"""
    EXECUTIVE = "executive"
    DEVELOPMENT = "development"
    PRODUCT = "product"
    SALES = "sales"
    MARKETING = "marketing"
    FINANCE = "finance"
    HR = "hr"
    OPERATIONS = "operations"


class DecisionLevel(str, Enum):
    """의사결정 레벨"""
    STRATEGIC = "strategic"     # 전략적 (CEO, C-Level)
    TACTICAL = "tactical"       # 전술적 (부서장, 팀장)
    OPERATIONAL = "operational" # 운영적 (개별 직원)


class TaskStatus(str, Enum):
    """태스크 상태"""
    BACKLOG = "backlog"
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    DONE = "done"
    BLOCKED = "blocked"


class TaskPriority(str, Enum):
    """태스크 우선순위"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ===== 기본 모델 =====

class Skill(BaseModel):
    """스킬"""
    name: str
    level: int = Field(ge=1, le=10)  # 1-10
    years_experience: float = 0.0


class Employee(BaseModel):
    """직원"""
    id: str
    name: str
    email: str
    role: EmployeeRole
    department_id: str
    team_id: Optional[str] = None
    skills: List[Skill] = []
    performance_score: float = Field(default=5.0, ge=0.0, le=10.0)
    availability: float = Field(default=1.0, ge=0.0, le=1.0)  # 0.0 ~ 1.0
    reports_to: Optional[str] = None  # Manager's employee ID
    salary: float = 0.0
    hired_date: datetime
    is_active: bool = True

    def get_utilization(self) -> float:
        """현재 업무 부하율 계산"""
        return 1.0 - self.availability


class Team(BaseModel):
    """팀"""
    id: str
    name: str
    department_id: str
    team_lead_id: Optional[str] = None
    member_ids: List[str] = []
    goals: List[str] = []
    budget: float = 0.0

    def get_team_size(self) -> int:
        return len(self.member_ids)


class Department(BaseModel):
    """부서"""
    id: str
    name: str
    type: DepartmentType
    head_id: Optional[str] = None  # Department head employee ID
    team_ids: List[str] = []
    budget: float = 0.0
    goals: List[str] = []

    def get_total_headcount(self, teams: List[Team]) -> int:
        """부서 전체 인원 수"""
        dept_teams = [t for t in teams if t.id in self.team_ids]
        return sum(t.get_team_size() for t in dept_teams)


class Company(BaseModel):
    """회사"""
    id: str
    name: str
    description: str
    founded_date: datetime
    departments: List[Department] = []
    total_budget: float = 0.0
    vision: str = ""
    mission: str = ""
    values: List[str] = []

    def get_total_headcount(self, teams: List[Team]) -> int:
        """전체 직원 수"""
        return sum(dept.get_total_headcount(teams) for dept in self.departments)


# ===== 업무 관련 모델 =====

class Task(BaseModel):
    """태스크"""
    id: str
    title: str
    description: str
    assignee_id: Optional[str] = None
    reporter_id: str
    project_id: str
    status: TaskStatus = TaskStatus.TODO
    priority: TaskPriority = TaskPriority.MEDIUM
    estimated_hours: float = 0.0
    actual_hours: float = 0.0
    created_at: datetime = Field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    dependencies: List[str] = []  # Task IDs
    tags: List[str] = []

    def is_overdue(self) -> bool:
        """마감일 지났는지 확인"""
        if not self.due_date:
            return False
        return datetime.now() > self.due_date and self.status != TaskStatus.DONE

    def is_blocked(self) -> bool:
        """블로킹 상태인지"""
        return self.status == TaskStatus.BLOCKED


class Milestone(BaseModel):
    """마일스톤"""
    id: str
    name: str
    description: str
    project_id: str
    due_date: datetime
    completion_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    task_ids: List[str] = []

    def is_complete(self) -> bool:
        return self.completion_percentage >= 100.0


class Project(BaseModel):
    """프로젝트"""
    id: str
    name: str
    description: str
    owner_id: str  # Employee ID
    department_id: str
    status: str = "planning"  # planning, active, on_hold, completed
    priority: TaskPriority = TaskPriority.MEDIUM
    budget: float = 0.0
    spent: float = 0.0
    start_date: datetime
    end_date: Optional[datetime] = None
    milestone_ids: List[str] = []
    task_ids: List[str] = []
    team_member_ids: List[str] = []

    def get_budget_utilization(self) -> float:
        """예산 사용률"""
        if self.budget == 0:
            return 0.0
        return (self.spent / self.budget) * 100.0

    def is_overbudget(self) -> bool:
        """예산 초과 여부"""
        return self.spent > self.budget


# ===== 재무 관련 모델 =====

class ExpenseCategory(str, Enum):
    """비용 카테고리"""
    PERSONNEL = "personnel"
    OPERATIONAL = "operational"
    MARKETING = "marketing"
    R_AND_D = "r_and_d"
    CAPITAL = "capital"
    OTHER = "other"


class Expense(BaseModel):
    """지출"""
    id: str
    category: ExpenseCategory
    amount: float
    description: str
    department_id: Optional[str] = None
    project_id: Optional[str] = None
    date: datetime = Field(default_factory=datetime.now)
    approved_by: Optional[str] = None


class Revenue(BaseModel):
    """수익"""
    id: str
    amount: float
    source: str
    description: str
    date: datetime = Field(default_factory=datetime.now)


class Budget(BaseModel):
    """예산"""
    id: str
    department_id: Optional[str] = None
    project_id: Optional[str] = None
    fiscal_year: int
    fiscal_quarter: int  # 1-4
    allocated: float
    spent: float = 0.0

    def get_remaining(self) -> float:
        return self.allocated - self.spent

    def get_utilization_rate(self) -> float:
        if self.allocated == 0:
            return 0.0
        return (self.spent / self.allocated) * 100.0


class FinancialReport(BaseModel):
    """재무 보고서"""
    period_start: datetime
    period_end: datetime
    total_revenue: float = 0.0
    total_expenses: float = 0.0
    net_income: float = 0.0
    expenses_by_category: Dict[str, float] = {}
    revenue_by_source: Dict[str, float] = {}

    def calculate_profit_margin(self) -> float:
        """이익률 계산"""
        if self.total_revenue == 0:
            return 0.0
        return (self.net_income / self.total_revenue) * 100.0


# ===== 성과 관련 모델 =====

class KPI(BaseModel):
    """핵심 성과 지표"""
    id: str
    name: str
    description: str
    target_value: float
    current_value: float = 0.0
    unit: str  # e.g., "%", "$", "count"
    department_id: Optional[str] = None
    employee_id: Optional[str] = None
    period_start: datetime
    period_end: datetime

    def get_achievement_rate(self) -> float:
        """목표 달성률"""
        if self.target_value == 0:
            return 0.0
        return (self.current_value / self.target_value) * 100.0

    def is_achieved(self) -> bool:
        """목표 달성 여부"""
        return self.current_value >= self.target_value


class PerformanceReview(BaseModel):
    """성과 평가"""
    id: str
    employee_id: str
    reviewer_id: str
    period_start: datetime
    period_end: datetime
    overall_score: float = Field(ge=0.0, le=10.0)
    technical_skills: float = Field(ge=0.0, le=10.0)
    communication: float = Field(ge=0.0, le=10.0)
    teamwork: float = Field(ge=0.0, le=10.0)
    leadership: float = Field(default=0.0, ge=0.0, le=10.0)
    comments: str = ""
    goals_for_next_period: List[str] = []


# ===== 의사결정 관련 모델 =====

class Decision(BaseModel):
    """의사결정"""
    id: str
    title: str
    description: str
    level: DecisionLevel
    decision_maker_id: str  # Employee ID
    options: List[Dict[str, Any]]  # List of possible options
    selected_option: Optional[Dict[str, Any]] = None
    reasoning: str = ""
    impact_assessment: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.now)
    decided_at: Optional[datetime] = None
    status: str = "pending"  # pending, approved, rejected, implemented

    def is_strategic(self) -> bool:
        return self.level == DecisionLevel.STRATEGIC

    def is_automated(self) -> bool:
        """자동화 가능한 의사결정인지"""
        return self.level == DecisionLevel.OPERATIONAL


class Approval(BaseModel):
    """승인"""
    id: str
    request_type: str  # expense, hire, project, etc.
    requester_id: str
    approver_id: str
    amount: Optional[float] = None
    description: str
    status: str = "pending"  # pending, approved, rejected
    requested_at: datetime = Field(default_factory=datetime.now)
    responded_at: Optional[datetime] = None
    comments: str = ""
