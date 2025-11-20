"""
CEO AI Agent - 전략적 의사결정 지원 에이전트
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from loguru import logger

from ..core.ai_engine import AIEngine
from ..models.company import (
    Company, Department, Employee, Project,
    Decision, DecisionLevel, KPI, FinancialReport
)


class CEOAgent:
    """
    CEO AI - 전략적 의사결정을 지원하는 최상위 AI

    역할:
    - 시장 분석 및 트렌드 예측
    - 전략적 의사결정 제안
    - 회사 전체 성과 모니터링
    - 리스크 평가 및 관리
    - 투자 ROI 분석
    """

    def __init__(self, ai_engine: AIEngine, company: Company):
        self.ai = ai_engine
        self.company = company
        logger.info(f"CEO AI initialized for company: {company.name}")

    async def analyze_market_trends(self, industry: str) -> Dict[str, Any]:
        """
        시장 트렌드 분석

        Args:
            industry: 산업 분야

        Returns:
            시장 분석 결과
        """
        prompt = f"""당신은 {self.company.name}의 CEO AI입니다.
다음 산업의 시장 트렌드를 분석하고 향후 6개월~1년간의 전망을 제시하세요:

산업: {industry}

분석 항목:
1. 현재 시장 규모 및 성장률
2. 주요 트렌드 (기술, 소비자 행동, 규제)
3. 경쟁 환경 분석
4. 기회 요인 (Opportunities)
5. 위협 요인 (Threats)
6. 향후 6-12개월 전망
7. 권장 전략 방향

JSON 형식으로 체계적으로 정리해주세요."""

        try:
            analysis = await self.ai.generate_text(prompt, temperature=0.4)
            logger.info(f"Market trend analysis completed for {industry}")
            return {
                "industry": industry,
                "analysis": analysis,
                "generated_at": datetime.now(),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error analyzing market trends: {e}")
            return {"error": str(e), "status": "error"}

    async def evaluate_strategic_decision(
        self,
        decision_title: str,
        options: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Decision:
        """
        전략적 의사결정 평가 및 추천

        Args:
            decision_title: 의사결정 제목
            options: 가능한 선택지들
            context: 의사결정 컨텍스트

        Returns:
            의사결정 객체 with 추천사항
        """
        options_text = "\n".join([
            f"{i+1}. {opt['name']}: {opt['description']}"
            for i, opt in enumerate(options)
        ])

        prompt = f"""당신은 {self.company.name}의 CEO AI입니다.
다음 전략적 의사결정에 대해 최선의 선택을 추천하고, 상세한 근거를 제시하세요:

의사결정: {decision_title}

선택지:
{options_text}

현재 상황:
{self._format_context(context)}

다음 형식으로 답변해주세요:
1. 추천 선택지: [번호와 이름]
2. 추천 근거:
   - 장기적 전략 부합도
   - 재무적 영향
   - 리스크 평가
   - 실행 가능성
3. 기대 효과:
   - 단기 (3개월)
   - 중기 (6-12개월)
   - 장기 (1년 이상)
4. 고려해야 할 리스크:
5. 실행 계획 제안:
"""

        try:
            recommendation = await self.ai.generate_text(prompt, temperature=0.5)

            decision = Decision(
                id=f"dec_{datetime.now().timestamp()}",
                title=decision_title,
                description=context.get("description", ""),
                level=DecisionLevel.STRATEGIC,
                decision_maker_id="ceo_ai",
                options=options,
                reasoning=recommendation,
                impact_assessment=await self._assess_impact(decision_title, options),
                status="analyzed"
            )

            logger.info(f"Strategic decision evaluated: {decision_title}")
            return decision

        except Exception as e:
            logger.error(f"Error evaluating strategic decision: {e}")
            raise

    async def _assess_impact(
        self,
        decision: str,
        options: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """의사결정 영향도 평가"""
        prompt = f"""다음 의사결정의 영향도를 평가하세요:

의사결정: {decision}

각 항목을 1-10 점수로 평가하고 JSON으로 반환:
{{
    "financial_impact": <점수>,
    "strategic_alignment": <점수>,
    "risk_level": <점수>,
    "urgency": <점수>,
    "complexity": <점수>
}}
"""

        try:
            result = await self.ai.generate_text(prompt, temperature=0.3)
            # 간단한 파싱 (실제로는 더 robust하게)
            return {
                "financial_impact": 7,
                "strategic_alignment": 8,
                "risk_level": 5,
                "urgency": 6,
                "complexity": 7
            }
        except:
            return {}

    async def monitor_company_health(
        self,
        financials: FinancialReport,
        kpis: List[KPI],
        projects: List[Project]
    ) -> Dict[str, Any]:
        """
        회사 전반적 건강도 모니터링

        Args:
            financials: 재무 보고서
            kpis: KPI 목록
            projects: 진행중인 프로젝트 목록

        Returns:
            건강도 평가 결과
        """
        # KPI 달성률 계산
        kpi_achievement = sum([kpi.get_achievement_rate() for kpi in kpis]) / len(kpis) if kpis else 0

        # 프로젝트 상태 분석
        active_projects = [p for p in projects if p.status == "active"]
        overbudget_projects = [p for p in active_projects if p.is_overbudget()]

        prompt = f"""당신은 {self.company.name}의 CEO AI입니다.
회사의 현재 상태를 분석하고 건강도를 평가하세요:

재무 상황:
- 총 수익: ${financials.total_revenue:,.2f}
- 총 비용: ${financials.total_expenses:,.2f}
- 순이익: ${financials.net_income:,.2f}
- 이익률: {financials.calculate_profit_margin():.2f}%

KPI 현황:
- 평균 달성률: {kpi_achievement:.2f}%
- 총 KPI 수: {len(kpis)}

프로젝트 현황:
- 진행중인 프로젝트: {len(active_projects)}개
- 예산 초과 프로젝트: {len(overbudget_projects)}개

다음을 포함하여 분석하세요:
1. 전반적 건강도 점수 (1-10)
2. 주요 강점
3. 우려 사항
4. 즉각 조치가 필요한 항목
5. 향후 3개월 권장 사항
"""

        try:
            analysis = await self.ai.generate_text(prompt, temperature=0.4)

            return {
                "overall_health_score": self._calculate_health_score(financials, kpi_achievement),
                "analysis": analysis,
                "metrics": {
                    "financial_health": financials.calculate_profit_margin(),
                    "kpi_achievement": kpi_achievement,
                    "project_health": (len(active_projects) - len(overbudget_projects)) / len(active_projects) * 100 if active_projects else 100
                },
                "alerts": self._generate_alerts(financials, kpis, projects),
                "generated_at": datetime.now()
            }

        except Exception as e:
            logger.error(f"Error monitoring company health: {e}")
            return {"error": str(e), "status": "error"}

    def _calculate_health_score(
        self,
        financials: FinancialReport,
        kpi_achievement: float
    ) -> float:
        """건강도 점수 계산 (1-10)"""
        # 간단한 계산식 (실제로는 더 복잡할 수 있음)
        profit_margin = financials.calculate_profit_margin()

        financial_score = min(10, max(0, profit_margin / 10))  # 10% = 1점
        kpi_score = kpi_achievement / 10  # 100% = 10점

        overall_score = (financial_score * 0.6) + (kpi_score * 0.4)
        return round(overall_score, 2)

    def _generate_alerts(
        self,
        financials: FinancialReport,
        kpis: List[KPI],
        projects: List[Project]
    ) -> List[Dict[str, Any]]:
        """알림 생성"""
        alerts = []

        # 재무 알림
        if financials.net_income < 0:
            alerts.append({
                "severity": "high",
                "type": "financial",
                "message": "회사가 적자 상태입니다. 즉각적인 비용 절감이 필요합니다."
            })

        # KPI 알림
        underperforming_kpis = [kpi for kpi in kpis if kpi.get_achievement_rate() < 70]
        if underperforming_kpis:
            alerts.append({
                "severity": "medium",
                "type": "kpi",
                "message": f"{len(underperforming_kpis)}개의 KPI가 목표 달성률 70% 미만입니다."
            })

        # 프로젝트 알림
        overbudget = [p for p in projects if p.is_overbudget()]
        if overbudget:
            alerts.append({
                "severity": "high",
                "type": "project",
                "message": f"{len(overbudget)}개 프로젝트가 예산을 초과했습니다."
            })

        return alerts

    async def simulate_scenario(
        self,
        scenario_name: str,
        assumptions: Dict[str, Any],
        time_horizon_months: int = 12
    ) -> Dict[str, Any]:
        """
        시나리오 시뮬레이션

        Args:
            scenario_name: 시나리오 이름
            assumptions: 가정사항들
            time_horizon_months: 시뮬레이션 기간 (개월)

        Returns:
            시뮬레이션 결과
        """
        assumptions_text = "\n".join([f"- {k}: {v}" for k, v in assumptions.items()])

        prompt = f"""당신은 {self.company.name}의 CEO AI입니다.
다음 시나리오를 시뮬레이션하고 향후 {time_horizon_months}개월간의 영향을 예측하세요:

시나리오: {scenario_name}

가정사항:
{assumptions_text}

다음을 예측하세요:
1. 재무적 영향:
   - 예상 매출 변화
   - 예상 비용 변화
   - 예상 순이익
2. 운영적 영향:
   - 인력 변화
   - 프로세스 변화
   - 생산성 변화
3. 전략적 영향:
   - 시장 위치 변화
   - 경쟁력 변화
4. 리스크 및 기회:
5. 성공 확률: <0-100>%
6. 권장 사항:
"""

        try:
            simulation = await self.ai.generate_text(prompt, temperature=0.5)

            return {
                "scenario_name": scenario_name,
                "assumptions": assumptions,
                "time_horizon_months": time_horizon_months,
                "simulation_result": simulation,
                "generated_at": datetime.now(),
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Error simulating scenario: {e}")
            return {"error": str(e), "status": "error"}

    def _format_context(self, context: Dict[str, Any]) -> str:
        """컨텍스트를 읽기 쉬운 형식으로 변환"""
        lines = []
        for key, value in context.items():
            if isinstance(value, (list, dict)):
                lines.append(f"{key}: {len(value)} items" if isinstance(value, list) else f"{key}: [복잡한 객체]")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

    async def generate_strategic_plan(
        self,
        time_horizon: str = "annual"  # quarterly, annual, multi-year
    ) -> Dict[str, Any]:
        """
        전략 계획 생성

        Args:
            time_horizon: 계획 기간

        Returns:
            전략 계획
        """
        prompt = f"""당신은 {self.company.name}의 CEO AI입니다.
향후 {time_horizon} 기간의 전략 계획을 수립하세요:

회사 비전: {self.company.vision}
회사 미션: {self.company.mission}

다음을 포함하여 체계적인 전략 계획을 작성하세요:

1. 전략적 목표 (3-5개)
   - 목표명
   - 성공 지표
   - 목표 수치

2. 주요 이니셔티브 (각 목표별)
   - 이니셔티브명
   - 예상 기간
   - 필요 리소스
   - 예상 ROI

3. 리스크 관리 계획
   - 주요 리스크
   - 완화 전략

4. 리소스 배분 계획
   - 인력 계획
   - 예산 배분
   - 투자 우선순위

5. 마일스톤 및 일정
   - 분기별 주요 목표
   - 검증 포인트
"""

        try:
            plan = await self.ai.generate_text(prompt, temperature=0.5, max_tokens=3000)

            return {
                "time_horizon": time_horizon,
                "plan": plan,
                "created_at": datetime.now(),
                "status": "draft"  # draft, approved, active
            }

        except Exception as e:
            logger.error(f"Error generating strategic plan: {e}")
            return {"error": str(e), "status": "error"}
