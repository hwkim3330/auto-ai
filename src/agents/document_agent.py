"""
Document Agent - 문서 자동화 에이전트
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from loguru import logger
import json

from ..core.ai_engine import AIEngine


class DocumentAgent:
    """문서 생성 및 관리 자동화"""

    def __init__(self, ai_engine: AIEngine):
        self.ai = ai_engine
        logger.info("Document Agent initialized")

    async def generate_daily_report(
        self,
        data: Dict[str, Any],
        date: Optional[datetime] = None
    ) -> str:
        """
        일일 보고서 생성

        Args:
            data: 보고서에 포함할 데이터
            date: 보고서 날짜 (기본값: 오늘)

        Returns:
            생성된 보고서 (Markdown 형식)
        """
        if date is None:
            date = datetime.now()

        prompt = f"""다음 데이터를 바탕으로 {date.strftime('%Y년 %m월 %d일')} 일일 업무 보고서를 작성하세요:

데이터:
{json.dumps(data, indent=2, ensure_ascii=False)}

보고서는 다음 섹션을 포함해야 합니다:
1. 요약 (Executive Summary)
2. 주요 성과
3. 진행 중인 작업
4. 이슈 및 위험 사항
5. 다음 단계

Markdown 형식으로 작성해주세요."""

        try:
            report = await self.ai.generate_text(prompt, temperature=0.5)
            logger.info(f"Daily report generated for {date.strftime('%Y-%m-%d')}")
            return report
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
            raise

    async def generate_weekly_report(
        self,
        daily_reports: List[str],
        week_number: Optional[int] = None
    ) -> str:
        """
        주간 보고서 생성

        Args:
            daily_reports: 일일 보고서 목록
            week_number: 주차 번호

        Returns:
            생성된 주간 보고서
        """
        week_num = week_number or datetime.now().isocalendar()[1]

        combined_data = "\n\n---\n\n".join(daily_reports)

        prompt = f"""다음 일일 보고서들을 바탕으로 {week_num}주차 주간 보고서를 작성하세요:

{combined_data}

주간 보고서는 다음을 포함해야 합니다:
1. 주간 요약
2. 주요 성과 및 지표
3. 목표 달성률
4. 주요 이슈 및 해결 방안
5. 다음 주 계획

Markdown 형식으로 작성해주세요."""

        try:
            report = await self.ai.generate_text(prompt, temperature=0.5, max_tokens=3000)
            logger.info(f"Weekly report generated for week {week_num}")
            return report
        except Exception as e:
            logger.error(f"Error generating weekly report: {e}")
            raise

    async def generate_email(
        self,
        purpose: str,
        recipient: str,
        context: Dict[str, Any],
        tone: str = "professional"
    ) -> Dict[str, str]:
        """
        이메일 자동 생성

        Args:
            purpose: 이메일 목적
            recipient: 수신자
            context: 컨텍스트 정보
            tone: 톤 (professional, friendly, formal)

        Returns:
            제목과 본문을 포함한 딕셔너리
        """
        tone_guide = {
            "professional": "전문적이고 비즈니스적인",
            "friendly": "친근하고 격식 없는",
            "formal": "매우 격식있고 공식적인"
        }

        tone_desc = tone_guide.get(tone, "전문적인")

        prompt = f"""다음 정보로 {tone_desc} 이메일을 작성하세요:

목적: {purpose}
수신자: {recipient}
컨텍스트:
{json.dumps(context, indent=2, ensure_ascii=False)}

JSON 형식으로 제목(subject)과 본문(body)을 제공해주세요.
"""

        try:
            response = await self.ai.generate_text(prompt, temperature=0.6)
            # JSON 파싱 (간단한 예시)
            logger.info(f"Email generated for purpose: {purpose}")
            return {"subject": "생성된 제목", "body": response}
        except Exception as e:
            logger.error(f"Error generating email: {e}")
            raise

    async def generate_meeting_minutes(
        self,
        meeting_title: str,
        attendees: List[str],
        discussion_points: List[str],
        decisions: List[str],
        action_items: List[Dict[str, str]]
    ) -> str:
        """
        회의록 생성

        Args:
            meeting_title: 회의 제목
            attendees: 참석자 목록
            discussion_points: 논의 사항
            decisions: 결정 사항
            action_items: 액션 아이템 (owner, task, deadline)

        Returns:
            생성된 회의록
        """
        attendees_list = "\n".join(f"- {name}" for name in attendees)
        discussions = "\n".join(f"- {point}" for point in discussion_points)
        decisions_list = "\n".join(f"- {decision}" for decision in decisions)
        actions = "\n".join(
            f"- {item['task']} (담당: {item['owner']}, 기한: {item.get('deadline', 'TBD')})"
            for item in action_items
        )

        prompt = f"""다음 정보를 바탕으로 전문적인 회의록을 작성하세요:

회의 제목: {meeting_title}
날짜: {datetime.now().strftime('%Y년 %m월 %d일')}

참석자:
{attendees_list}

논의 사항:
{discussions}

결정 사항:
{decisions_list}

액션 아이템:
{actions}

회의록은 다음 형식을 따라주세요:
1. 회의 개요
2. 참석자
3. 논의 내용 (상세)
4. 결정 사항
5. 액션 아이템
6. 다음 회의 일정 (제안)

Markdown 형식으로 작성해주세요."""

        try:
            minutes = await self.ai.generate_text(prompt, temperature=0.4)
            logger.info(f"Meeting minutes generated for: {meeting_title}")
            return minutes
        except Exception as e:
            logger.error(f"Error generating meeting minutes: {e}")
            raise

    async def generate_proposal(
        self,
        project_name: str,
        objectives: List[str],
        scope: str,
        budget: Optional[float] = None,
        timeline: Optional[str] = None
    ) -> str:
        """
        제안서 생성

        Args:
            project_name: 프로젝트 이름
            objectives: 목표 리스트
            scope: 범위
            budget: 예산
            timeline: 타임라인

        Returns:
            생성된 제안서
        """
        objectives_list = "\n".join(f"- {obj}" for obj in objectives)
        budget_text = f"예산: {budget:,.0f}원" if budget else "예산: 협의 필요"
        timeline_text = f"일정: {timeline}" if timeline else "일정: 협의 필요"

        prompt = f"""다음 정보를 바탕으로 전문적인 프로젝트 제안서를 작성하세요:

프로젝트명: {project_name}

목표:
{objectives_list}

범위:
{scope}

{budget_text}
{timeline_text}

제안서는 다음 섹션을 포함해야 합니다:
1. 개요 (Executive Summary)
2. 배경 및 필요성
3. 프로젝트 목표
4. 범위 및 산출물
5. 방법론 및 접근 방식
6. 일정 및 마일스톤
7. 예산 및 자원
8. 위험 관리
9. 기대 효과
10. 결론

Markdown 형식으로 전문적이고 설득력 있게 작성해주세요."""

        try:
            proposal = await self.ai.generate_text(prompt, temperature=0.6, max_tokens=4000)
            logger.info(f"Proposal generated for: {project_name}")
            return proposal
        except Exception as e:
            logger.error(f"Error generating proposal: {e}")
            raise

    async def summarize_document(
        self,
        document: str,
        max_length: int = 500
    ) -> str:
        """
        문서 요약

        Args:
            document: 원본 문서
            max_length: 최대 길이

        Returns:
            요약된 문서
        """
        try:
            summary = await self.ai.summarize(
                document,
                max_length=max_length,
                style="executive"
            )
            logger.info("Document summarized")
            return summary
        except Exception as e:
            logger.error(f"Error summarizing document: {e}")
            raise

    async def translate_document(
        self,
        document: str,
        target_language: str
    ) -> str:
        """
        문서 번역

        Args:
            document: 원본 문서
            target_language: 목표 언어

        Returns:
            번역된 문서
        """
        try:
            translated = await self.ai.translate(document, target_language)
            logger.info(f"Document translated to {target_language}")
            return translated
        except Exception as e:
            logger.error(f"Error translating document: {e}")
            raise
