"""
Task Scheduler - 작업 스케줄링 및 관리
"""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime, timedelta
from typing import Callable, Optional, Dict, Any
from loguru import logger
import asyncio


class TaskScheduler:
    """작업 스케줄러 - 자동화된 업무 실행 관리"""

    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.tasks: Dict[str, Dict[str, Any]] = {}
        logger.info("Task Scheduler initialized")

    def start(self):
        """스케줄러 시작"""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Scheduler started")

    def shutdown(self):
        """스케줄러 종료"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler shutdown")

    def add_daily_task(
        self,
        task_id: str,
        func: Callable,
        hour: int,
        minute: int = 0,
        **kwargs
    ):
        """
        매일 특정 시간에 실행되는 작업 추가

        Args:
            task_id: 작업 ID
            func: 실행할 함수
            hour: 시 (0-23)
            minute: 분 (0-59)
            **kwargs: 함수에 전달할 인자
        """
        trigger = CronTrigger(hour=hour, minute=minute)
        job = self.scheduler.add_job(
            func,
            trigger=trigger,
            id=task_id,
            kwargs=kwargs,
            replace_existing=True
        )

        self.tasks[task_id] = {
            "func": func,
            "type": "daily",
            "schedule": f"{hour:02d}:{minute:02d}",
            "job": job
        }

        logger.info(f"Added daily task '{task_id}' at {hour:02d}:{minute:02d}")

    def add_weekly_task(
        self,
        task_id: str,
        func: Callable,
        day_of_week: int,  # 0=Monday, 6=Sunday
        hour: int,
        minute: int = 0,
        **kwargs
    ):
        """
        매주 특정 요일, 특정 시간에 실행되는 작업 추가

        Args:
            task_id: 작업 ID
            func: 실행할 함수
            day_of_week: 요일 (0=월요일, 6=일요일)
            hour: 시
            minute: 분
            **kwargs: 함수에 전달할 인자
        """
        trigger = CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute)
        job = self.scheduler.add_job(
            func,
            trigger=trigger,
            id=task_id,
            kwargs=kwargs,
            replace_existing=True
        )

        days = ["월", "화", "수", "목", "금", "토", "일"]
        self.tasks[task_id] = {
            "func": func,
            "type": "weekly",
            "schedule": f"{days[day_of_week]} {hour:02d}:{minute:02d}",
            "job": job
        }

        logger.info(f"Added weekly task '{task_id}' on {days[day_of_week]} at {hour:02d}:{minute:02d}")

    def add_monthly_task(
        self,
        task_id: str,
        func: Callable,
        day: int,  # 1-31
        hour: int,
        minute: int = 0,
        **kwargs
    ):
        """
        매월 특정 일, 특정 시간에 실행되는 작업 추가

        Args:
            task_id: 작업 ID
            func: 실행할 함수
            day: 일 (1-31)
            hour: 시
            minute: 분
            **kwargs: 함수에 전달할 인자
        """
        trigger = CronTrigger(day=day, hour=hour, minute=minute)
        job = self.scheduler.add_job(
            func,
            trigger=trigger,
            id=task_id,
            kwargs=kwargs,
            replace_existing=True
        )

        self.tasks[task_id] = {
            "func": func,
            "type": "monthly",
            "schedule": f"매월 {day}일 {hour:02d}:{minute:02d}",
            "job": job
        }

        logger.info(f"Added monthly task '{task_id}' on day {day} at {hour:02d}:{minute:02d}")

    def add_interval_task(
        self,
        task_id: str,
        func: Callable,
        minutes: Optional[int] = None,
        hours: Optional[int] = None,
        **kwargs
    ):
        """
        일정 간격으로 실행되는 작업 추가

        Args:
            task_id: 작업 ID
            func: 실행할 함수
            minutes: 분 간격
            hours: 시간 간격
            **kwargs: 함수에 전달할 인자
        """
        trigger = IntervalTrigger(minutes=minutes or 0, hours=hours or 0)
        job = self.scheduler.add_job(
            func,
            trigger=trigger,
            id=task_id,
            kwargs=kwargs,
            replace_existing=True
        )

        interval_desc = f"{hours}시간" if hours else f"{minutes}분"
        self.tasks[task_id] = {
            "func": func,
            "type": "interval",
            "schedule": f"매 {interval_desc}",
            "job": job
        }

        logger.info(f"Added interval task '{task_id}' every {interval_desc}")

    def remove_task(self, task_id: str):
        """작업 제거"""
        if task_id in self.tasks:
            try:
                self.scheduler.remove_job(task_id)
                del self.tasks[task_id]
                logger.info(f"Removed task '{task_id}'")
            except Exception as e:
                logger.error(f"Error removing task '{task_id}': {e}")
        else:
            logger.warning(f"Task '{task_id}' not found")

    def list_tasks(self) -> Dict[str, Dict[str, Any]]:
        """등록된 모든 작업 목록 반환"""
        return {
            task_id: {
                "type": info["type"],
                "schedule": info["schedule"],
                "next_run": info["job"].next_run_time
            }
            for task_id, info in self.tasks.items()
        }

    async def run_task_now(self, task_id: str):
        """작업 즉시 실행"""
        if task_id in self.tasks:
            task_info = self.tasks[task_id]
            logger.info(f"Running task '{task_id}' immediately")
            try:
                await task_info["func"]()
            except Exception as e:
                logger.error(f"Error running task '{task_id}': {e}")
        else:
            logger.warning(f"Task '{task_id}' not found")


class WorkflowScheduler(TaskScheduler):
    """워크플로우 전용 스케줄러"""

    def __init__(self):
        super().__init__()
        self.setup_default_workflows()

    def setup_default_workflows(self):
        """기본 워크플로우 설정"""
        # 일일 워크플로우
        self.add_daily_task(
            "daily_morning_briefing",
            self._daily_morning_briefing,
            hour=9,
            minute=0
        )

        self.add_daily_task(
            "daily_evening_summary",
            self._daily_evening_summary,
            hour=18,
            minute=0
        )

        # 주간 워크플로우
        self.add_weekly_task(
            "weekly_monday_planning",
            self._weekly_monday_planning,
            day_of_week=0,  # 월요일
            hour=9,
            minute=30
        )

        self.add_weekly_task(
            "weekly_friday_review",
            self._weekly_friday_review,
            day_of_week=4,  # 금요일
            hour=17,
            minute=0
        )

        # 월간 워크플로우
        self.add_monthly_task(
            "monthly_report",
            self._monthly_report,
            day=1,
            hour=10,
            minute=0
        )

        logger.info("Default workflows configured")

    async def _daily_morning_briefing(self):
        """일일 아침 브리핑"""
        logger.info("Executing daily morning briefing")
        # 이메일 확인, 일정 정리, 우선순위 작업 생성 등
        pass

    async def _daily_evening_summary(self):
        """일일 저녁 요약"""
        logger.info("Executing daily evening summary")
        # 하루 작업 요약, 내일 준비 등
        pass

    async def _weekly_monday_planning(self):
        """주간 월요일 계획"""
        logger.info("Executing weekly Monday planning")
        # 주간 목표 설정, 리소스 할당 등
        pass

    async def _weekly_friday_review(self):
        """주간 금요일 리뷰"""
        logger.info("Executing weekly Friday review")
        # 주간 성과 분석, 다음 주 계획 등
        pass

    async def _monthly_report(self):
        """월간 리포트"""
        logger.info("Executing monthly report")
        # 월간 실적 분석, 리포트 생성 등
        pass
