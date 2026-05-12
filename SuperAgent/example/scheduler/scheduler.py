"""
Scheduler 模块 - 任务调度器核心
支持定时任务、延迟任务、周期任务
"""
import asyncio
import os
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from loguru import logger
from pydantic import BaseModel, Field


class JobType(str, Enum):
    """任务类型"""
    CRON = "cron"           # Cron 表达式任务
    INTERVAL = "interval"   # 间隔循环任务
    DATE = "date"           # 一次性延迟任务


class JobStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"     # 待执行
    RUNNING = "running"     # 执行中
    SUCCESS = "success"     # 执行成功
    FAILED = "failed"       # 执行失败


# ============================================================================
# 数据模型
# ============================================================================

class JobConfig(BaseModel):
    """任务配置"""
    job_id: Optional[str] = None
    job_name: str
    job_type: JobType = JobType.CRON
    target_url: str
    target_method: str = "POST"
    request_data: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    # Cron 配置
    crontab: Optional[str] = None  # 5位: 分 时 日 月 周
    # Interval 配置
    interval_seconds: Optional[int] = None
    interval_minutes: Optional[int] = None
    interval_hours: Optional[int] = None
    # Date 配置
    run_date: Optional[str] = None  # ISO 格式
    # 其他
    timezone: str = "Asia/Shanghai"
    enabled: bool = True
    max_instances: int = 1


class JobInfo(BaseModel):
    """任务信息"""
    job_id: str
    job_name: str
    job_type: JobType
    target_url: str
    target_method: str
    crontab: Optional[str] = None
    interval_seconds: Optional[int] = None
    interval_hours: Optional[int] = None
    interval_minutes: Optional[int] = None
    run_date: Optional[str] = None
    next_run_time: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    enabled: bool = True
    created_at: str


class JobResponse(BaseModel):
    """任务响应"""
    success: bool
    message: str
    job_id: Optional[str] = None
    job_info: Optional[JobInfo] = None


class JobListResponse(BaseModel):
    """任务列表响应"""
    success: bool
    total: int
    jobs: List[JobInfo]


# ============================================================================
# 任务执行器
# ============================================================================

class JobExecutor:
    """任务执行器"""

    def __init__(self):
        self._clients: Dict[str, httpx.AsyncClient] = {}

    async def execute(
        self,
        job_id: str,
        job_name: str,
        target_url: str,
        target_method: str = "POST",
        request_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30
    ) -> bool:
        """
        执行 HTTP 请求任务

        Args:
            job_id: 任务ID
            job_name: 任务名称
            target_url: 目标URL
            target_method: HTTP 方法
            request_data: 请求数据
            headers: 请求头
            timeout: 超时时间(秒)

        Returns:
            bool: 执行是否成功
        """
        start_time = datetime.now()
        logger.info(f"[{job_id}] 任务开始执行: {job_name} -> {target_method} {target_url}")

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if target_method.upper() == "POST":
                    response = await client.post(
                        target_url,
                        json=request_data or {},
                        headers=headers
                    )
                elif target_method.upper() == "GET":
                    response = await client.get(
                        target_url,
                        params=request_data,
                        headers=headers
                    )
                elif target_method.upper() == "PUT":
                    response = await client.put(
                        target_url,
                        json=request_data or {},
                        headers=headers
                    )
                elif target_method.upper() == "DELETE":
                    response = await client.delete(
                        target_url,
                        headers=headers
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {target_method}")

                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(
                    f"[{job_id}] 任务执行完成: {job_name}, "
                    f"状态码: {response.status_code}, 耗时: {elapsed:.2f}s"
                )

                return response.is_success or response.status_code in [200, 201, 202]

        except httpx.TimeoutException:
            logger.error(f"[{job_id}] 任务执行超时: {job_name}")
            return False
        except httpx.RequestError as e:
            logger.error(f"[{job_id}] 任务执行失败: {job_name}, 错误: {e}")
            return False
        except Exception as e:
            logger.error(f"[{job_id}] 任务执行异常: {job_name}, 错误: {e}")
            return False

    def create_job_func(
        self,
        job_config: JobConfig
    ) -> Callable:
        """创建任务执行函数"""
        async def job_func():
            await self.execute(
                job_id=job_config.job_id or "unknown",
                job_name=job_config.job_name,
                target_url=job_config.target_url,
                target_method=job_config.target_method,
                request_data=job_config.request_data,
                headers=job_config.headers
            )
        return job_func


# ============================================================================
# 调度器管理器
# ============================================================================

class SchedulerManager:
    """
    调度器管理器
    统一管理定时任务、延迟任务、周期任务
    """

    def __init__(self):
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._jobs: Dict[str, JobConfig] = {}
        self._executor = JobExecutor()
        self._job_counter = 0

    @property
    def scheduler(self) -> AsyncIOScheduler:
        """获取调度器实例"""
        if self._scheduler is None:
            self._scheduler = AsyncIOScheduler(
                timezone="Asia/Shanghai",
                job_defaults={
                    "coalesce": True,           # 合并错过的执行
                    "max_instances": 3,         # 最大并发实例
                    "misfire_grace_time": 60    # 错过执行的宽限时间(秒)
                }
            )
        return self._scheduler

    def _generate_job_id(self) -> str:
        """生成唯一任务ID"""
        self._job_counter += 1
        return f"job_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._job_counter}"

    def _parse_cron_expression(self, cron_expr: str) -> tuple:
        """
        解析 5 位 cron 表达式

        Args:
            cron_expr: 5位 cron 表达式 "分 时 日 月 周"

        Returns:
            tuple: (minute, hour, day, month, day_of_week)
        """
        parts = cron_expr.split()
        if len(parts) != 5:
            raise ValueError(f"Cron 表达式必须是 5 位, 实际: {cron_expr}")

        return tuple(parts)

    def _get_trigger(self, job_config: JobConfig):
        """获取触发器"""
        if job_config.job_type == JobType.CRON:
            if not job_config.crontab:
                raise ValueError("Cron 任务必须提供 crontab 表达式")

            minute, hour, day, month, day_of_week = self._parse_cron_expression(
                job_config.crontab
            )
            return CronTrigger(
                minute=minute,
                hour=hour,
                day=day,
                month=month,
                day_of_week=day_of_week,
                timezone=job_config.timezone
            )

        elif job_config.job_type == JobType.INTERVAL:
            interval_kwargs = {}

            if job_config.interval_seconds:
                interval_kwargs["seconds"] = job_config.interval_seconds
            elif job_config.interval_minutes:
                interval_kwargs["minutes"] = job_config.interval_minutes
            elif job_config.interval_hours:
                interval_kwargs["hours"] = job_config.interval_hours
            else:
                # 默认 1 小时
                interval_kwargs["hours"] = 1

            return IntervalTrigger(**interval_kwargs, timezone=job_config.timezone)

        elif job_config.job_type == JobType.DATE:
            if not job_config.run_date:
                raise ValueError("Date 任务必须提供 run_date")

            run_dt = datetime.fromisoformat(job_config.run_date.replace("Z", "+00:00"))
            return DateTrigger(run_date=run_dt)

        else:
            raise ValueError(f"Unknown job type: {job_config.job_type}")

    def add_job(self, job_config: JobConfig) -> str:
        """
        添加任务

        Args:
            job_config: 任务配置

        Returns:
            str: 任务ID
        """
        # 生成任务ID
        job_id = job_config.job_id or self._generate_job_id()
        job_config.job_id = job_id

        # 获取触发器
        trigger = self._get_trigger(job_config)

        # 创建执行函数
        job_func = self._executor.create_job_func(job_config)

        # 添加到调度器
        self.scheduler.add_job(
            func=job_func,
            trigger=trigger,
            id=job_id,
            name=job_config.job_name,
            replace_existing=True,
            max_instances=job_config.max_instances
        )

        # 保存任务配置
        self._jobs[job_id] = job_config

        logger.info(f"任务添加成功: {job_id} - {job_config.job_name}")
        return job_id

    def remove_job(self, job_id: str) -> bool:
        """
        移除任务

        Args:
            job_id: 任务ID

        Returns:
            bool: 是否成功
        """
        try:
            self.scheduler.remove_job(job_id)
            self._jobs.pop(job_id, None)
            logger.info(f"任务移除成功: {job_id}")
            return True
        except Exception as e:
            logger.error(f"任务移除失败: {job_id}, 错误: {e}")
            raise e
            return False

    def get_job(self, job_id: str) -> Optional[JobInfo]:
        """获取任务信息"""
        job_config = self._jobs.get(job_id)
        if not job_config:
            return None

        # 获取下次执行时间
        next_run = None
        try:
            apscheduler_job = self.scheduler.get_job(job_id)
            if apscheduler_job and apscheduler_job.next_run_time:
                next_run = apscheduler_job.next_run_time.isoformat()
        except Exception:
            # 调度器未启动时可能获取不到
            pass

        return JobInfo(
            job_id=job_config.job_id or job_id,
            job_name=job_config.job_name,
            job_type=job_config.job_type,
            target_url=job_config.target_url,
            target_method=job_config.target_method,
            crontab=job_config.crontab,
            interval_seconds=job_config.interval_seconds,
            interval_minutes=job_config.interval_minutes,
            interval_hours=job_config.interval_hours,
            run_date=job_config.run_date,
            next_run_time=next_run,
            status=JobStatus.PENDING,
            enabled=job_config.enabled,
            created_at=datetime.now().isoformat()
        )

    def list_jobs(self) -> List[JobInfo]:
        """列出所有任务"""
        jobs = []
        for job_id in self._jobs:
            job_info = self.get_job(job_id)
            if job_info:
                jobs.append(job_info)
        return jobs

    def pause_job(self, job_id: str) -> bool:
        """暂停任务"""
        try:
            self.scheduler.pause_job(job_id)
            logger.info(f"任务已暂停: {job_id}")
            return True
        except Exception as e:
            logger.error(f"暂停任务失败: {job_id}, 错误: {e}")
            return False

    def resume_job(self, job_id: str) -> bool:
        """恢复任务"""
        try:
            self.scheduler.resume_job(job_id)
            logger.info(f"任务已恢复: {job_id}")
            return True
        except Exception as e:
            logger.error(f"恢复任务失败: {job_id}, 错误: {e}")
            return False

    def run_job(self, job_id: str) -> bool:
        """立即执行任务"""
        try:
            job_config = self._jobs.get(job_id)
            if not job_config:
                logger.error(f"任务不存在: {job_id}")
                return False

            asyncio.create_task(self._executor.execute(
                job_id=job_config.job_id or job_id,
                job_name=job_config.job_name,
                target_url=job_config.target_url,
                target_method=job_config.target_method,
                request_data=job_config.request_data,
                headers=job_config.headers
            ))
            logger.info(f"任务立即执行: {job_id}")
            return True
        except Exception as e:
            logger.error(f"立即执行任务失败: {job_id}, 错误: {e}")
            return False

    def start(self):
        """启动调度器"""
        if not self.scheduler.running:
            self.scheduler.add_listener(self._on_job_executed, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
            self.scheduler.start()
            logger.info(f"Scheduler started with {len(self._jobs)} jobs")

    def _on_job_executed(self, event):
        """任务执行完成事件回调"""
        job_id = event.job_id
        job_config = self._jobs.get(job_id)
        if job_config and job_config.job_type == JobType.DATE:
            self._jobs.pop(job_id, None)
            logger.info(f"Date 任务执行完成，自动清理: {job_id}")

    def shutdown(self, wait: bool = True):
        """关闭调度器"""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=wait)
            logger.info("Scheduler shutdown")

    # ========================================================================
    # 便捷方法
    # ========================================================================

    def create_reminder_job(
        self,
        job_name: str,
        target_url: str,
        crontab: str,
        request_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> str:
        """
        创建定时提醒任务

        Args:
            job_name: 任务名称
            target_url: 提醒目标 URL
            crontab: 5位 cron 表达式
            request_data: 请求数据
            headers: 请求头

        Returns:
            str: 任务ID
        """
        job_config = JobConfig(
            job_name=job_name,
            job_type=JobType.CRON,
            target_url=target_url,
            crontab=crontab,
            request_data=request_data,
            headers=headers
        )
        return self.add_job(job_config)

    def create_interval_job(
        self,
        job_name: str,
        target_url: str,
        interval_seconds: Optional[int] = None,
        interval_minutes: Optional[int] = None,
        interval_hours: Optional[int] = None,
        request_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> str:
        """
        创建间隔循环任务

        Args:
            job_name: 任务名称
            target_url: 目标 URL
            interval_seconds/minutes/hours: 间隔时间
            request_data: 请求数据
            headers: 请求头

        Returns:
            str: 任务ID
        """
        job_config = JobConfig(
            job_name=job_name,
            job_type=JobType.INTERVAL,
            target_url=target_url,
            interval_seconds=interval_seconds,
            interval_minutes=interval_minutes,
            interval_hours=interval_hours,
            request_data=request_data,
            headers=headers
        )
        return self.add_job(job_config)

    def create_delayed_job(
        self,
        job_name: str,
        target_url: str,
        delay_seconds: int,
        request_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> str:
        """
        创建延迟任务（在指定秒数后执行一次）

        Args:
            job_name: 任务名称
            target_url: 目标 URL
            delay_seconds: 延迟秒数
            request_data: 请求数据
            headers: 请求头

        Returns:
            str: 任务ID
        """
        run_date = (datetime.now() + timedelta(seconds=delay_seconds)).isoformat()

        job_config = JobConfig(
            job_name=job_name,
            job_type=JobType.DATE,
            target_url=target_url,
            run_date=run_date,
            request_data=request_data,
            headers=headers
        )
        return self.add_job(job_config)


# ============================================================================
# 全局调度器实例
# ============================================================================

scheduler_manager = SchedulerManager()