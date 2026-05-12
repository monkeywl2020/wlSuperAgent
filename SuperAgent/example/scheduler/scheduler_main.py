"""
Scheduler 服务入口 - 任务调度服务
提供定时任务、延迟任务、周期任务的调度管理

启动方式：
1. 独立启动：python scheduler_main.py（直接运行）
2. 作为子进程启动：由 SuperAgent 通过 subprocess.Popen 拉起

Scheduler 本质是独立进程，与 SuperAgent 通过 HTTP 端口通信，互不影响
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

sys.path.append(os.path.dirname(__file__))

from scheduler import (
    JobConfig,
    JobInfo,
    JobResponse,
    JobListResponse,
    JobType,
    scheduler_manager,
)

from loguru import logger
# ============================================================================
# 日志配置
# ============================================================================
file_log_enable = False
print(f"===========file_log_enable:{file_log_enable}", flush=True)
if file_log_enable:
    log_dir = os.path.join("./", "log")
    log_path = os.path.join(log_dir, "nuwa_main_log.log")
    logger.remove()
    logger.add(log_path, rotation="10 MB", retention=30, enqueue=True)


scheduler_app: Optional[FastAPI] = None

# ============================================================================
# 生命周期管理
# ============================================================================
def create_scheduler_app() -> FastAPI:
    """创建 Scheduler FastAPI 应用（不自动启动）"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Scheduler 服务启动中...")
        scheduler_manager.start()
        yield
        logger.info("Scheduler 服务关闭中...")
        scheduler_manager.shutdown()

    app = FastAPI(
        title="wlSuperAgent Scheduler",
        description="任务调度服务 - 提供定时任务、延迟任务、周期任务的调度管理",
        version="1.0.0",
        lifespan=lifespan
    )

    class CreateJobRequest(BaseModel):
        job_name: str = Field(..., description="任务名称")
        job_type: str = Field(JobType.CRON.value, description="任务类型: cron/interval/date")
        target_url: str = Field(..., description="目标 URL")
        target_method: str = Field("POST", description="HTTP 方法")
        request_data: Optional[Dict[str, Any]] = Field(None, description="请求数据")
        headers: Optional[Dict[str, str]] = Field(None, description="请求头")
        crontab: Optional[str] = Field(None, description="Cron 表达式 (分 时 日 月 周)")
        interval_seconds: Optional[int] = Field(None, description="间隔秒数")
        interval_minutes: Optional[int] = Field(None, description="间隔分钟数")
        interval_hours: Optional[int] = Field(None, description="间隔小时数")
        run_date: Optional[str] = Field(None, description="执行时间 (ISO 格式)")
        timezone: str = Field("Asia/Shanghai", description="时区")
        enabled: bool = Field(True, description="是否启用")
        max_instances: int = Field(1, description="最大并发实例数")

    #================================================
    #  apscheduler 任务类型，支持 cron/interval/date
    #    只跑一次 → date
    #    每隔X时间跑 → interval
    #    每到X点/星期X跑 → cron

    # Cron 表达式的含义
    # 一、字段结构（共 7 个，从左到右）
    #| 字段名         | 取值范围           | 说明  | 默认值 |
    #|:--------------|:-------------------|:------|:-----|
    #| `second`      | `0-59`             | 秒    | `0` |
    #| `minute`      | `0-59`             | 分    | `0` |
    #| `hour`        | `0-23`             | 时    | `0` |
    #| `day`         | `1-31`             | 日    | `*` |
    #| `month`       | `1-12` 或 `jan-dec`| 月    | `*` |
    #| `day_of_week` | `0-6` 或 `mon-sun` | 星期  | `*` |
    #| `year`        | `4位数字`（可选）   | 年    | `*` |
    # 二、特殊字符与语法
    # | 符号          | 含义             | 示例                      | 实际触发点                  |
    # |:--------------|:-----------------|:-------------------------|:---------------------------|
    # | `*`           | 任意值           | `hour='*'`                | 每小时                      |
    # | `,`           | 枚举多个         | `hour='9,12,18'`          | 9点、12点、18点             |
    # | `-`           | 连续范围         | `day_of_week='mon-fri'`   | 周一到周五                  |
    # | `/`           | 步长/间隔        | `minute='*/15'`           | 0,15,30,45 分               |
    # | `数字/步长`   | 从某值开始步进    | `second='5/10'`           | 5,15,25,35,45,55 秒         |
    #================================================
    class ConversationSummaryRequest(BaseModel):
        crontab: str = Field("0 2 * * *", description="Cron 表达式，默认每天凌晨2点")
        target_url: str = Field("http://localhost:7701/v1/internal/summary", description="摘要接口 URL")
        session_id: Optional[str] = Field(None, description="指定 session_id，不填则处理所有")
        headers: Optional[Dict[str, str]] = Field(None, description="请求头")

    class MemoryCleanupRequest(BaseModel):
        crontab: str = Field("0 3 * * *", description="Cron 表达式，默认每天凌晨3点")
        target_url: str = Field("http://localhost:7701/v1/internal/cleanup", description="清理接口 URL")
        days: int = Field(30, description="清理多少天前的数据")
        headers: Optional[Dict[str, str]] = Field(None, description="请求头")

    class ReminderRequest(BaseModel):
        crontab: str = Field(..., description="Cron 表达式")
        target_url: str = Field(..., description="提醒目标 URL")
        request_data: Optional[Dict[str, Any]] = Field(None, description="请求数据")
        headers: Optional[Dict[str, str]] = Field(None, description="请求头")

    @app.get("/v1/scheduler/health", tags=["健康检查"])
    async def health_check():
        return {
            "status": "healthy",
            "service": "scheduler",
            "version": "1.0.0",
            "jobs_count": len(scheduler_manager.list_jobs())
        }

    @app.post("/v1/scheduler/jobs", response_model=JobResponse, tags=["任务管理"])
    async def create_job(request: CreateJobRequest):
        try:
            job_type = JobType(request.job_type)
            job_config = JobConfig(
                job_name=request.job_name,
                job_type=job_type,
                target_url=request.target_url,
                target_method=request.target_method,
                request_data=request.request_data,
                headers=request.headers,
                crontab=request.crontab,
                interval_seconds=request.interval_seconds,
                interval_minutes=request.interval_minutes,
                interval_hours=request.interval_hours,
                run_date=request.run_date,
                timezone=request.timezone,
                enabled=request.enabled,
                max_instances=request.max_instances
            )
            job_id = scheduler_manager.add_job(job_config)
            job_info = scheduler_manager.get_job(job_id)
            return JobResponse(success=True, message="任务创建成功", job_id=job_id, job_info=job_info)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            logger.error(f"创建任务失败: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @app.get("/v1/scheduler/jobs", response_model=JobListResponse, tags=["任务管理"])
    async def list_jobs():
        try:
            jobs = scheduler_manager.list_jobs()
            return JobListResponse(success=True, total=len(jobs), jobs=jobs)
        except Exception as e:
            logger.error(f"列出任务失败: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @app.get("/v1/scheduler/jobs/{job_id}", response_model=JobResponse, tags=["任务管理"])
    async def get_job(job_id: str):
        job_info = scheduler_manager.get_job(job_id)
        if not job_info:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
        return JobResponse(success=True, message="获取成功", job_id=job_id, job_info=job_info)

    @app.delete("/v1/scheduler/jobs/{job_id}", tags=["任务管理"])
    async def delete_job(job_id: str):
        if scheduler_manager.remove_job(job_id):
            return {"success": True, "message": "任务已删除"}
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")

    @app.post("/v1/scheduler/jobs/{job_id}/run", tags=["任务管理"])
    async def run_job_now(job_id: str):
        if scheduler_manager.run_job(job_id):
            return {"success": True, "message": "任务已开始执行"}
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")

    @app.post("/v1/scheduler/jobs/{job_id}/pause", tags=["任务管理"])
    async def pause_job(job_id: str):
        if scheduler_manager.pause_job(job_id):
            return {"success": True, "message": "任务已暂停"}
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")

    @app.post("/v1/scheduler/jobs/{job_id}/resume", tags=["任务管理"])
    async def resume_job(job_id: str):
        if scheduler_manager.resume_job(job_id):
            return {"success": True, "message": "任务已恢复"}
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")

    @app.post("/v1/scheduler/jobs/conversation-summary", response_model=JobResponse, tags=["便捷任务"])
    async def create_conversation_summary_job(request: ConversationSummaryRequest):
        try:
            request_data = {}
            if request.session_id:
                request_data["session_id"] = request.session_id
            job_id = scheduler_manager.create_reminder_job(
                job_name="会话摘要任务",
                target_url=request.target_url,
                crontab=request.crontab,
                request_data=request_data if request_data else None,
                headers=request.headers
            )
            job_info = scheduler_manager.get_job(job_id)
            return JobResponse(success=True, message="会话摘要任务创建成功", job_id=job_id, job_info=job_info)
        except Exception as e:
            logger.error(f"创建会话摘要任务失败: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @app.post("/v1/scheduler/jobs/memory-cleanup", response_model=JobResponse, tags=["便捷任务"])
    async def create_memory_cleanup_job(request: MemoryCleanupRequest):
        try:
            job_id = scheduler_manager.create_reminder_job(
                job_name="记忆清理任务",
                target_url=request.target_url,
                crontab=request.crontab,
                request_data={"days": request.days},
                headers=request.headers
            )
            job_info = scheduler_manager.get_job(job_id)
            return JobResponse(success=True, message="记忆清理任务创建成功", job_id=job_id, job_info=job_info)
        except Exception as e:
            logger.error(f"创建记忆清理任务失败: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @app.post("/v1/scheduler/jobs/reminder", response_model=JobResponse, tags=["便捷任务"])
    async def create_reminder(request: ReminderRequest):
        try:
            job_id = scheduler_manager.create_reminder_job(
                job_name=f"定时提醒-{request.crontab}",
                target_url=request.target_url,
                crontab=request.crontab,
                request_data=request.request_data,
                headers=request.headers
            )
            job_info = scheduler_manager.get_job(job_id)
            return JobResponse(success=True, message="定时提醒创建成功", job_id=job_id, job_info=job_info)
        except Exception as e:
            logger.error(f"创建定时提醒失败: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return app


def get_scheduler_app() -> FastAPI:
    """获取或创建 Scheduler FastAPI 应用"""
    global scheduler_app
    if scheduler_app is None:
        scheduler_app = create_scheduler_app()
    return scheduler_app


def start_scheduler_standalone(port: int = 7702):
    """独立启动 Scheduler 服务"""
    app = get_scheduler_app()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    start_scheduler_standalone(port=7702)
