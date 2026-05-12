"""
Scheduler 模块测试用例
全面测试调度服务的各项功能，包括 API 接口和实际任务触发执行
"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Optional
import httpx
from aiohttp import web
from loguru import logger


SCHEDULER_BASE_URL = "http://localhost:7702"
TEST_TIMEOUT = 30
CALLBACK_SERVER_PORT = 8899


# ============================================================================
# 本地回调服务器 - 用于接收任务执行结果
# ============================================================================

class CallbackServer:
    """本地回调服务器，接收任务执行的 HTTP 请求"""

    def __init__(self, port: int = CALLBACK_SERVER_PORT):
        self.port = port
        self.app = web.Application()
        self.handler = None
        self.runner = None
        self.received_requests: List[dict] = []
        self.request_event = asyncio.Event()

        self.app.router.add_route("POST", "/callback", self.handle_callback)
        self.app.router.add_route("GET", "/callback", self.handle_callback)
        self.app.router.add_route("DELETE", "/callback", self.handle_callback)

    async def handle_callback(self, request):
        """处理回调请求"""
        try:
            data = await request.json()
        except:
            data = {}

        logger.info(f"======> 回调服务器收到请求: {request.method} {request.path}, 数据: {data}")

        query_params = dict(request.query)
        headers = dict(request.headers)

        request_info = {
            "method": request.method,
            "path": request.path,
            "data": data,
            "query": query_params,
            "headers": {k: v for k, v in headers.items() if k.lower() not in ["host", "content-length"]},
            "timestamp": datetime.now().isoformat()
        }

        self.received_requests.append(request_info)
        logger.info(f"回调服务器收到请求: {request.method} {request.path}, 数据: {data}")

        self.request_event.set()
        return web.Response(text="OK", status=200)

    async def start(self):
        """启动服务器"""
        self.handler = web.AppRunner(self.app)
        await self.handler.setup()
        self.site = web.TCPSite(self.handler, "localhost", self.port)
        await self.site.start()
        logger.info(f"回调服务器已启动: http://localhost:{self.port}")

    async def stop(self):
        """停止服务器"""
        if self.handler:
            await self.handler.cleanup()
        logger.info("回调服务器已停止")

    def reset(self):
        """重置接收记录"""
        self.received_requests.clear()
        self.request_event.clear()

    async def wait_for_request(self, timeout: float = 15) -> Optional[dict]:
        """等待接收请求"""
        try:
            await asyncio.wait_for(self.request_event.wait(), timeout=timeout)
            self.request_event.clear()
            if self.received_requests:
                return self.received_requests[-1]
        except asyncio.TimeoutError:
            logger.warning(f"等待回调请求超时 ({timeout}s)")
        return None


# ============================================================================
# 辅助函数
# ============================================================================

def print_section(title: str):
    """打印测试标题"""
    logger.info(f"{'='*60}")
    logger.info(f"  {title}")
    logger.info(f"{'='*60}")


def print_result(name: str, success: bool, detail: str = ""):
    """打印测试结果"""
    status = "✅ PASS" if success else "❌ FAIL"
    logger.info(f"{status} | {name}")
    if detail:
        logger.info(f"      详情: {detail}\n\n")


async def wait_for_scheduler():
    """等待调度器启动"""
    logger.info("等待 Scheduler 服务启动...")
    max_retries = 10
    for i in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{SCHEDULER_BASE_URL}/v1/scheduler/health")
                if response.status_code == 200:
                    logger.info("Scheduler 服务已就绪！")
                    return True
        except Exception:
            pass
        await asyncio.sleep(1)
    logger.info("Scheduler 服务启动超时！")
    return False


# ============================================================================
# 测试用例：API 接口测试
# ============================================================================

async def test_health_check():
    """测试 1: 健康检查"""
    print_section("测试 1: 健康检查")

    async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
        response = await client.get(f"{SCHEDULER_BASE_URL}/v1/scheduler/health")
        data = response.json()

        success = (
            response.status_code == 200 and
            data.get("status") == "healthy" and
            data.get("service") == "scheduler"
        )
        print_result("健康检查", success, data)
        return success


async def test_create_cron_job():
    """测试 2: 创建 Cron 任务"""
    print_section("测试 2: 创建 Cron 任务")

    async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
        payload = {
            "job_name": "测试-Cron任务",
            "job_type": "cron",
            "target_url": "http://httpbin.org/post",
            "target_method": "POST",
            "crontab": "0 9 * * *",
            "request_data": {"test": "cron_job"}
        }

        response = await client.post(
            f"{SCHEDULER_BASE_URL}/v1/scheduler/jobs",
            json=payload
        )
        data = response.json()

        success = (
            response.status_code == 200 and
            data.get("success") and
            data.get("job_id") is not None
        )
        print_result("创建 Cron 任务", success, data)
        return data.get("job_id") if success else None


async def test_create_interval_job():
    """测试 3: 创建 Interval 任务"""
    print_section("测试 3: 创建 Interval 任务")

    async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
        payload = {
            "job_name": "测试-Interval任务",
            "job_type": "interval",
            "target_url": "http://httpbin.org/post",
            "interval_seconds": 60,
            "request_data": {"test": "interval_job"}
        }

        response = await client.post(
            f"{SCHEDULER_BASE_URL}/v1/scheduler/jobs",
            json=payload
        )
        data = response.json()

        success = (
            response.status_code == 200 and
            data.get("success") and
            data.get("job_id") is not None
        )
        print_result("创建 Interval 任务", success, data)
        return data.get("job_id") if success else None


async def test_create_date_job():
    """测试 4: 创建 Date 任务（延迟任务）"""
    print_section("测试 4: 创建 Date 任务（延迟任务）")

    run_date = (datetime.now() + timedelta(seconds=10)).isoformat()

    async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
        payload = {
            "job_name": "测试-延迟任务",
            "job_type": "date",
            "target_url": "http://httpbin.org/post",
            "run_date": run_date,
            "request_data": {"test": "date_job"}
        }

        response = await client.post(
            f"{SCHEDULER_BASE_URL}/v1/scheduler/jobs",
            json=payload
        )
        data = response.json()

        success = (
            response.status_code == 200 and
            data.get("success") and
            data.get("job_id") is not None
        )
        print_result("创建延迟任务", success, data)
        return data.get("job_id") if success else None


async def test_list_jobs():
    """测试 5: 列出所有任务"""
    print_section("测试 5: 列出所有任务")

    async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
        response = await client.get(f"{SCHEDULER_BASE_URL}/v1/scheduler/jobs")
        data = response.json()

        success = (
            response.status_code == 200 and
            data.get("success") and
            isinstance(data.get("jobs"), list)
        )
        print_result("列出所有任务", success, f"共 {data.get('total', 0)} 个任务")
        return data.get("jobs", []) if success else []


async def test_get_job(job_id: str):
    """测试 6: 获取单个任务"""
    print_section("测试 6: 获取单个任务")

    async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
        response = await client.get(f"{SCHEDULER_BASE_URL}/v1/scheduler/jobs/{job_id}")
        data = response.json()

        success = (
            response.status_code == 200 and
            data.get("success") and
            data.get("job_info", {}).get("job_id") == job_id
        )
        print_result("获取单个任务", success, data)
        return success


async def test_run_job_now(job_id: str):
    """测试 7: 立即执行任务"""
    print_section("测试 7: 立即执行任务")

    async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
        response = await client.post(
            f"{SCHEDULER_BASE_URL}/v1/scheduler/jobs/{job_id}/run"
        )
        data = response.json()

        success = (
            response.status_code == 200 and
            data.get("success")
        )
        print_result("立即执行任务", success, data)
        return success


async def test_pause_resume_job(job_id: str):
    """测试 8: 暂停和恢复任务"""
    print_section("测试 8: 暂停和恢复任务")

    async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
        response1 = await client.post(
            f"{SCHEDULER_BASE_URL}/v1/scheduler/jobs/{job_id}/pause"
        )
        success1 = response1.status_code == 200 and response1.json().get("success")
        print_result("暂停任务", success1)

        response2 = await client.post(
            f"{SCHEDULER_BASE_URL}/v1/scheduler/jobs/{job_id}/resume"
        )
        success2 = response2.status_code == 200 and response2.json().get("success")
        print_result("恢复任务", success2)

        return success1 and success2


async def test_delete_job(job_id: str):
    """测试 9: 删除任务"""
    print_section("测试 9: 删除任务")

    async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
        response = await client.delete(
            f"{SCHEDULER_BASE_URL}/v1/scheduler/jobs/{job_id}"
        )
        data = response.json()

        success = response.status_code == 200 and data.get("success")
        print_result("删除任务", success, data)
        return success


# ============================================================================
# 测试用例：实际任务触发执行测试
# ============================================================================

async def test_immediate_job_execution(callback_server: CallbackServer):
    """测试 10: 立即执行任务并验证实际 HTTP 请求"""
    print_section("测试 10: 立即执行任务并验证实际 HTTP 请求")

    callback_server.reset()

    async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
        payload = {
            "job_name": "测试-立即执行回调",
            "job_type": "date",
            "target_url": f"http://localhost:{callback_server.port}/callback",
            "target_method": "POST",
            "run_date": (datetime.now() + timedelta(seconds=300)).isoformat(),
            "request_data": {"test": "immediate_execution", "timestamp": datetime.now().isoformat()}
        }

        # 创建立即执行任务
        response = await client.post(
            f"{SCHEDULER_BASE_URL}/v1/scheduler/jobs",
            json=payload
        )
        data = response.json()

        if not (response.status_code == 200 and data.get("success")):
            print_result("创建立即执行任务", False, data)
            return False

        job_id = data.get("job_id")
        logger.info(f"任务已创建: {job_id}，调用 /run 立即执行...")

        response = await client.post(f"{SCHEDULER_BASE_URL}/v1/scheduler/jobs/{job_id}/run")
        run_result = response.json()
        logger.info(f"/run 响应: {run_result}")

        received = await callback_server.wait_for_request(timeout=10)

        if received:
            logger.info(f"收到回调请求: {received}")
            success = (
                received.get("method") == "POST" and
                received.get("data", {}).get("test") == "immediate_execution"
            )
            print_result("立即执行任务实际触发", success, received)
        else:
            success = False
            print_result("立即执行任务实际触发", False, "未收到回调请求")

        return success


async def test_interval_job_execution(callback_server: CallbackServer):
    """测试 11: Interval 任务实际触发测试"""
    print_section("测试 11: Interval 任务实际触发测试")

    callback_server.reset()

    async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
        payload = {
            "job_name": "测试-Interval回调",
            "job_type": "interval",
            "target_url": f"http://localhost:{callback_server.port}/callback",
            "interval_seconds": 5,
            "request_data": {"test": "interval_execution"}
        }

        # 创建 Interval 任务
        response = await client.post(
            f"{SCHEDULER_BASE_URL}/v1/scheduler/jobs",
            json=payload
        )
        data = response.json()

        if not (response.status_code == 200 and data.get("success")):
            print_result("创建 Interval 任务", False, data)
            return False

        job_id = data.get("job_id")
        logger.info(f"Interval 任务已创建: {job_id}，等待触发（每5秒执行一次）...")

        received1 = await callback_server.wait_for_request(timeout=10)

        if received1:
            logger.info(f"第1次触发: {received1}")
            success1 = received1.get("data", {}).get("test") == "interval_execution"
            print_result("Interval 任务第1次触发", success1)
        else:
            print_result("Interval 任务第1次触发", False, "未收到回调")
            success1 = False

        await client.delete(f"{SCHEDULER_BASE_URL}/v1/scheduler/jobs/{job_id}")
        return success1


async def test_delayed_job_execution(callback_server: CallbackServer):
    """测试 12: 延迟任务实际执行测试"""
    print_section("测试 12: 延迟任务实际执行测试")

    callback_server.reset()

    async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
        payload = {
            "job_name": "测试-延迟回调",
            "job_type": "date",
            "target_url": f"http://localhost:{callback_server.port}/callback",
            "run_date": (datetime.now() + timedelta(seconds=5)).isoformat(),
            "request_data": {"test": "delayed_execution", "delay": "5s"}
        }

        response = await client.post(
            f"{SCHEDULER_BASE_URL}/v1/scheduler/jobs",
            json=payload
        )
        data = response.json()

        if not (response.status_code == 200 and data.get("success")):
            print_result("创建延迟任务", False, data)
            return False

        job_id = data.get("job_id")
        logger.info(f"延迟任务已创建: {job_id}，5秒后执行...")

        start_time = time.time()
        received = await callback_server.wait_for_request(timeout=15)
        elapsed = time.time() - start_time

        if received:
            logger.info(f"收到延迟回调请求: {received}")
            logger.info(f"延迟执行耗时: {elapsed:.2f}秒")

            success = (
                received.get("data", {}).get("test") == "delayed_execution" and
                4 <= elapsed <= 8
            )
            print_result("延迟任务实际执行", success, f"耗时 {elapsed:.2f}秒")
        else:
            success = False
            print_result("延迟任务实际执行", False, "未收到回调请求")

        return success


async def test_http_methods_execution(callback_server: CallbackServer):
    """测试 13: 不同 HTTP 方法实际执行测试"""
    print_section("测试 13: 不同 HTTP 方法实际执行测试")

    methods = ["POST", "GET", "DELETE"]
    results = []

    for method in methods:
        callback_server.reset()

        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            payload = {
                "job_name": f"测试-{method}方法回调",
                "job_type": "date",
                "target_url": f"http://localhost:{callback_server.port}/callback",
                "target_method": method,
                "run_date": (datetime.now() + timedelta(seconds=3)).isoformat(),
                "request_data": {"method": method, "test": "http_method"}
            }

            response = await client.post(
                f"{SCHEDULER_BASE_URL}/v1/scheduler/jobs",
                json=payload
            )
            data = response.json()
            logger.info(f"测试 13:===HTTP {method} 创建响应: {data}")
            if response.status_code != 200 or not data.get("success"):
                print_result(f"HTTP {method} 创建", False)
                results.append((method, False))
                continue

            job_id = data.get("job_id")
            received = await callback_server.wait_for_request(timeout=8)

            if received:
                success = received.get("method") == method
                logger.info(f"HTTP {method} 执行结果: {success}, 收到: {received.get('method')}")
                results.append((method, success))
            else:
                results.append((method, False))

    for method, success in results:
        print_result(f"HTTP {method} 方法实际执行", success)

    return all(success for _, success in results)


async def test_convenience_reminder(callback_server: CallbackServer):
    """测试 14: 便捷定时提醒实际触发测试"""
    print_section("测试 14: 便捷定时提醒实际触发测试")

    callback_server.reset()

    now = datetime.now()
    reminder_time = now + timedelta(minutes=1)
    crontab = f"{reminder_time.minute} {reminder_time.hour} * * *"

    logger.info(f"创建定时提醒，触发时间: {crontab} (约1分钟后)")

    async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
        payload = {
            "job_name": "测试-定时提醒",
            "target_url": f"http://localhost:{callback_server.port}/callback",
            "crontab": crontab,
            "request_data": {"type": "reminder", "content": "测试提醒"}
        }

        response = await client.post(
            f"{SCHEDULER_BASE_URL}/v1/scheduler/jobs/reminder",
            json=payload
        )
        data = response.json()

        if response.status_code != 200 or not data.get("success"):
            print_result("创建定时提醒", False, data)
            return False

        job_id = data.get("job_id")
        print_result("创建定时提醒", True, f"job_id: {job_id}, crontab: {crontab}")

        logger.info(f"等待定时提醒触发...")
        received = await callback_server.wait_for_request(timeout=90)

        if received:
            success = received.get("data", {}).get("type") == "reminder"
            print_result("定时提醒实际触发", success, received)
        else:
            success = False
            print_result("定时提醒实际触发", False, "未收到回调")

        await client.delete(f"{SCHEDULER_BASE_URL}/v1/scheduler/jobs/{job_id}")
        return success


# ============================================================================
# 测试用例：边界情况测试
# ============================================================================

async def test_invalid_cron():
    """测试 15: 无效 Cron 表达式"""
    print_section("测试 15: 无效 Cron 表达式")

    async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
        payload = {
            "job_name": "测试-无效Cron",
            "job_type": "cron",
            "target_url": "http://httpbin.org/post",
            "crontab": "invalid"
        }

        response = await client.post(
            f"{SCHEDULER_BASE_URL}/v1/scheduler/jobs",
            json=payload
        )

        success = response.status_code == 400
        print_result("无效 Cron 表达式校验", success, f"状态码: {response.status_code}")
        return success


async def test_missing_required_fields():
    """测试 16: 缺少必填字段"""
    print_section("测试 16: 缺少必填字段")

    async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
        payload = {
            "job_type": "cron"
        }

        response = await client.post(
            f"{SCHEDULER_BASE_URL}/v1/scheduler/jobs",
            json=payload
        )

        success = response.status_code == 422
        print_result("必填字段校验", success, f"状态码: {response.status_code}")
        return success


async def test_nonexistent_job():
    """测试 17: 获取不存在的任务"""
    print_section("测试 17: 获取不存在的任务")

    async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
        response = await client.get(
            f"{SCHEDULER_BASE_URL}/v1/scheduler/jobs/nonexistent_job_id"
        )

        success = response.status_code == 404
        print_result("不存在的任务返回404", success, f"状态码: {response.status_code}")
        return success


async def test_concurrent_job_creation():
    """测试 18: 并发创建任务"""
    print_section("测试 18: 并发创建任务")

    async def create_test_job(index: int):
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            payload = {
                "job_name": f"并发测试任务-{index}",
                "job_type": "date",
                "target_url": "http://httpbin.org/post",
                "run_date": (datetime.now() + timedelta(seconds=300)).isoformat()
            }
            response = await client.post(
                f"{SCHEDULER_BASE_URL}/v1/scheduler/jobs",
                json=payload
            )
            return response.status_code == 200 and response.json().get("success")

    tasks = [create_test_job(i) for i in range(5)]
    results = await asyncio.gather(*tasks)

    success = all(results)
    print_result("并发创建5个任务", success, f"成功: {sum(results)}/5")
    return success


# ============================================================================
# 主测试函数
# ============================================================================

async def run_all_tests():
    """运行所有测试"""
    logger.info(f"\n{'='*60}")
    logger.info("  wlSuperAgent Scheduler 全面测试")
    logger.info(f"{'='*60}")

    if not await wait_for_scheduler():
        logger.info("Scheduler 服务未就绪，测试终止")
        return

    callback_server = CallbackServer()
    await callback_server.start()

    test_results = []
    created_job_ids = []

    try:
        test_results.append(("健康检查", await test_health_check()))

        if 1:
            cron_job_id = await test_create_cron_job()
            test_results.append(("创建 Cron 任务", cron_job_id is not None))
            if cron_job_id:
                created_job_ids.append(cron_job_id)

            interval_job_id = await test_create_interval_job()
            test_results.append(("创建 Interval 任务", interval_job_id is not None))
            if interval_job_id:
                created_job_ids.append(interval_job_id)

            date_job_id = await test_create_date_job()
            test_results.append(("创建 Date 任务", date_job_id is not None))
            if date_job_id:
                created_job_ids.append(date_job_id)

            jobs = await test_list_jobs()
            test_results.append(("列出所有任务", len(jobs) >= 3))

            if cron_job_id:
                test_results.append(("获取单个任务", await test_get_job(cron_job_id)))
                test_results.append(("暂停/恢复任务", await test_pause_resume_job(cron_job_id)))
                created_job_ids.remove(cron_job_id)

            test_results.append(("删除任务", await test_delete_job(date_job_id)))

            test_results.append(("立即执行任务 API", await test_run_job_now(interval_job_id) if interval_job_id else False))

            test_results.append(("无效 Cron 校验", await test_invalid_cron()))
            test_results.append(("必填字段校验", await test_missing_required_fields()))
            test_results.append(("不存在的任务", await test_nonexistent_job()))

            test_results.append(("并发创建任务", await test_concurrent_job_creation()))

            logger.info("="*60)
            logger.info("  实际触发执行测试")
            logger.info("="*60 + "\n")

            test_results.append(("立即执行实际触发", await test_immediate_job_execution(callback_server)))
            test_results.append(("Interval 任务实际触发", await test_interval_job_execution(callback_server)))
            test_results.append(("延迟任务实际执行", await test_delayed_job_execution(callback_server)))
            test_results.append(("HTTP 方法实际执行", await test_http_methods_execution(callback_server)))
        test_results.append(("定时提醒实际触发", await test_convenience_reminder(callback_server)))

        logger.info("\n" + "="*60)
        logger.info("  清理测试任务")
        logger.info("="*60 + "\n")

        async with httpx.AsyncClient(timeout=10) as client:
            for job_id in created_job_ids:
                await client.delete(f"{SCHEDULER_BASE_URL}/v1/scheduler/jobs/{job_id}")
            logger.info(f"已清理 {len(created_job_ids)} 个测试任务")

    finally:
        await callback_server.stop()

    print_section("测试报告")
    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)

    logger.info(f"总测试数: {total}")
    logger.info(f"通过: {passed} ✅")
    logger.info(f"失败: {total - passed} ❌")
    logger.info(f"通过率: {passed/total*100:.1f}%")

    logger.info("详细结果:")
    for name, success in test_results:
        status = "✅" if success else "❌"
        logger.info(f"  {status} {name}")

    return passed == total


if __name__ == "__main__":
    """
    使用方式:

    1. 先启动 Scheduler 服务:
       python scheduler_main.py

    2. 再运行测试:
       python Scheduler_example.py

    3. 如需单独测试某项功能，可导入对应函数:
       from Scheduler_example import test_health_check
       asyncio.run(test_health_check())
    """
    logger.info("wlSuperAgent Scheduler 测试用例")
    logger.info("请确保 Scheduler 服务已启动 (python scheduler_main.py)")
    logger.info("")

    result = asyncio.run(run_all_tests())

    if result:
        logger.info("🎉 所有测试通过！")
    else:
        logger.info("⚠️ 部分测试失败，请检查日志")

    exit(0 if result else 1)
