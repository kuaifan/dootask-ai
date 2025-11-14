# 标准库导入
import asyncio
import logging

# 第三方库导入
import httpx
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager

# 本地模块导入
from helper.redis import RedisManager
from helper.mcp import ensure_dootask_mcp_config
from helper.config import MCP_HEALTH_URL, MCP_CHECK_INTERVAL

# 日志配置
logger = logging.getLogger("ai")
logging.getLogger("httpx").setLevel(logging.WARNING)

async def check_mcp_health(app: FastAPI) -> None:
    """检查 MCP 服务的健康状态并将结果写入 app.state.mcp。"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(MCP_HEALTH_URL, timeout=3)
            is_ok = response.json().get("status") == "ok"
            app.state.dootask_mcp = is_ok
            if is_ok:
                ensure_dootask_mcp_config(enabled=True)
    except Exception as exc:  # pragma: no cover - best effort external check
        app.state.dootask_mcp = False
        logger.error(f"❌ 检测 MCP 失败: {MCP_HEALTH_URL} - 错误: {exc}")


async def periodic_mcp_check(app: FastAPI, interval: int = MCP_CHECK_INTERVAL) -> None:
    """每隔 interval 秒轮询 MCP 健康状态。"""
    while True:
        await check_mcp_health(app)
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan_context(app: FastAPI):
    """FastAPI 生命周期钩子，负责启动/停止 Redis 和周期任务。"""
    task = None
    try:
        task = asyncio.create_task(periodic_mcp_check(app))
        redis_manager = RedisManager()
        app.state.redis_manager = redis_manager
        logger.info("✅ 初始化成功")
    except Exception as exc:
        logger.info(f"❌ 初始化失败: {str(exc)}")
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info("✅ 定时任务已停止")
        logger.info("🛑 AI服务正在关闭...")
