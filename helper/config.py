import os
import re
from pathlib import Path

# 路径与基础配置
BASE_DIR = Path(__file__).resolve().parent.parent

# 服务启动端口
SERVER_PORT = int(os.environ.get('PORT', 5001))

# UI 静态资源路径
UI_DIST_PATH = BASE_DIR / "static" / "ui"

# 清空上下文的命令
CLEAR_COMMANDS = [":clear", ":reset", ":restart", ":new", ":清空上下文", ":重置上下文", ":重启", ":重启对话"]

# 流式响应超时时间
STREAM_TIMEOUT = 300

# MCP 服务器相关配置
MCP_SERVER_URL = "http://nginx/apps/mcp_server"
MCP_STREAM_URL = MCP_SERVER_URL + "/mcp"
MCP_HEALTH_URL = MCP_SERVER_URL + "/healthz"
MCP_CHECK_INTERVAL = 60  # 检查间隔，单位秒

# MCP 配置文件路径及默认名称
MCP_CONFIG_PATH = BASE_DIR / "config" / "mcp-config.json"
DOOTASK_MCP_NAME = "DooTask MCP"
DOOTASK_MCP_ID = "dootask-mcp"

# Vision 配置文件路径
VISION_CONFIG_PATH = BASE_DIR / "config" / "vision-config.json"
VISION_DATA_DIR = BASE_DIR / "data" / "vision"
VISION_PREVIEW_URL_PREFIX = "http://nginx/ai/vision/preview"
VISION_CLEANUP_DAYS = 7
VISION_CLEANUP_INTERVAL = 86400  # 24 hours in seconds

# LangChain 思考标记正则
THINK_START_PATTERN = re.compile(r'<think>\s*')
THINK_END_PATTERN = re.compile(r'\s*</think>')
REASONING_PATTERN = re.compile(r'::: reasoning\n.*?:::', re.DOTALL)

# 工具调用标记的正则模式
TOOL_CALL_PATTERN = re.compile(r'\n?> <tool-use>Tool: [^<]+</tool-use>\n*')

# 默认模型列表
DEFAULT_MODELS = {
    "openai": [
        {"id": "gpt-5.5", "name": "GPT-5.5", "support_mcp": True, "support_vision": True},
        {"id": "gpt-5.5-pro", "name": "GPT-5.5 Pro", "support_mcp": True, "support_vision": True},
        {"id": "gpt-5.4", "name": "GPT-5.4", "support_mcp": True, "support_vision": True},
        {"id": "gpt-5.4-pro", "name": "GPT-5.4 Pro", "support_mcp": True, "support_vision": True},
        {"id": "gpt-5.4-mini", "name": "GPT-5.4 Mini", "support_mcp": True, "support_vision": True},
        {"id": "gpt-5.4-nano", "name": "GPT-5.4 Nano", "support_mcp": True, "support_vision": True},
        {"id": "gpt-5.3-codex", "name": "GPT-5.3 Codex", "support_mcp": True, "support_vision": True},
        {"id": "gpt-5.2", "name": "GPT-5.2", "support_mcp": True, "support_vision": True},
        {"id": "o3-pro", "name": "o3 Pro", "support_mcp": True, "support_vision": True},
        {"id": "o3", "name": "o3", "support_mcp": True, "support_vision": True},
        {"id": "o4-mini", "name": "o4 Mini", "support_mcp": True, "support_vision": True},
    ],
    "claude": [
        {"id": "claude-opus-4-7 (thinking)", "name": "Claude Opus 4.7", "support_mcp": True, "support_vision": True},
        {"id": "claude-opus-4-6 (thinking)", "name": "Claude Opus 4.6", "support_mcp": True, "support_vision": True},
        {"id": "claude-sonnet-4-6 (thinking)", "name": "Claude Sonnet 4.6", "support_mcp": True, "support_vision": True},
        {"id": "claude-haiku-4-5 (thinking)", "name": "Claude Haiku 4.5", "support_mcp": True, "support_vision": True},
    ],
    "deepseek": [
        {"id": "deepseek-v4-pro", "name": "DeepSeek V4 Pro", "support_mcp": True, "support_vision": False},
        {"id": "deepseek-v4-flash", "name": "DeepSeek V4 Flash", "support_mcp": True, "support_vision": False},
    ],
    "gemini": [
        {"id": "gemini-3.1-pro-preview", "name": "Gemini 3.1 Pro", "support_mcp": True, "support_vision": True},
        {"id": "gemini-3.1-flash-lite-preview", "name": "Gemini 3.1 Flash Lite", "support_mcp": True, "support_vision": True},
        {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "support_mcp": True, "support_vision": True},
        {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "support_mcp": True, "support_vision": True},
    ],
    "grok": [
        {"id": "grok-4.20-0309-reasoning", "name": "Grok 4.20 Reasoning", "support_mcp": True, "support_vision": True},
        {"id": "grok-4.20-0309-non-reasoning", "name": "Grok 4.20", "support_mcp": True, "support_vision": True},
        {"id": "grok-4-1-fast-reasoning", "name": "Grok 4.1 Fast Reasoning", "support_mcp": True, "support_vision": True},
        {"id": "grok-4-1-fast-non-reasoning", "name": "Grok 4.1 Fast", "support_mcp": True, "support_vision": True},
    ],
    "zhipu": [
        {"id": "glm-5.1", "name": "GLM-5.1", "support_mcp": True, "support_vision": False},
        {"id": "glm-5", "name": "GLM-5", "support_mcp": True, "support_vision": False},
        {"id": "glm-5-turbo", "name": "GLM-5 Turbo", "support_mcp": True, "support_vision": False},
        {"id": "glm-5v-turbo", "name": "GLM-5V Turbo", "support_mcp": True, "support_vision": True},
        {"id": "glm-4.7", "name": "GLM-4.7", "support_mcp": True, "support_vision": False},
    ],
    "qianwen": [
        {"id": "qwen3.6-max-preview", "name": "Qwen3.6 Max", "support_mcp": True, "support_vision": False},
        {"id": "qwen3.6-plus", "name": "Qwen3.6 Plus", "support_mcp": True, "support_vision": True},
        {"id": "qwen3.6-flash", "name": "Qwen3.6 Flash", "support_mcp": True, "support_vision": False},
        {"id": "qwen3-max", "name": "Qwen3 Max", "support_mcp": True, "support_vision": False},
        {"id": "qwen-plus", "name": "Qwen Plus", "support_mcp": True, "support_vision": False},
        {"id": "qwen-turbo", "name": "Qwen Turbo", "support_mcp": True, "support_vision": False},
    ],
    "wenxin": [
        {"id": "ernie-x1.1", "name": "ERNIE X1.1", "support_mcp": False, "support_vision": False},
        {"id": "ernie-5.0-thinking-preview", "name": "ERNIE 5.0 Thinking", "support_mcp": False, "support_vision": True},
        {"id": "ernie-4.5-turbo-128k", "name": "ERNIE 4.5 Turbo 128K", "support_mcp": False, "support_vision": False},
        {"id": "ernie-4.5-turbo-vl-32k", "name": "ERNIE 4.5 Turbo VL 32K", "support_mcp": False, "support_vision": True},
    ],
}

# 模型上下文限制（token数）
# 原则：只配置明确知道的，不知道的使用最小值
# 数值为官方文档的原始值
CONTEXT_LIMITS = {
    "openai": {
        # GPT-5.5: 1M context
        "gpt-5.5": 1050000,
        "gpt-5.5-pro": 1050000,
        # GPT-5.4: 1M context
        "gpt-5.4": 1050000,
        "gpt-5.4-pro": 1050000,
        "gpt-5.4-mini": 1050000,
        "gpt-5.4-nano": 1050000,
        "gpt-5.3-codex": 1050000,
        "gpt-5.2": 128000,
        # o 系列: 200K context
        "o3-pro": 200000,
        "o3": 200000,
        "o4-mini": 200000,
        "default": 128000,
    },
    "claude": {
        # Claude 4.6: 1M context
        "default": 1000000,
    },
    "deepseek": {
        # DeepSeek V4: 1M context
        "deepseek-v4-pro": 1000000,
        "deepseek-v4-flash": 1000000,
        "default": 128000,
    },
    "gemini": {
        # Gemini 3.x/2.5: ~1M context
        "default": 1000000,
    },
    "grok": {
        # Grok 4.x: 2M context
        "default": 2000000,
    },
    "zhipu": {
        # GLM-5.x: 200K context
        "glm-5.1": 200000,
        "glm-5": 200000,
        "glm-5-turbo": 200000,
        "glm-5v-turbo": 200000,
        "glm-4.7": 128000,
        "default": 128000,
    },
    "qianwen": {
        "qwen3-max": 128000,
        "qwen-plus": 32000,
        "qwen-turbo": 32000,
        "default": 32000,
    },
    "wenxin": {
        "ernie-5.0-thinking-preview": 128000,
        "ernie-4.5-turbo-128k": 128000,
        "ernie-4.5-turbo-vl-32k": 32000,
        "default": 32000,
    },
}
