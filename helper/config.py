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

# LangChain 思考标记正则
THINK_START_PATTERN = re.compile(r'<think>\s*')
THINK_END_PATTERN = re.compile(r'\s*</think>')
REASONING_PATTERN = re.compile(r'::: reasoning\n.*?:::', re.DOTALL)

# 默认模型列表
DEFAULT_MODELS = {
    "openai": [
        {"id": "gpt-5.2-chat-latest", "name": "GPT-5.2 Chat", "support_mcp": True},
        {"id": "gpt-5.2-codex", "name": "GPT-5.2 Codex", "support_mcp": True},
        {"id": "gpt-5.2-pro", "name": "GPT-5.2 Pro", "support_mcp": True},
        {"id": "gpt-5.2", "name": "GPT-5.2", "support_mcp": True},

        {"id": "gpt-5.1-chat-latest", "name": "GPT-5.1 Chat", "support_mcp": True},
        {"id": "gpt-5.1-codex-mini", "name": "GPT-5.1 Codex mini", "support_mcp": True},
        {"id": "gpt-5.1-codex", "name": "GPT-5.1 Codex", "support_mcp": True},
        {"id": "gpt-5.1", "name": "GPT-5.1", "support_mcp": True},

        {"id": "gpt-5-chat-latest", "name": "GPT-5 Chat", "support_mcp": True},
        {"id": "gpt-5-codex", "name": "GPT-5 Codex", "support_mcp": True},
        {"id": "gpt-5-mini", "name": "GPT-5 mini", "support_mcp": True},
        {"id": "gpt-5-nano", "name": "GPT-5 nano", "support_mcp": True},
        {"id": "gpt-5-pro", "name": "GPT-5 pro", "support_mcp": True},
        {"id": "gpt-5", "name": "GPT-5", "support_mcp": True},

        {"id": "gpt-4.1", "name": "GPT-4.1", "support_mcp": True},
        {"id": "gpt-4o", "name": "GPT-4o", "support_mcp": True},
        {"id": "gpt-4", "name": "GPT-4", "support_mcp": True},
        {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "support_mcp": True},
        {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "support_mcp": True},

        {"id": "o3", "name": "GPT-o3", "support_mcp": True},
        {"id": "o1", "name": "GPT-o1", "support_mcp": False},
        {"id": "o4-mini", "name": "GPT-o4 Mini", "support_mcp": True},
        {"id": "o3-mini", "name": "GPT-o3 Mini", "support_mcp": True},
        {"id": "o1-mini", "name": "GPT-o1 Mini", "support_mcp": False},

        {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "support_mcp": False},
        {"id": "gpt-3.5-turbo-16k", "name": "GPT-3.5 Turbo 16K", "support_mcp": False},
        {"id": "gpt-3.5-turbo-0125", "name": "GPT-3.5 Turbo 0125", "support_mcp": False},
        {"id": "gpt-3.5-turbo-1106", "name": "GPT-3.5 Turbo 1106", "support_mcp": False},
    ],
    "claude": [
        {"id": "claude-opus-4-5 (thinking)", "name": "Claude Opus 4.5", "support_mcp": True},
        {"id": "claude-sonnet-4-5 (thinking)", "name": "Claude Sonnet 4.5", "support_mcp": True},
        {"id": "claude-haiku-4-5 (thinking)", "name": "Claude Haiku 4.5", "support_mcp": True},

        {"id": "claude-opus-4-1 (thinking)", "name": "Claude Opus 4.1", "support_mcp": True},

        {"id": "claude-sonnet-4-0 (thinking)", "name": "Claude Sonnet 4.0", "support_mcp": True},
        {"id": "claude-opus-4-0 (thinking)", "name": "Claude Opus 4.0", "support_mcp": True},

        {"id": "claude-3-7-sonnet-latest (thinking)", "name": "Claude Sonnet 3.7", "support_mcp": True},
        {"id": "claude-3-5-haiku-latest", "name": "Claude Haiku 3.5", "support_mcp": True},
    ],
    "deepseek": [
        {"id": "deepseek-chat", "name": "DeepSeek-V3.2", "support_mcp": True},
        {"id": "deepseek-reasoner", "name": "DeepSeek-V3.2-Reasoner", "support_mcp": True},
    ],
    "gemini": [
        {"id": "gemini-3-pro-preview", "name": "Gemini 3 Pro", "support_mcp": True},
        {"id": "gemini-3-flash-preview", "name": "Gemini 3 Flash", "support_mcp": True},

        {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "support_mcp": True},
        {"id": "gemini-2.5-pro-tts", "name": "Gemini 2.5 Pro TTS", "support_mcp": True},
        {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "support_mcp": True},
        {"id": "gemini-2.5-flash-tts", "name": "Gemini 2.5 Flash TTS", "support_mcp": True},
        {"id": "gemini-2.5-flash-live", "name": "Gemini 2.5 Flash Live", "support_mcp": True},
        {"id": "gemini-2.5-flash-image", "name": "Gemini 2.5 Flash Image", "support_mcp": True},
        
        {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash", "support_mcp": False},
        {"id": "gemini-2.0-flash-lite", "name": "Gemini 2.0 Flash Lite", "support_mcp": False},
    ],
    "grok": [
        {"id": "grok-4-1-fast-reasoning", "name": "Grok 4.1 Fast Reasoning", "support_mcp": True},
        {"id": "grok-4-1-fast-non-reasoning", "name": "Grok 4.1 Fast", "support_mcp": True},
        {"id": "grok-code-fast-1", "name": "Grok Code Fast 1", "support_mcp": True},
        {"id": "grok-4-fast-reasoning", "name": "Grok 4 Fast Reasoning", "support_mcp": True},
        {"id": "grok-4-fast-non-reasoning", "name": "Grok 4 Fast", "support_mcp": True},
        {"id": "grok-4-0709", "name": "Grok 4", "support_mcp": True},
        {"id": "grok-3-latest", "name": "Grok 3", "support_mcp": False},
        {"id": "grok-3-fast-latest", "name": "Grok 3 Fast", "support_mcp": False},
    ],
    "zhipu": [
        {"id": "glm-4", "name": "GLM-4", "support_mcp": True},
        {"id": "glm-4-plus", "name": "GLM-4 Plus", "support_mcp": True},
        {"id": "glm-4-air", "name": "GLM-4 Air", "support_mcp": True},
        {"id": "glm-4-airx", "name": "GLM-4 AirX", "support_mcp": True},
        {"id": "glm-4-long", "name": "GLM-4 Long", "support_mcp": True},
        {"id": "glm-4-flash", "name": "GLM-4 Flash", "support_mcp": True},
        {"id": "glm-4v", "name": "GLM-4V", "support_mcp": False},
        {"id": "glm-4v-plus", "name": "GLM-4V Plus", "support_mcp": False},
        {"id": "glm-3-turbo", "name": "GLM-3 Turbo", "support_mcp": False},
    ],
    "qianwen": [
        {"id": "qwen-max", "name": "QWEN Max", "support_mcp": True},
        {"id": "qwen-max-latest", "name": "QWEN Max Latest", "support_mcp": False},
        {"id": "qwen-turbo", "name": "QWEN Turbo", "support_mcp": False},
        {"id": "qwen-turbo-latest", "name": "QWEN Turbo Latest", "support_mcp": False},
        {"id": "qwen-plus", "name": "QWEN Plus", "support_mcp": False},
        {"id": "qwen-plus-latest", "name": "QWEN Plus Latest", "support_mcp": False},
        {"id": "qwen-long", "name": "QWEN Long", "support_mcp": False},
    ],
    "wenxin": [
        {"id": "ernie-4.5-turbo-128k", "name": "Ernie 4.5 Turbo 128K", "support_mcp": True},
        {"id": "ernie-4.5-turbo-32k", "name": "Ernie 4.5 Turbo 32K", "support_mcp": True},
        {"id": "ernie-4.5-turbo-latest", "name": "Ernie 4.5 Turbo Latest", "support_mcp": True},
        {"id": "ernie-4.5-turbo-vl", "name": "Ernie 4.5 Turbo VL", "support_mcp": True},
        {"id": "ernie-4.0-8k", "name": "Ernie 4.0 8K", "support_mcp": False},
        {"id": "ernie-4.0-8k-latest", "name": "Ernie 4.0 8K Latest", "support_mcp": False},
        {"id": "ernie-4.0-turbo-128k", "name": "Ernie 4.0 Turbo 128K", "support_mcp": False},
        {"id": "ernie-4.0-turbo-8k", "name": "Ernie 4.0 Turbo 8K", "support_mcp": False},
        {"id": "ernie-3.5-128k", "name": "Ernie 3.5 128K", "support_mcp": False},
        {"id": "ernie-3.5-8k", "name": "Ernie 3.5 8K", "support_mcp": False},
        {"id": "ernie-speed-128k", "name": "Ernie Speed 128K", "support_mcp": False},
        {"id": "ernie-speed-8k", "name": "Ernie Speed 8K", "support_mcp": False},
        {"id": "ernie-lite-8k", "name": "Ernie Lite 8K", "support_mcp": False},
        {"id": "ernie-tiny-8k", "name": "Ernie Tiny 8K", "support_mcp": False},
    ],
}

# 模型上下文限制（token数）
CONTEXT_LIMITS = {
    "openai": {
        "default": 400000
    },
    "claude": {
        "default": 200000
    },
    "deepseek": {
        "default": 128000
    },
    "gemini": {
        "default": 200000
    },
    "zhipu": {
        "default": 128000
    },
    "qwen": {
        "default": 128000
    },
    "grok": {
        "default": 128000
    }
}
