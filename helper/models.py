from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

import httpx

logger = logging.getLogger("ai")

MCP_CONFIG_PATH = Path(__file__).resolve().parent.parent / "mcp-config.json"
DOOTASK_MCP_NAME = "DooTask MCP"


class ModelInfo(TypedDict):
    """模型信息类型定义"""
    id: str
    name: str
    support_mcp: bool


DEFAULT_MODELS: Dict[str, List[ModelInfo]] = {
    "openai": [
        {"id": "gpt-5-chat-latest", "name": "GPT-5 Chat", "support_mcp": True},
        {"id": "gpt-5-codex", "name": "GPT-5-Codex", "support_mcp": True},
        {"id": "gpt-5-mini", "name": "GPT-5 mini", "support_mcp": True},
        {"id": "gpt-5-nano", "name": "GPT-5 nano", "support_mcp": True},
        {"id": "gpt-5-pro", "name": "GPT-5 pro", "support_mcp": True},
        {"id": "gpt-5.1-chat-latest", "name": "GPT-5.1 Chat", "support_mcp": True},
        {"id": "gpt-5.1-codex-mini", "name": "GPT-5.1 Codex mini", "support_mcp": True},
        {"id": "gpt-5.1-codex", "name": "GPT-5.1 Codex", "support_mcp": True},
        {"id": "gpt-5.1", "name": "GPT-5.1", "support_mcp": True},
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


class MCPConfigError(Exception):
    """Raised when MCP 配置文件读写失败。"""


def _default_mcp_config() -> Dict[str, object]:
    return {"mcps": []}


def _normalize_mcp_config(data: Dict[str, object]) -> Dict[str, object]:
    mcps = data.get("mcps")
    if not isinstance(mcps, list):
        data["mcps"] = []
    return data


def load_mcp_config_data(fallback_empty: bool = False) -> Dict[str, object]:
    """读取 MCP 配置文件。"""
    if not MCP_CONFIG_PATH.exists():
        return _default_mcp_config()

    try:
        with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as config_file:
            data = json.load(config_file)
            if not isinstance(data, dict):
                data = _default_mcp_config()
            return _normalize_mcp_config(data)
    except Exception as exc:
        if fallback_empty:
            logger.error(f"❌ 读取 MCP 配置失败: {exc}")
            return _default_mcp_config()
        raise MCPConfigError("Failed to read MCP config") from exc


def save_mcp_config_data(data: Dict[str, object]) -> None:
    """写入 MCP 配置文件。"""
    normalized = _normalize_mcp_config(data)
    try:
        with open(MCP_CONFIG_PATH, "w", encoding="utf-8") as config_file:
            json.dump(normalized, config_file, ensure_ascii=False, indent=2)
    except Exception as exc:
        raise MCPConfigError("Failed to write MCP config") from exc


def _collect_supported_mcp_models() -> list[str]:
    """从默认模型表收集支持 MCP 的模型 ID。"""
    seen: set[str] = set()
    for items in DEFAULT_MODELS.values():
        for item in items:
            if item.get("support_mcp"):
                seen.add(item["id"])
    return sorted(seen)


def ensure_dootask_mcp_config(enabled: bool) -> None:
    """确保 DooTask MCP 配置已写入配置文件。"""
    config_data = load_mcp_config_data(fallback_empty=True)
    mcps = config_data.get("mcps") or []

    if any(mcp.get("name") == DOOTASK_MCP_NAME for mcp in mcps):
        return

    default_config = {
        "name": DOOTASK_MCP_NAME,
        "config": "{}",
        "supportedModels": _collect_supported_mcp_models(),
        "enabled": enabled,
        "isSystem": True,
    }
    mcps.append(default_config)
    config_data["mcps"] = mcps

    try:
        save_mcp_config_data(config_data)
        logger.info("✅ 已写入默认 DooTask MCP 配置")
    except MCPConfigError as exc:  # pragma: no cover - best effort config write
        logger.error(f"❌ 写入 MCP 配置失败: {exc}")


class ModelListError(Exception):
    """Raised when model list retrieval fails."""


def _fetch_ollama_models(
    base_url: str,
    key: Optional[str] = None,
    agency: Optional[str] = None,
) -> Dict[str, object]:
    if not base_url:
        raise ModelListError("请先填写 Base URL")

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    request_kwargs: Dict[str, object] = {
        "headers": headers,
        "timeout": 15,
    }

    if agency:
        request_kwargs["proxies"] = agency

    url = base_url.rstrip("/") + "/api/tags"
    try:
        with httpx.Client(**request_kwargs) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        raise ModelListError(f"获取失败：HTTP {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise ModelListError(f"获取失败：{exc}") from exc
    except ValueError as exc:
        raise ModelListError("获取失败：响应解析错误") from exc

    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        raise ModelListError("获取失败：无效的返回结构")

    formatted: List[ModelInfo] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        model_name = item.get("model")
        display_name = item.get("name")
        if not model_name:
            continue
        formatted.append({
            "id": str(model_name),
            "name": display_name if display_name and display_name != model_name else str(model_name),
            "support_mcp": False
        })

    if not formatted:
        raise ModelListError("未找到默认模型")

    return {"models": formatted, "original": models}


def get_models_list(
    model_type: str,
    base_url: Optional[str] = None,
    key: Optional[str] = None,
    agency: Optional[str] = None,
) -> Dict[str, object]:
    """Retrieve models list data for the given model type."""
    model_type = (model_type or "").strip().lower()
    if not model_type:
        raise ModelListError("缺少参数 type")

    if model_type == "ollama":
        return _fetch_ollama_models(base_url=base_url or "", key=key or None, agency=agency or None)

    default_models = DEFAULT_MODELS.get(model_type)
    if not default_models:
        raise ModelListError("未找到默认模型")

    return {"models": default_models}
