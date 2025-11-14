from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, TypedDict

import httpx

from helper.config import DOOTASK_MCP_NAME, MCP_CONFIG_PATH, DEFAULT_MODELS

logger = logging.getLogger("ai")

class ModelInfo(TypedDict):
    """模型信息类型定义"""
    id: str
    name: str
    support_mcp: bool



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


def _collect_supported_mcp_models() -> list[Dict[str, str]]:
    """从默认模型表收集支持 MCP 的模型列表。"""
    seen: Dict[str, str] = {}
    for items in DEFAULT_MODELS.values():
        for item in items:
            if item.get("support_mcp"):
                seen[item["id"]] = item.get("name") or item["id"]
    return [
        {"id": model_id, "name": seen[model_id]}
        for model_id in sorted(seen.keys())
    ]


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
