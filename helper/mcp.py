import json
import logging
from typing import Dict, List, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient

from helper.config import DOOTASK_MCP_ID, DOOTASK_MCP_NAME, MCP_CONFIG_PATH, MCP_STREAM_URL, DEFAULT_MODELS

logger = logging.getLogger("ai")


class MCPConfigError(Exception):
    """Raised when MCP 配置文件读写失败。"""


def _default_mcp_config() -> Dict[str, object]:
    return {"mcps": []}


def _normalize_mcp_config(data: Dict[str, object]) -> Dict[str, object]:
    mcps = data.get("mcps")
    if not isinstance(mcps, list):
        data["mcps"] = []
        return data

    normalized_mcps: List[Dict[str, object]] = []

    for item in mcps:
        if not isinstance(item, dict):
            continue

        normalized = dict(item)
        is_system = bool(normalized.get("isSystem"))
        if is_system:
            normalized["id"] = DOOTASK_MCP_ID

        if not isinstance(normalized.get("supportedModels"), list):
            normalized["supportedModels"] = []

        normalized_mcps.append(normalized)

    data["mcps"] = normalized_mcps
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
            logger.error("❌ 读取 MCP 配置失败: %s", exc)
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


def _collect_supported_mcp_models() -> List[Dict[str, str]]:
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

    if any(
        isinstance(mcp, dict) and mcp.get("id") == DOOTASK_MCP_ID
        for mcp in mcps
    ):
        return

    default_config = {
        "id": DOOTASK_MCP_ID,
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
        logger.error("❌ 写入 MCP 配置失败: %s", exc)


def _mcp_supports_model(mcp_entry: Dict[str, object], model_name: str) -> bool:
    """检查目标模型是否在 MCP 的支持列表中。"""
    if not model_name:
        return True
    supported = mcp_entry.get("supportedModels")
    if not isinstance(supported, list) or not supported:
        return True
    for item in supported:
        model_id = None
        if isinstance(item, dict):
            model_id = item.get("id")
        elif isinstance(item, str):
            model_id = item
        if isinstance(model_id, str) and model_id == model_name:
            return True
    return False


def _pick_token(token_candidates: List[Optional[str]]) -> str:
    for token in token_candidates:
        if isinstance(token, str) and token:
            return token
    return "unknown"


def _build_dootask_mcp_config(
    token: str,
) -> Optional[Dict[str, object]]:
    return {
        "url": MCP_STREAM_URL,
        "transport": "streamable_http",
        "headers": {
            "token": token or "unknown"
        },
    }


def _load_custom_mcp_config(mcp_entry: Dict[str, object], server_key: str) -> Optional[Dict[str, object]]:
    config_value = mcp_entry.get("config")
    parsed_config: Optional[Dict[str, object]] = None

    if isinstance(config_value, str):
        config_text = config_value.strip()
        if not config_text:
            logger.warning("Skipping MCP %s because config is empty", server_key)
            return None
        try:
            maybe_config = json.loads(config_text)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping MCP %s due to invalid JSON config: %s", server_key, exc)
            return None
        if isinstance(maybe_config, dict):
            parsed_config = maybe_config
    elif isinstance(config_value, dict):
        parsed_config = config_value

    if not isinstance(parsed_config, dict):
        logger.warning("Skipping MCP %s because config is not a valid object", server_key)
        return None
    return parsed_config


async def load_mcp_tools_for_model(
    model_name: str,
    *,
    dootask_available: bool,
    token_candidates: List[Optional[str]],
) -> List[object]:
    """根据配置文件加载与当前模型匹配的 MCP 工具列表。"""
    try:
        config_data = load_mcp_config_data(fallback_empty=True)
    except MCPConfigError:
        logger.error("Failed to read MCP config when building tool list")
        return []

    mcps = config_data.get("mcps") or []
    if not isinstance(mcps, list):
        return []

    server_configs: Dict[str, Dict[str, object]] = {}
    token_value = _pick_token(token_candidates)

    for mcp in mcps:
        if not isinstance(mcp, dict):
            continue
        if mcp.get("enabled") is False:
            continue
        if not _mcp_supports_model(mcp, model_name):
            continue

        is_dootask = mcp.get("id") == DOOTASK_MCP_ID
        if is_dootask:
            server_key = "dootask-task"
        else:
            server_key = str(mcp.get("id") or mcp.get("name") or "").strip()
            if not server_key:
                server_key = f"mcp-{len(server_configs) + 1}"

        if is_dootask:
            if not dootask_available:
                continue
            config = _build_dootask_mcp_config(token_value)
        else:
            config = _load_custom_mcp_config(mcp, server_key)

        if not config:
            continue
        server_configs[server_key] = config

    if not server_configs:
        return []

    client = MultiServerMCPClient(server_configs)
    try:
        return await client.get_tools()
    except Exception as exc:
        logger.error("Failed to load MCP tools: %s", exc)
        return []
