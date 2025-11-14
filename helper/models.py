from __future__ import annotations

import logging
from typing import Dict, List, Optional, TypedDict

import httpx

from helper.config import DEFAULT_MODELS

logger = logging.getLogger("ai")

class ModelInfo(TypedDict):
    """模型信息类型定义"""
    id: str
    name: str
    support_mcp: bool

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
