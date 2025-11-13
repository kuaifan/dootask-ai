from __future__ import annotations

from typing import Dict, List, Optional

import httpx


DEFAULT_MODELS: Dict[str, List[str]] = {
    "openai": [
        "gpt-5-chat-latest | GPT-5 Chat",
        "gpt-5-codex | GPT-5-Codex",
        "gpt-5-mini | GPT-5 mini",
        "gpt-5-nano | GPT-5 nano",
        "gpt-5-pro | GPT-5 pro",
        "gpt-5.1-chat-latest | GPT-5.1 Chat",
        "gpt-5.1-codex-mini | GPT-5.1 Codex mini",
        "gpt-5.1-codex | GPT-5.1 Codex",
        "gpt-5.1 | GPT-5.1",
        "gpt-5 | GPT-5",
        "gpt-4.1 | GPT-4.1",
        "gpt-4o | GPT-4o",
        "gpt-4 | GPT-4",
        "gpt-4o-mini | GPT-4o Mini",
        "gpt-4-turbo | GPT-4 Turbo",
        "o3 | GPT-o3",
        "o1 | GPT-o1",
        "o4-mini | GPT-o4 Mini",
        "o3-mini | GPT-o3 Mini",
        "o1-mini | GPT-o1 Mini",
        "gpt-3.5-turbo | GPT-3.5 Turbo",
        "gpt-3.5-turbo-16k | GPT-3.5 Turbo 16K",
        "gpt-3.5-turbo-0125 | GPT-3.5 Turbo 0125",
        "gpt-3.5-turbo-1106 | GPT-3.5 Turbo 1106",
    ],
    "claude": [
        "claude-sonnet-4-5 (thinking) | Claude Sonnet 4.5",
        "claude-haiku-4-5 (thinking) | Claude Haiku 4.5",
        "claude-opus-4-1 (thinking) | Claude Opus 4.1",
        "claude-sonnet-4-0 (thinking) | Claude Sonnet 4.0",
        "claude-opus-4-0 (thinking) | Claude Opus 4.0",
        "claude-3-7-sonnet-latest (thinking) | Claude Sonnet 3.7",
        "claude-3-5-haiku-latest | Claude Haiku 3.5",
    ],
    "deepseek": [
        "deepseek-chat | DeepSeek-V3.2",
        "deepseek-reasoner | DeepSeek-V3.2-Reasoner",
    ],
    "gemini": [
        "gemini-2.5-pro | Gemini 2.5 Pro",
        "gemini-2.5-pro-tts | Gemini 2.5 Pro TTS",
        "gemini-2.5-flash | Gemini 2.5 Flash",
        "gemini-2.5-flash-tts | Gemini 2.5 Flash TTS",
        "gemini-2.5-flash-live | Gemini 2.5 Flash Live",
        "gemini-2.5-flash-image | Gemini 2.5 Flash Image",
        "gemini-2.0-flash | Gemini 2.0 Flash",
        "gemini-2.0-flash-lite | Gemini 2.0 Flash Lite",
    ],
    "grok": [
        "grok-code-fast-1 | Grok Code Fast 1",
        "grok-4-fast-reasoning | Grok 4 Fast Reasoning",
        "grok-4-fast-non-reasoning | Grok 4 Fast",
        "grok-4-0709 | Grok 4",
        "grok-3-latest | Grok 3",
        "grok-3-fast-latest | Grok 3 Fast",
    ],
    "zhipu": [
        "glm-4 | GLM-4",
        "glm-4-plus | GLM-4 Plus",
        "glm-4-air | GLM-4 Air",
        "glm-4-airx | GLM-4 AirX",
        "glm-4-long | GLM-4 Long",
        "glm-4-flash | GLM-4 Flash",
        "glm-4v | GLM-4V",
        "glm-4v-plus | GLM-4V Plus",
        "glm-3-turbo | GLM-3 Turbo",
    ],
    "qianwen": [
        "qwen-max | QWEN Max",
        "qwen-max-latest | QWEN Max Latest",
        "qwen-turbo | QWEN Turbo",
        "qwen-turbo-latest | QWEN Turbo Latest",
        "qwen-plus | QWEN Plus",
        "qwen-plus-latest | QWEN Plus Latest",
        "qwen-long | QWEN Long",
    ],
    "wenxin": [
        "ernie-4.5-turbo-128k | Ernie 4.5 Turbo 128K",
        "ernie-4.5-turbo-32k | Ernie 4.5 Turbo 32K",
        "ernie-4.5-turbo-latest | Ernie 4.5 Turbo Latest",
        "ernie-4.5-turbo-vl | Ernie 4.5 Turbo VL",
        "ernie-4.0-8k | Ernie 4.0 8K",
        "ernie-4.0-8k-latest | Ernie 4.0 8K Latest",
        "ernie-4.0-turbo-128k | Ernie 4.0 Turbo 128K",
        "ernie-4.0-turbo-8k | Ernie 4.0 Turbo 8K",
        "ernie-3.5-128k | Ernie 3.5 128K",
        "ernie-3.5-8k | Ernie 3.5 8K",
        "ernie-speed-128k | Ernie Speed 128K",
        "ernie-speed-8k | Ernie Speed 8K",
        "ernie-lite-8k | Ernie Lite 8K",
        "ernie-tiny-8k | Ernie Tiny 8K",
    ],
}


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

    formatted: List[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        model_name = item.get("model")
        display_name = item.get("name")
        if not model_name:
            continue
        if display_name and display_name != model_name:
            formatted.append(f"{model_name} | {display_name}")
        else:
            formatted.append(str(model_name))

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
