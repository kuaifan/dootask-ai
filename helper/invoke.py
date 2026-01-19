import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

Message = Tuple[str, str]
RawContext = Union[str, Dict[str, Any], Sequence[Any], None]


def coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_str(value: Any, default: Optional[str] = None) -> Optional[str]:
    if value is None:
        return default
    string_value = str(value).strip()
    if not string_value:
        return default
    return string_value


def _normalize_role(role: Any) -> Any:
    if not isinstance(role, str):
        return role
    lowered = role.lower()
    if lowered in ("user", "human"):
        return "human"
    if lowered in ("assistant", "ai", "bot"):
        return "assistant"
    if lowered in ("system", "sys"):
        return "system"
    return role


def _normalize_message(item: Any) -> Optional[BaseMessage]:
    role = None
    content = None
    if isinstance(item, dict):
        role = item.get("role") or item.get("type") or "human"
        content = item.get("content") or item.get("text")
    elif isinstance(item, (list, tuple)) and len(item) >= 2:
        role, content = item[0], item[1]
    elif isinstance(item, str):
        role, content = "human", item
    elif item is not None:
        role, content = "human", str(item)

    # 无内容则返回 None
    if content is None:
        return None

    # 如果 content 是空字符串，也返回 None
    if isinstance(content, str) and not content:
        return None

    # 角色标准化
    normalized_role = _normalize_role(role)

    # 处理 content - 支持字符串或多模态列表
    # 多模态列表格式: [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {...}}]
    final_content = content
    if isinstance(content, list):
        # Keep list content as-is for multimodal support
        final_content = content
    elif not isinstance(content, str):
        final_content = str(content)

    # 构造 LangChain 消息对象
    if normalized_role == "human":
        return HumanMessage(content=final_content)
    elif normalized_role == "assistant":
        # AI messages should be string for context storage
        return AIMessage(content=final_content if isinstance(final_content, str) else str(final_content))
    elif normalized_role == "system":
        # System messages should be string
        return SystemMessage(content=final_content if isinstance(final_content, str) else str(final_content))
    else:
        # 默认安全回退
        return HumanMessage(content=final_content)


def parse_context(raw_context: RawContext) -> List[Message]:
    """
    将不同形式的上下文统一转为 [(role, content), ...]
    """
    if raw_context is None:
        return []

    parsed = raw_context
    if isinstance(raw_context, str):
        stripped = raw_context.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [("human", stripped)]

    if isinstance(parsed, list):
        messages: List[Message] = []
        for item in parsed:
            normalized = _normalize_message(item)
            if normalized:
                messages.append(normalized)
        return messages

    if isinstance(parsed, dict):
        normalized = _normalize_message(parsed)
        return [normalized] if normalized else []

    normalized = _normalize_message(parsed)
    return [normalized] if normalized else []

def build_invoke_stream_key(stream_key: str) -> str:
    return f"invoke_stream_{stream_key}"