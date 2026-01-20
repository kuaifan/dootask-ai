"""
上下文管理模块

负责：
- Token 计数
- 模型上下文限制查询（智能规则匹配）
- 上下文截断处理
"""

import re
import tiktoken
from typing import List, Optional

from helper.config import CONTEXT_LIMITS

# 提前加载所需的编码
tiktoken.get_encoding("o200k_base")
tiktoken.get_encoding("cl100k_base")


# =============================================================================
# 版本提取与比较工具
# =============================================================================

# 版本号提取正则：支持 x.y.z、x.y、x 等格式
VERSION_PATTERN = re.compile(r'(\d+)(?:\.(\d+))?(?:\.(\d+))?')


def _extract_version(model_name: str) -> Optional[tuple]:
    """
    从模型名称中提取版本号
    返回 (major, minor, patch) 元组，未匹配的部分为 0
    例如：
      "gpt-5.2-chat" → (5, 2, 0)
      "claude-4.5" → (4, 5, 0)
      "gemini-2.5-pro" → (2, 5, 0)
      "glm-4" → (4, 0, 0)

    特殊处理：
      "gpt-4o" → (4, 1, 0)  # 4o 系列视为 4.1 级别
      "gpt-4-turbo" → (4, 1, 0)  # turbo 系列视为 4.1 级别
    """
    model_lower = model_name.lower()

    # 特殊处理：gpt-4o 和 gpt-4-turbo 系列应视为 4.1 级别
    if re.search(r'\bgpt[-_]?4o\b', model_lower) or re.search(r'\bgpt[-_]?4[-_]?turbo\b', model_lower):
        return (4, 1, 0)

    match = VERSION_PATTERN.search(model_name)
    if not match:
        return None
    major = int(match.group(1))
    minor = int(match.group(2)) if match.group(2) else 0
    patch = int(match.group(3)) if match.group(3) else 0
    return (major, minor, patch)


# =============================================================================
# 模型家族检测规则
# =============================================================================

# 模型家族检测模式（按优先级排序）
# 格式: (pattern, family_name, model_type_hint)
MODEL_FAMILY_PATTERNS = [
    # OpenAI o 系列（优先检测，避免被 gpt 规则误匹配）
    (re.compile(r'\bo([134])[-_]?(mini)?', re.I), 'o-series', 'openai'),

    # OpenAI GPT 系列（宽松匹配，支持自定义命名）
    (re.compile(r'\bgpt', re.I), 'gpt', 'openai'),

    # Claude 系列
    (re.compile(r'\bclaude', re.I), 'claude', 'claude'),

    # Google Gemini 系列
    (re.compile(r'\bgemini', re.I), 'gemini', 'gemini'),

    # DeepSeek 系列
    (re.compile(r'\bdeepseek', re.I), 'deepseek', 'deepseek'),

    # xAI Grok 系列
    (re.compile(r'\bgrok', re.I), 'grok', 'grok'),

    # 智谱 GLM 系列
    (re.compile(r'\bglm', re.I), 'glm', 'zhipu'),

    # 通义千问 Qwen 系列
    (re.compile(r'\bqwen', re.I), 'qwen', 'qianwen'),

    # 文心一言 Ernie 系列
    (re.compile(r'\bernie', re.I), 'ernie', 'wenxin'),
]


def _detect_model_family(model_name: str) -> Optional[tuple]:
    """
    检测模型所属家族
    返回 (family_name, model_type_hint) 或 None
    """
    model_lower = model_name.lower()
    for pattern, family, model_type in MODEL_FAMILY_PATTERNS:
        if pattern.search(model_lower):
            return (family, model_type)
    return None


# =============================================================================
# 基于规则的上下文限制
# =============================================================================

# 规则格式: (min_version, max_version, limit)
# min_version: 最小版本（含），None 表示无下限
# max_version: 最大版本（不含），None 表示无上限
# limit: token 限制

FAMILY_VERSION_RULES = {
    'gpt': [
        # GPT-5.x 及以上：预计 400K context
        ((5, 0, 0), None, 400000),
        # GPT-4.1 及以上（包括 4o 系列）：128K
        ((4, 1, 0), (5, 0, 0), 128000),
        # GPT-4.0：原始 8K
        ((4, 0, 0), (4, 1, 0), 8000),
        # GPT-3.5：4K-16K，保守使用 4K
        ((3, 5, 0), (4, 0, 0), 4000),
        # GPT-3 及更早：4K
        (None, (3, 5, 0), 4000),
    ],
    'o-series': [
        # o4 系列：200K
        ((4, 0, 0), None, 200000),
        # o3 系列：200K
        ((3, 0, 0), (4, 0, 0), 200000),
        # o1 系列：128K
        ((1, 0, 0), (3, 0, 0), 128000),
    ],
    'claude': [
        # Claude 全系列：200K context
        (None, None, 200000),
    ],
    'gemini': [
        # Gemini 2.x 及以上：~1M context
        ((2, 0, 0), None, 1000000),
        # Gemini 1.x：32K
        ((1, 0, 0), (2, 0, 0), 32000),
    ],
    'deepseek': [
        # DeepSeek 全系列：128K
        (None, None, 128000),
    ],
    'grok': [
        # Grok 4.x：2M context
        ((4, 0, 0), None, 2000000),
        # Grok 3.x：128K
        ((3, 0, 0), (4, 0, 0), 128000),
    ],
    'glm': [
        # GLM-4 系列：128K
        ((4, 0, 0), None, 128000),
        # GLM-3 系列：8K
        ((3, 0, 0), (4, 0, 0), 8000),
    ],
    'qwen': [
        # Qwen 全系列：默认 32K
        (None, None, 32000),
    ],
    'ernie': [
        # Ernie 全系列：默认 8K
        (None, None, 8000),
    ],
}


def _apply_version_rules(family: str, version: Optional[tuple]) -> Optional[int]:
    """
    根据模型家族和版本应用规则
    返回 token 限制或 None（未匹配任何规则）
    """
    rules = FAMILY_VERSION_RULES.get(family)
    if not rules:
        return None

    for min_ver, max_ver, limit in rules:
        # 检查版本范围
        if version is None:
            # 无法提取版本，使用最后一条规则（通常是默认/最小值）
            continue

        in_range = True
        if min_ver is not None and version < min_ver:
            in_range = False
        if max_ver is not None and version >= max_ver:
            in_range = False

        if in_range:
            return limit

    # 未匹配任何规则，返回该家族的最后一条规则值（作为默认）
    return rules[-1][2] if rules else None


# =============================================================================
# 特殊模式检测（从模型名称中提取上下文大小提示）
# =============================================================================

def _detect_context_hint_from_name(model_name: str) -> Optional[int]:
    """
    从模型名称中检测上下文大小提示
    例如：xxx-128k、xxx-32k、xxx-long 等
    返回 token 限制或 None
    """
    model_lower = model_name.lower()

    # 检测 xxxk 模式（如 128k、32k、16k、8k）
    k_match = re.search(r'[-_](\d+)k\b', model_lower)
    if k_match:
        k_size = int(k_match.group(1))
        return k_size * 1000

    # 检测 long 模式（通常表示大上下文）
    if '-long' in model_lower or '_long' in model_lower:
        # 假设 long 版本至少 128K
        return 128000

    return None


# =============================================================================
# 主函数：智能模型限制解析
# =============================================================================

def model_limit(model_type: str, model_name: str) -> int:
    """
    获取模型 token 限制（智能规则匹配）

    匹配优先级：
    1. 精确匹配 CONTEXT_LIMITS 中的模型名
    2. 从模型名称中检测上下文大小提示（如 128k、32k、long）
    3. 检测模型家族 + 版本，应用版本规则
    4. 使用 model_type 的默认值
    5. 全局默认值 4096
    """
    model_name_lower = model_name.lower() if model_name else ""

    # 1. 精确匹配
    if model_type in CONTEXT_LIMITS:
        model_limits = CONTEXT_LIMITS[model_type]
        if model_name in model_limits:
            return model_limits[model_name]
        # 也尝试小写匹配
        if model_name_lower in model_limits:
            return model_limits[model_name_lower]

    # 2. 从模型名称中检测上下文大小提示
    hint_limit = _detect_context_hint_from_name(model_name)
    if hint_limit:
        return hint_limit

    # 3. 检测模型家族 + 版本，应用版本规则
    family_info = _detect_model_family(model_name)
    if family_info:
        family, inferred_type = family_info
        version = _extract_version(model_name)
        rule_limit = _apply_version_rules(family, version)
        if rule_limit:
            return rule_limit

    # 4. 使用 model_type 的默认值
    if model_type in CONTEXT_LIMITS:
        return CONTEXT_LIMITS[model_type].get('default', 4096)

    # 5. 全局默认值
    return 4096


# =============================================================================
# Token 计数
# =============================================================================

def count_tokens(text, model_type: str, model_name: str) -> int:
    """计算文本或多模态内容的 token 数量

    Args:
        text: 文本字符串或多模态内容列表
        model_type: 模型类型
        model_name: 模型名称

    Returns:
        token 数量
    """
    if not text:
        return 0

    # 处理多模态内容（列表格式）
    if isinstance(text, list):
        total_tokens = 0
        for item in text:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                if item_type == "text":
                    # 文本内容，递归计算
                    total_tokens += count_tokens(item.get("text", ""), model_type, model_name)
                elif item_type == "image_url":
                    # 图片内容，使用估算值（低分辨率约 85 tokens，高分辨率更多）
                    # 由于前端已压缩到 1024px，使用 765 tokens 作为合理估算
                    total_tokens += 765
            elif isinstance(item, str):
                total_tokens += count_tokens(item, model_type, model_name)
        return total_tokens

    # 确保是字符串
    if not isinstance(text, str):
        text = str(text)

    # 默认使用 cl100k_base 编码
    encoding_name = "cl100k_base"

    # 根据模型类型选择合适的编码
    if model_type == "openai":
        try:
            # 对 OpenAI 模型尝试获取特定的编码
            encoding = tiktoken.encoding_for_model(model_name)
            return len(encoding.encode(text))
        except KeyError:
            # 如果失败，使用默认编码
            pass

    # 对于其他模型使用默认的 cl100k_base 编码
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


# =============================================================================
# 上下文截断处理
# =============================================================================

def handle_context_limits(
    pre_context: list,
    middle_context: list,
    end_context: list,
    model_type: str = None,
    model_name: str = None,
    custom_limit: int = None,
    default_ratio: float = 1.0
) -> List:
    """
    处理上下文，确保不超过模型 token 限制

    优先级：end_context > pre_context > middle_context
    middle_context 从最新消息开始保留

    Args:
        pre_context: 前置上下文（如系统提示）
        middle_context: 中间上下文（历史消息）
        end_context: 结尾上下文（当前输入）
        model_type: 模型类型
        model_name: 模型名称
        custom_limit: 自定义 token 限制（优先使用，不受 default_ratio 影响）
        default_ratio: 默认限制的使用比例（0-1），如 0.9 表示使用 90%，预留 10% 给输出

    Returns:
        截断后的上下文列表
    """
    all_context = pre_context + middle_context + end_context
    if not all_context:
        return []

    # 获取 token 限制
    if custom_limit and custom_limit > 0:
        # 用户自定义限制，直接使用，不应用比例
        token_limit = custom_limit
    else:
        # 使用模型默认限制，应用比例
        token_limit = model_limit(model_type, model_name)
        if 0 < default_ratio < 1:
            token_limit = int(token_limit * default_ratio)

    # 按优先级处理上下文
    result = []
    current_tokens = 0

    # 1. 首先添加 end_context（最高优先级）
    for msg in end_context:
        msg_tokens = count_tokens(msg.content, model_type, model_name)
        if current_tokens + msg_tokens <= token_limit:
            result.append(msg)
            current_tokens += msg_tokens
        else:
            # 如果连 end_context 都放不下，直接返回能放下的部分
            return result

    # 2. 其次添加 pre_context（第二优先级）
    for msg in pre_context:
        msg_tokens = count_tokens(msg.content, model_type, model_name)
        if current_tokens + msg_tokens <= token_limit:
            result.insert(len(result) - len(end_context), msg)
            current_tokens += msg_tokens
        else:
            break

    # 3. 最后添加 middle_context（最低优先级）
    # 从最新的消息开始添加，保存到临时列表中
    temp_middle = []
    for msg in reversed(middle_context):
        msg_tokens = count_tokens(msg.content, model_type, model_name)
        if current_tokens + msg_tokens <= token_limit:
            temp_middle.append(msg)
            current_tokens += msg_tokens
        else:
            break

    # 将收集到的 middle_context 按原始顺序插入
    for msg in reversed(temp_middle):
        result.insert(len(result) - len(end_context), msg)

    return result
