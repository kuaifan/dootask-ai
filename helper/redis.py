import redis
import json
import os
import re
import tiktoken
from typing import List, Tuple

# 提前加载所需的编码
tiktoken.get_encoding("o200k_base")
tiktoken.get_encoding("cl100k_base")

# 定义模型的上下文限制（token数）
CONTEXT_LIMITS = {
    "openai": {
        "gpt-4": 8192,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
        "gpt-4o-mini": 16384,
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-16k": 16384,
        "gpt-3.5-turbo-0125": 4096,
        "gpt-3.5-turbo-1106": 4096,
        "default": 4096
    },
    "claude": {
        "claude-3-5-sonnet-latest": 200000,
        "claude-3-5-haiku-latest": 200000,
        "claude-3-5-opus-latest": 200000,
        "claude-2.1": 100000,
        "default": 200000
    },
    "deepseek": {
        "deepseek-chat": 32768,
        "deepseek-reasoner": 32768,
        "default": 32768
    },
    "gemini": {
        "gemini-1.5-flash": 1000000,
        "gemini-1.5-pro": 1000000,
        "default": 1000000
    },
    "zhipu": {
        "glm-4": 128000,
        "glm-4-long": 128000,
        "default": 128000
    },
    "qwen": {
        "qwen-turbo": 32000,
        "default": 32000
    },
    "grok": {
        "grok-2": 128000,
        "default": 128000
    }
}

def count_tokens(text: str, model_type: str, model_name: str) -> int:
    """计算文本的token数量"""
    if not text:
        return 0

    # 默认使用cl100k_base编码
    encoding_name = "cl100k_base"
    
    # 根据模型类型选择合适的编码
    if model_type == "openai":
        try:
            # 对OpenAI模型尝试获取特定的编码
            encoding = tiktoken.encoding_for_model(model_name)
            return len(encoding.encode(text))
        except KeyError:
            # 如果失败，使用默认编码
            pass
    
    # 对于deepseek模型和所有其他情况（包括OpenAI模型编码获取失败）
    # 使用默认的cl100k_base编码
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def handle_context_limits(pre_context: List[Tuple[str, str]], middle_context: List[Tuple[str, str]], end_context: List[Tuple[str, str]], model_type: str = None, model_name: str = None, custom_limit: int = None) -> List[Tuple[str, str]]:
    """处理上下文，确保不超过模型token限制"""
    all_context = pre_context + middle_context + end_context
    if not all_context:
        return []
        
    # 获取token限制
    if custom_limit and custom_limit > 0:
        token_limit = custom_limit
    else:
        token_limit = 4096  # 默认限制
        if model_type in CONTEXT_LIMITS:
            model_limits = CONTEXT_LIMITS[model_type]
            token_limit = model_limits.get(model_name, model_limits.get('default', 4096))
    
    # 按优先级处理上下文
    result = []
    current_tokens = 0

    # 1. 首先添加 end_context（最高优先级）
    for msg in end_context:
        msg_tokens = count_tokens(msg[1], model_type, model_name)
        if current_tokens + msg_tokens <= token_limit:
            result.append(msg)
            current_tokens += msg_tokens
        else:
            # 如果连 end_context 都放不下，直接返回能放下的部分
            return result

    # 2. 其次添加 pre_context（第二优先级）
    for msg in pre_context:
        msg_tokens = count_tokens(msg[1], model_type, model_name)
        if current_tokens + msg_tokens <= token_limit:
            result.insert(len(result) - len(end_context), msg)
            current_tokens += msg_tokens
        else:
            break

    # 3. 最后添加 middle_context（最低优先级）
    # 从最新的消息开始添加，保存到临时列表中
    temp_middle = []
    for msg in reversed(middle_context):
        msg_tokens = count_tokens(msg[1], model_type, model_name)
        if current_tokens + msg_tokens <= token_limit:
            temp_middle.append(msg)
            current_tokens += msg_tokens
        else:
            break
    
    # 将收集到的 middle_context 按原始顺序插入
    for msg in reversed(temp_middle):
        result.insert(len(result) - len(end_context), msg)
            
    return result

class RedisManager:
    _instance = None
    _prefix = "dootask_ai:"  # 添加全局应用前缀

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisManager, cls).__new__(cls)
            cls._instance.client = redis.Redis(
                host=os.environ.get('REDIS_HOST', 'localhost'),
                port=int(os.environ.get('REDIS_PORT', 6379)),
                db=int(os.environ.get('REDIS_DB', 0)),  # 添加数据库配置
                decode_responses=True
            )
        return cls._instance

    def _make_key(self, type_prefix, key):
        """生成带有应用前缀的完整键名"""
        return f"{self._prefix}{type_prefix}:{key}"

    # 上下文部分
    def get_context(self, key):
        """从 Redis 获取上下文"""
        data = self.client.get(self._make_key("context", key))
        if data:
            context = json.loads(data)
            return context if isinstance(context, list) else []
        return []

    def set_context(self, key, value, model_type=None, model_name=None, context_limit=None):
        """设置上下文到 Redis，根据模型限制截断内容"""
        # 确保 value 是列表格式
        if not isinstance(value, list):
            raise ValueError("Context must be a list of tuples")
        # 处理模型限制
        final_context = handle_context_limits(
            pre_context=[],
            middle_context=value,
            end_context=[],
            model_type=model_type,
            model_name=model_name,
            custom_limit=context_limit
        )
        # 保存到 Redis
        self.client.set(self._make_key("context", key), json.dumps(value))

    def append_context(self, key, role, content, model_type=None, model_name=None, context_limit=None):
        """添加新的上下文消息"""
        context = self.get_context(key)
        context.append((role, content))
        self.set_context(key, context, model_type, model_name, context_limit)

    def extend_contexts(self, key, contents, model_type=None, model_name=None, context_limit=None):
        """添加新的上下文消息"""
        context = self.get_context(key)
        context.extend(contents)
        self.set_context(key, context, model_type, model_name, context_limit)

    def delete_context(self, key):
        """删除上下文"""
        self.client.delete(self._make_key("context", key))


    # 输入部分
    def get_input(self, key):
        """从 Redis 获取输入"""
        data = self.client.get(self._make_key("input", key))
        return json.loads(data) if data else None

    def set_input(self, key, value, expire=86400):
        """设置输入到 Redis"""
        self.client.set(self._make_key("input", key), json.dumps(value), ex=expire)

    def delete_input(self, key):
        """删除输入"""
        self.client.delete(self._make_key("input", key))

    def scan_inputs(self, match="input:*"):
        """扫描所有输入"""
        full_match = f"{self._prefix}{match}"
        prefix_len = len(self._prefix) + len("input:")
        for key in self.client.scan_iter(full_match):
            key_id = key[prefix_len:]  # 移除前缀部分
            data = self.get_input(key_id)
            if data:
                yield key_id, data

    def set_cache(self, key, value, **kwargs):
        """设置临时缓存，支持超时"""
        cache_key = self._make_key("cache", key)
        return self.client.set(cache_key, value, **kwargs)

    def get_cache(self, key):
        """获取临时缓存的值"""
        cache_key = self._make_key("cache", key)
        return self.client.get(cache_key) or ""

    def delete_cache(self, key):
        """删除临时缓存"""
        cache_key = self._make_key("cache", key)
        return self.client.delete(cache_key)
