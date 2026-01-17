"""
Redis 管理模块

负责与 Redis 的交互，包括上下文存储、输入缓存等。
"""

import redis.asyncio as redis
import json
import os


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
    async def get_context(self, key):
        """从 Redis 获取上下文"""
        data = await self.client.get(self._make_key("context", key))
        if data:
            context = json.loads(data)
            return context if isinstance(context, list) else []
        return []

    async def set_context(self, key, value, model_type=None, model_name=None, context_limit=None):
        """设置上下文到 Redis，根据模型限制截断内容"""
        if not isinstance(value, list):
            raise ValueError("Context must be a list")
        await self.client.set(self._make_key("context", key), json.dumps(value))

    async def append_context(self, key, role, content, model_type=None, model_name=None, context_limit=None):
        """添加新的上下文消息"""
        context = await self.get_context(key)
        context.append({"type": role, "content": content})
        await self.set_context(key, context, model_type, model_name, context_limit)

    async def extend_contexts(self, key, contents, model_type=None, model_name=None, context_limit=None):
        """添加新的上下文消息"""
        context = await self.get_context(key)
        context.extend(contents)
        await self.set_context(key, context, model_type, model_name, context_limit)

    async def delete_context(self, key):
        """删除上下文"""
        await self.client.delete(self._make_key("context", key))


    # 输入部分
    async def get_input(self, key):
        """从 Redis 获取输入"""
        data = await self.client.get(self._make_key("input", key))
        return json.loads(data) if data else None

    async def set_input(self, key, value, expire=86400):
        """设置输入到 Redis"""
        await self.client.set(self._make_key("input", key), json.dumps(value), ex=expire)

    async def delete_input(self, key):
        """删除输入"""
        await self.client.delete(self._make_key("input", key))

    async def scan_inputs(self, match="input:*"):
        """扫描所有输入"""
        full_match = f"{self._prefix}{match}"
        prefix_len = len(self._prefix) + len("input:")
        async for key in self.client.scan_iter(full_match):
            key_id = key[prefix_len:]
            data = await self.get_input(key_id)
            if data:
                yield key_id, data

    async def set_cache(self, key, value, **kwargs):
        """设置临时缓存，支持超时"""
        cache_key = self._make_key("cache", key)
        return await self.client.set(cache_key, value, **kwargs)

    async def get_cache(self, key):
        """获取临时缓存的值"""
        cache_key = self._make_key("cache", key)
        return await self.client.get(cache_key) or ""

    async def delete_cache(self, key):
        """删除临时缓存"""
        cache_key = self._make_key("cache", key)
        return await self.client.delete(cache_key)
