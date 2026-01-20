"""
Built-in Tools Module

Defines internal tools that are loaded alongside MCP tools.
"""

import json
import logging
from typing import Any, List, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("ai")


class GetSessionImageInput(BaseModel):
    """Input schema for get_session_image tool."""

    image_md5: str = Field(
        description="The MD5 hash of the session image (from [picture:session_xxx] placeholder)"
    )


class GetSessionImageTool(BaseTool):
    """Tool for retrieving session images from cache."""

    name: str = "get_session_image"
    description: str = """获取当前会话中用户上传的图片。
当用户询问会话中图片的细节（如"刚才那张图的右上角是什么"）时，
使用此工具获取图片内容进行分析。
输入参数为图片的 MD5 哈希值（从 [picture:session_xxx] 占位符中提取）。"""

    args_schema: Type[BaseModel] = GetSessionImageInput
    response_format: str = "content_and_artifact"

    redis_manager: Any = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, image_md5: str) -> tuple:
        """Sync version - not implemented."""
        raise NotImplementedError("Use async version")

    async def _arun(self, image_md5: str) -> tuple:
        """Retrieve a session image from cache.

        Args:
            image_md5: MD5 hash of the image (with or without 'session_' prefix)

        Returns:
            Tuple of (content_blocks, artifact)
        """
        # Validate input
        if not image_md5 or len(image_md5) < 8:
            return ([{"type": "text", "text": "无效的图片标识符"}], None)

        # Normalize key (support with or without session_ prefix)
        if image_md5.startswith("session_"):
            md5_hash = image_md5[8:]
        else:
            md5_hash = image_md5

        # Retrieve from cache
        cache_key = f"session_image_{md5_hash}"
        try:
            cached = await self.redis_manager.get_cache(cache_key)
        except Exception as e:
            logger.error(f"Failed to get session image: {e}")
            return ([{"type": "text", "text": f"获取图片失败: {e}"}], None)

        if not cached:
            return ([{"type": "text", "text": "图片不存在或已过期"}], None)

        # Parse cached data
        try:
            cache_data = json.loads(cached)
            base64_data = cache_data["data"]
            mime_type = cache_data.get("mime_type", "image/jpeg")
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Invalid cache data for session image: {e}")
            return ([{"type": "text", "text": "图片数据格式错误"}], None)

        # Return multimodal content
        return ([{
            "type": "image",
            "mime_type": mime_type,
            "base64": base64_data
        }], None)


def load_builtin_tools(redis_manager: Any) -> List[BaseTool]:
    """Load all built-in tools.

    Args:
        redis_manager: Redis manager instance

    Returns:
        List of built-in tools
    """
    return [
        GetSessionImageTool(redis_manager=redis_manager)
    ]
