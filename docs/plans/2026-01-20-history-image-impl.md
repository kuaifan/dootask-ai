# History Image Processing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace historical images with placeholders to save tokens while allowing AI to retrieve them on demand via a built-in tool.

**Architecture:** Images in historical messages are replaced with `[Picture:history_{md5}]` placeholders and cached in Redis with 2-hour TTL. A new built-in tool `get_history_image` allows AI to fetch cached images when needed, returning multimodal content blocks.

**Tech Stack:** Python, FastAPI, Redis, LangChain (BaseTool, ToolMessage)

---

## Task 1: Create History Image Module

**Files:**
- Create: `helper/history_image.py`
- Test: `tests/test_history_image.py`

### Step 1: Write the failing test for `extract_base64_and_mime`

```python
# tests/test_history_image.py
import pytest


class TestExtractBase64AndMime:
    """Tests for extract_base64_and_mime function."""

    def test_extract_jpeg_image(self):
        """Should extract base64 data and mime type from JPEG data URL."""
        from helper.history_image import extract_base64_and_mime

        data_url = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        base64_data, mime_type = extract_base64_and_mime(data_url)

        assert base64_data == "/9j/4AAQSkZJRg=="
        assert mime_type == "image/jpeg"

    def test_extract_png_image(self):
        """Should extract base64 data and mime type from PNG data URL."""
        from helper.history_image import extract_base64_and_mime

        data_url = "data:image/png;base64,iVBORw0KGgo="
        base64_data, mime_type = extract_base64_and_mime(data_url)

        assert base64_data == "iVBORw0KGgo="
        assert mime_type == "image/png"

    def test_invalid_data_url_returns_none(self):
        """Should return None for invalid data URLs."""
        from helper.history_image import extract_base64_and_mime

        result = extract_base64_and_mime("https://example.com/image.jpg")
        assert result is None

        result = extract_base64_and_mime("not a url")
        assert result is None
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_history_image.py::TestExtractBase64AndMime -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'helper.history_image'"

### Step 3: Write minimal implementation

```python
# helper/history_image.py
"""
History Image Processing Module

Handles caching of historical images and replacing them with placeholders
to reduce token usage while allowing AI to retrieve them on demand.
"""

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("ai")

# Cache TTL: 2 hours
HISTORY_IMAGE_TTL = 7200

# Placeholder pattern
PLACEHOLDER_PATTERN = re.compile(r"\[Picture:history_([a-f0-9]{32})\]")


def extract_base64_and_mime(data_url: str) -> Optional[Tuple[str, str]]:
    """Extract base64 data and MIME type from a data URL.

    Args:
        data_url: A data URL string (e.g., "data:image/jpeg;base64,...")

    Returns:
        Tuple of (base64_data, mime_type) or None if invalid
    """
    if not data_url or not data_url.startswith("data:"):
        return None

    try:
        # Format: data:image/jpeg;base64,/9j/4AAQ...
        header, base64_data = data_url.split(",", 1)
        # Extract mime type from header
        mime_match = re.match(r"data:([^;]+);base64", header)
        if not mime_match:
            return None
        mime_type = mime_match.group(1)
        return base64_data, mime_type
    except (ValueError, AttributeError):
        return None
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_history_image.py::TestExtractBase64AndMime -v`
Expected: PASS

### Step 5: Commit

```bash
git add helper/history_image.py tests/test_history_image.py
git commit -m "feat(history_image): add extract_base64_and_mime function"
```

---

## Task 2: Add find_last_human_index Function

**Files:**
- Modify: `helper/history_image.py`
- Modify: `tests/test_history_image.py`

### Step 1: Write the failing test

```python
# Add to tests/test_history_image.py

class TestFindLastHumanIndex:
    """Tests for find_last_human_index function."""

    def test_find_last_human_in_list(self):
        """Should find the index of the last human message."""
        from helper.history_image import find_last_human_index

        messages = [
            {"type": "human", "content": "first"},
            {"type": "assistant", "content": "response"},
            {"type": "human", "content": "second"},
            {"type": "assistant", "content": "response2"},
        ]
        assert find_last_human_index(messages) == 2

    def test_find_last_human_at_end(self):
        """Should find human message at the end."""
        from helper.history_image import find_last_human_index

        messages = [
            {"type": "human", "content": "first"},
            {"type": "assistant", "content": "response"},
            {"type": "human", "content": "last"},
        ]
        assert find_last_human_index(messages) == 2

    def test_no_human_messages(self):
        """Should return -1 when no human messages exist."""
        from helper.history_image import find_last_human_index

        messages = [
            {"type": "assistant", "content": "response"},
            {"type": "system", "content": "system"},
        ]
        assert find_last_human_index(messages) == -1

    def test_empty_list(self):
        """Should return -1 for empty list."""
        from helper.history_image import find_last_human_index

        assert find_last_human_index([]) == -1

    def test_tuple_format_messages(self):
        """Should handle tuple format messages."""
        from helper.history_image import find_last_human_index

        messages = [
            ("human", "first"),
            ("assistant", "response"),
            ("human", "last"),
        ]
        assert find_last_human_index(messages) == 2
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_history_image.py::TestFindLastHumanIndex -v`
Expected: FAIL with "cannot import name 'find_last_human_index'"

### Step 3: Write minimal implementation

```python
# Add to helper/history_image.py

def find_last_human_index(messages: List[Any]) -> int:
    """Find the index of the last human message in the list.

    Args:
        messages: List of messages (dict or tuple format)

    Returns:
        Index of last human message, or -1 if not found
    """
    last_index = -1
    for i, msg in enumerate(messages):
        msg_type = None
        if isinstance(msg, dict):
            msg_type = msg.get("type")
        elif isinstance(msg, (list, tuple)) and len(msg) >= 1:
            msg_type = msg[0]

        if msg_type == "human":
            last_index = i

    return last_index
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_history_image.py::TestFindLastHumanIndex -v`
Expected: PASS

### Step 5: Commit

```bash
git add helper/history_image.py tests/test_history_image.py
git commit -m "feat(history_image): add find_last_human_index function"
```

---

## Task 3: Add replace_images_with_placeholders Function

**Files:**
- Modify: `helper/history_image.py`
- Modify: `tests/test_history_image.py`

### Step 1: Write the failing test

```python
# Add to tests/test_history_image.py
import pytest


class TestReplaceImagesWithPlaceholders:
    """Tests for replace_images_with_placeholders function."""

    @pytest.fixture
    def mock_redis(self, mocker):
        """Mock Redis manager."""
        mock = mocker.AsyncMock()
        mock.set_cache = mocker.AsyncMock(return_value=True)
        return mock

    @pytest.mark.asyncio
    async def test_replace_single_image(self, mock_redis):
        """Should replace a single image with placeholder."""
        from helper.history_image import replace_images_with_placeholders

        content = [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc123"}},
            {"type": "text", "text": "What is this?"},
        ]

        result = await replace_images_with_placeholders(content, mock_redis)

        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[0]["text"].startswith("[Picture:history_")
        assert result[1] == {"type": "text", "text": "What is this?"}
        mock_redis.set_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_preserve_non_base64_images(self, mock_redis):
        """Should preserve images that are not base64 encoded."""
        from helper.history_image import replace_images_with_placeholders

        content = [
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
            {"type": "text", "text": "What is this?"},
        ]

        result = await replace_images_with_placeholders(content, mock_redis)

        assert len(result) == 2
        assert result[0] == content[0]  # Unchanged
        mock_redis.set_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_string_content_unchanged(self, mock_redis):
        """Should return string content unchanged."""
        from helper.history_image import replace_images_with_placeholders

        content = "Just a text message"

        result = await replace_images_with_placeholders(content, mock_redis)

        assert result == content
        mock_redis.set_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_stores_correct_data(self, mock_redis):
        """Should store base64 and mime type in cache."""
        from helper.history_image import replace_images_with_placeholders
        import hashlib

        base64_data = "abc123"
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_data}"}},
        ]

        await replace_images_with_placeholders(content, mock_redis)

        expected_md5 = hashlib.md5(base64_data.encode()).hexdigest()
        call_args = mock_redis.set_cache.call_args
        assert call_args[0][0] == f"history_image_{expected_md5}"
        cached_data = json.loads(call_args[0][1])
        assert cached_data["data"] == base64_data
        assert cached_data["mime_type"] == "image/png"
        assert call_args[1]["ex"] == 7200  # 2 hours
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_history_image.py::TestReplaceImagesWithPlaceholders -v`
Expected: FAIL with "cannot import name 'replace_images_with_placeholders'"

### Step 3: Write minimal implementation

```python
# Add to helper/history_image.py

async def replace_images_with_placeholders(
    content: Union[str, List[Dict[str, Any]]],
    redis_manager: Any,
) -> Union[str, List[Dict[str, Any]]]:
    """Replace base64 images with placeholders and cache them.

    Args:
        content: Message content (string or list of content blocks)
        redis_manager: Redis manager instance for caching

    Returns:
        Processed content with images replaced by placeholders
    """
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return content

    new_content = []
    for item in content:
        if not isinstance(item, dict):
            new_content.append(item)
            continue

        if item.get("type") != "image_url":
            new_content.append(item)
            continue

        image_url_data = item.get("image_url", {})
        url = image_url_data.get("url", "")

        # Only process base64 data URLs
        extracted = extract_base64_and_mime(url)
        if not extracted:
            new_content.append(item)
            continue

        base64_data, mime_type = extracted

        # Calculate MD5 hash
        md5_hash = hashlib.md5(base64_data.encode()).hexdigest()

        # Cache the image data
        cache_key = f"history_image_{md5_hash}"
        cache_value = json.dumps({"data": base64_data, "mime_type": mime_type})

        try:
            await redis_manager.set_cache(cache_key, cache_value, ex=HISTORY_IMAGE_TTL)
        except Exception as e:
            logger.warning(f"Failed to cache history image: {e}")
            # Fallback: keep original image
            new_content.append(item)
            continue

        # Replace with placeholder
        new_content.append({
            "type": "text",
            "text": f"[Picture:history_{md5_hash}]"
        })

    return new_content
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_history_image.py::TestReplaceImagesWithPlaceholders -v`
Expected: PASS

### Step 5: Commit

```bash
git add helper/history_image.py tests/test_history_image.py
git commit -m "feat(history_image): add replace_images_with_placeholders function"
```

---

## Task 4: Add process_history_images Main Function

**Files:**
- Modify: `helper/history_image.py`
- Modify: `tests/test_history_image.py`

### Step 1: Write the failing test

```python
# Add to tests/test_history_image.py

class TestProcessHistoryImages:
    """Tests for process_history_images function."""

    @pytest.fixture
    def mock_redis(self, mocker):
        """Mock Redis manager."""
        mock = mocker.AsyncMock()
        mock.set_cache = mocker.AsyncMock(return_value=True)
        return mock

    @pytest.mark.asyncio
    async def test_replace_only_historical_images(self, mock_redis):
        """Should replace images only in historical messages, not the last human message."""
        from helper.history_image import process_history_images

        messages = [
            {"type": "human", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,first"}},
                {"type": "text", "text": "What is this?"},
            ]},
            {"type": "assistant", "content": "This is a cat."},
            {"type": "human", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,last"}},
                {"type": "text", "text": "And this?"},
            ]},
        ]

        result = await process_history_images(messages, mock_redis)

        # First human message should have placeholder
        assert result[0]["content"][0]["type"] == "text"
        assert "[Picture:history_" in result[0]["content"][0]["text"]

        # Last human message should keep original image
        assert result[2]["content"][0]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_single_human_message_unchanged(self, mock_redis):
        """Should not replace images when there's only one human message."""
        from helper.history_image import process_history_images

        messages = [
            {"type": "human", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,only"}},
                {"type": "text", "text": "What is this?"},
            ]},
        ]

        result = await process_history_images(messages, mock_redis)

        # Should keep original image
        assert result[0]["content"][0]["type"] == "image_url"
        mock_redis.set_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_images_unchanged(self, mock_redis):
        """Should return messages unchanged when there are no images."""
        from helper.history_image import process_history_images

        messages = [
            {"type": "human", "content": "Hello"},
            {"type": "assistant", "content": "Hi there!"},
            {"type": "human", "content": "How are you?"},
        ]

        result = await process_history_images(messages, mock_redis)

        assert result == messages
        mock_redis.set_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_tuple_format_messages(self, mock_redis):
        """Should handle tuple format messages."""
        from helper.history_image import process_history_images

        messages = [
            ("human", [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,first"}},
                {"type": "text", "text": "What is this?"},
            ]),
            ("assistant", "This is a cat."),
            ("human", [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,last"}},
                {"type": "text", "text": "And this?"},
            ]),
        ]

        result = await process_history_images(messages, mock_redis)

        # First human message should have placeholder
        assert result[0][1][0]["type"] == "text"
        assert "[Picture:history_" in result[0][1][0]["text"]

        # Last human message should keep original image
        assert result[2][1][0]["type"] == "image_url"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_history_image.py::TestProcessHistoryImages -v`
Expected: FAIL with "cannot import name 'process_history_images'"

### Step 3: Write minimal implementation

```python
# Add to helper/history_image.py

async def process_history_images(
    messages: List[Any],
    redis_manager: Any,
) -> List[Any]:
    """Process messages to replace historical images with placeholders.

    The last human message keeps its images intact. All other human messages
    have their images replaced with placeholders.

    Args:
        messages: List of messages (dict or tuple format)
        redis_manager: Redis manager instance for caching

    Returns:
        Processed messages with historical images replaced
    """
    if not messages:
        return messages

    last_human_idx = find_last_human_index(messages)

    # If no human messages or only one, return unchanged
    if last_human_idx < 0:
        return messages

    result = []
    for i, msg in enumerate(messages):
        # Determine message type and content
        if isinstance(msg, dict):
            msg_type = msg.get("type")
            msg_content = msg.get("content")
        elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
            msg_type = msg[0]
            msg_content = msg[1]
        else:
            result.append(msg)
            continue

        # Skip if not human or is the last human message
        if msg_type != "human" or i == last_human_idx:
            result.append(msg)
            continue

        # Process content to replace images
        processed_content = await replace_images_with_placeholders(msg_content, redis_manager)

        # Rebuild message in original format
        if isinstance(msg, dict):
            new_msg = dict(msg)
            new_msg["content"] = processed_content
            result.append(new_msg)
        elif isinstance(msg, tuple):
            result.append((msg_type, processed_content) + msg[2:] if len(msg) > 2 else (msg_type, processed_content))
        else:
            result.append([msg_type, processed_content] + list(msg[2:]) if len(msg) > 2 else [msg_type, processed_content])

    return result
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_history_image.py::TestProcessHistoryImages -v`
Expected: PASS

### Step 5: Commit

```bash
git add helper/history_image.py tests/test_history_image.py
git commit -m "feat(history_image): add process_history_images main function"
```

---

## Task 5: Create Built-in Tools Module

**Files:**
- Create: `helper/tools.py`
- Create: `tests/test_tools.py`

### Step 1: Write the failing test

```python
# tests/test_tools.py
import pytest
import json


class TestGetHistoryImageTool:
    """Tests for GetHistoryImageTool."""

    @pytest.fixture
    def mock_redis(self, mocker):
        """Mock Redis manager."""
        mock = mocker.AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_get_existing_image(self, mock_redis, mocker):
        """Should return image content block for existing image."""
        from helper.tools import GetHistoryImageTool

        cached_data = json.dumps({
            "data": "abc123base64data",
            "mime_type": "image/png"
        })
        mock_redis.get_cache = mocker.AsyncMock(return_value=cached_data)

        tool = GetHistoryImageTool(redis_manager=mock_redis)
        result = await tool._arun(image_md5="abc123def456")

        content, artifact = result
        assert len(content) == 1
        assert content[0]["type"] == "image"
        assert content[0]["base64"] == "abc123base64data"
        assert content[0]["mime_type"] == "image/png"
        assert artifact is None

    @pytest.mark.asyncio
    async def test_get_image_with_history_prefix(self, mock_redis, mocker):
        """Should handle image_md5 with history_ prefix."""
        from helper.tools import GetHistoryImageTool

        cached_data = json.dumps({
            "data": "imagedata",
            "mime_type": "image/jpeg"
        })
        mock_redis.get_cache = mocker.AsyncMock(return_value=cached_data)

        tool = GetHistoryImageTool(redis_manager=mock_redis)
        await tool._arun(image_md5="history_abc123def456")

        mock_redis.get_cache.assert_called_with("history_image_abc123def456")

    @pytest.mark.asyncio
    async def test_image_not_found(self, mock_redis, mocker):
        """Should return error message for non-existent image."""
        from helper.tools import GetHistoryImageTool

        mock_redis.get_cache = mocker.AsyncMock(return_value="")

        tool = GetHistoryImageTool(redis_manager=mock_redis)
        result = await tool._arun(image_md5="nonexistent123")

        content, artifact = result
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert "不存在或已过期" in content[0]["text"]

    @pytest.mark.asyncio
    async def test_invalid_image_md5(self, mock_redis):
        """Should return error for invalid image_md5."""
        from helper.tools import GetHistoryImageTool

        tool = GetHistoryImageTool(redis_manager=mock_redis)
        result = await tool._arun(image_md5="short")

        content, artifact = result
        assert content[0]["type"] == "text"
        assert "无效" in content[0]["text"]

    def test_tool_metadata(self, mock_redis):
        """Should have correct metadata."""
        from helper.tools import GetHistoryImageTool

        tool = GetHistoryImageTool(redis_manager=mock_redis)

        assert tool.name == "get_history_image"
        assert "历史" in tool.description
        assert tool.response_format == "content_and_artifact"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_tools.py::TestGetHistoryImageTool -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'helper.tools'"

### Step 3: Write minimal implementation

```python
# helper/tools.py
"""
Built-in Tools Module

Defines internal tools that are loaded alongside MCP tools.
"""

import json
import logging
from typing import Any, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger("ai")


class GetHistoryImageInput(BaseModel):
    """Input schema for get_history_image tool."""

    image_md5: str = Field(
        description="The MD5 hash of the history image (from [Picture:history_xxx] placeholder)"
    )


class GetHistoryImageTool(BaseTool):
    """Tool for retrieving historical images from cache."""

    name: str = "get_history_image"
    description: str = """获取历史对话中用户上传的图片。
当用户询问历史图片的细节（如"刚才那张图的右上角是什么"）时，
使用此工具获取图片内容进行分析。
输入参数为图片的 MD5 哈希值（从 [Picture:history_xxx] 占位符中提取）。"""

    args_schema: Type[BaseModel] = GetHistoryImageInput
    response_format: str = "content_and_artifact"

    redis_manager: Any = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, image_md5: str) -> tuple:
        """Sync version - not implemented."""
        raise NotImplementedError("Use async version")

    async def _arun(self, image_md5: str) -> tuple:
        """Retrieve a historical image from cache.

        Args:
            image_md5: MD5 hash of the image (with or without 'history_' prefix)

        Returns:
            Tuple of (content_blocks, artifact)
        """
        # Validate input
        if not image_md5 or len(image_md5) < 8:
            return ([{"type": "text", "text": "无效的图片标识符"}], None)

        # Normalize key (support with or without history_ prefix)
        if image_md5.startswith("history_"):
            md5_hash = image_md5[8:]
        else:
            md5_hash = image_md5

        # Retrieve from cache
        cache_key = f"history_image_{md5_hash}"
        try:
            cached = await self.redis_manager.get_cache(cache_key)
        except Exception as e:
            logger.error(f"Failed to get history image: {e}")
            return ([{"type": "text", "text": f"获取图片失败: {e}"}], None)

        if not cached:
            return ([{"type": "text", "text": "图片不存在或已过期"}], None)

        # Parse cached data
        try:
            cache_data = json.loads(cached)
            base64_data = cache_data["data"]
            mime_type = cache_data.get("mime_type", "image/jpeg")
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Invalid cache data for history image: {e}")
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
        GetHistoryImageTool(redis_manager=redis_manager)
    ]
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_tools.py::TestGetHistoryImageTool -v`
Expected: PASS

### Step 5: Commit

```bash
git add helper/tools.py tests/test_tools.py
git commit -m "feat(tools): add get_history_image built-in tool"
```

---

## Task 6: Integrate Built-in Tools with MCP Tools

**Files:**
- Modify: `helper/mcp.py`
- Modify: `tests/test_tools.py`

### Step 1: Write the failing test

```python
# Add to tests/test_tools.py

class TestLoadBuiltinToolsIntegration:
    """Tests for load_builtin_tools integration."""

    def test_load_builtin_tools_returns_list(self, mocker):
        """Should return a list of tools."""
        from helper.tools import load_builtin_tools

        mock_redis = mocker.AsyncMock()
        tools = load_builtin_tools(mock_redis)

        assert isinstance(tools, list)
        assert len(tools) >= 1
        assert tools[0].name == "get_history_image"
```

### Step 2: Run test to verify it passes

Run: `pytest tests/test_tools.py::TestLoadBuiltinToolsIntegration -v`
Expected: PASS (already implemented)

### Step 3: Modify mcp.py to include built-in tools

Modify `helper/mcp.py`, add at the end of `load_mcp_tools_for_model` function:

```python
# In helper/mcp.py, modify load_mcp_tools_for_model function
# Add after the existing tool loading logic, before the return statement

async def load_mcp_tools_for_model(
    model_name: str,
    *,
    dootask_available: bool,
    token_candidates: List[Optional[str]],
    redis_manager: Optional[Any] = None,  # Add this parameter
) -> List[object]:
    """根据配置文件加载与当前模型匹配的 MCP 工具列表。"""
    # ... existing code ...

    if not server_configs:
        mcp_tools = []
    else:
        client = MultiServerMCPClient(server_configs)
        try:
            mcp_tools = await client.get_tools()
        except Exception as exc:
            logger.error("Failed to load MCP tools: %s", exc)
            mcp_tools = []

    # Load built-in tools if redis_manager provided
    builtin_tools = []
    if redis_manager is not None:
        from helper.tools import load_builtin_tools
        builtin_tools = load_builtin_tools(redis_manager)

    # Combine and wrap all tools
    all_tools = list(mcp_tools) + builtin_tools
    return [_wrap_tool_with_error_handling(tool) for tool in all_tools]
```

### Step 4: Run existing tests to ensure no regression

Run: `pytest tests/ -v`
Expected: PASS

### Step 5: Commit

```bash
git add helper/mcp.py
git commit -m "feat(mcp): integrate built-in tools with MCP tools"
```

---

## Task 7: Integrate History Image Processing in invoke/stream Endpoint

**Files:**
- Modify: `main.py` (around line 697-708)

### Step 1: Identify the modification point

Current code at `main.py:697-708`:
```python
# 处理上下文中的图片内容
vision_config = load_vision_config()
model_name = data.get("model_name", "")
processed_context = []
for msg in parsed_context:
    if hasattr(msg, 'content') and isinstance(msg.content, list):
        # Multimodal message - process images
        processed_content = await process_vision_content(msg.content, model_name, vision_config)
        processed_context.append(type(msg)(content=processed_content))
    else:
        processed_context.append(msg)
parsed_context = processed_context
```

### Step 2: Add history image processing before vision processing

Modify `main.py` to add history image processing:

```python
# At the top of the file, add import
from helper.history_image import process_history_images

# In invoke_stream function, around line 697, add BEFORE the vision processing loop:

# 处理历史图片（替换为占位符）
history_processed_context = await process_history_images(
    parsed_context,
    app.state.redis_manager
)
parsed_context = history_processed_context

# 处理上下文中的图片内容（现有代码）
vision_config = load_vision_config()
# ... rest of existing code
```

### Step 3: Update tool loading to pass redis_manager

Modify `main.py` around line 756-760:

```python
# Change from:
tools = await load_mcp_tools_for_model(
    data.get("model_name", ""),
    dootask_available=bool(getattr(app.state, "dootask_mcp", False)),
    token_candidates=[data.get("user_token"), data.get("token")],
)

# To:
tools = await load_mcp_tools_for_model(
    data.get("model_name", ""),
    dootask_available=bool(getattr(app.state, "dootask_mcp", False)),
    token_candidates=[data.get("user_token"), data.get("token")],
    redis_manager=app.state.redis_manager,
)
```

### Step 4: Commit

```bash
git add main.py
git commit -m "feat(invoke): integrate history image processing in direct mode"
```

---

## Task 8: Integrate History Image Processing in stream Endpoint

**Files:**
- Modify: `main.py` (inside stream_generate function, around line 336)

### Step 1: Identify the modification point

Current code flow in `stream_generate` function:
1. Build pre_context (system messages)
2. Get middle_context from Redis
3. Build end_context (current user message)
4. Call handle_context_limits

### Step 2: Add history image processing

Modify `main.py` inside `stream_generate` function, after building middle_messages and before handle_context_limits:

```python
# After line 361 (middle_messages = [dict_to_message(msg_dict) for msg_dict in middle_context])
# Add:

# 处理历史图片（替换中间上下文中的图片为占位符）
if middle_messages:
    from helper.history_image import process_history_images
    # Convert messages to dict format for processing
    middle_dicts = [{"type": "human" if isinstance(m, HumanMessage) else "assistant", "content": m.content} for m in middle_messages]
    processed_middle = await process_history_images(middle_dicts, redis_manager)
    middle_messages = [dict_to_message(msg_dict) for msg_dict in processed_middle]
```

Note: This is more complex because the stream endpoint uses LangChain message objects. We need to handle the conversion properly.

### Step 3: Update tool loading to pass redis_manager

Modify `main.py` around line 290-294:

```python
# Change from:
tools = await load_mcp_tools_for_model(
    data.get("model_name", ""),
    dootask_available=dootask_available,
    token_candidates=[data.get("msg_user_token"), data.get("token")],
)

# To:
tools = await load_mcp_tools_for_model(
    data.get("model_name", ""),
    dootask_available=dootask_available,
    token_candidates=[data.get("msg_user_token"), data.get("token")],
    redis_manager=app.state.redis_manager,
)
```

### Step 4: Commit

```bash
git add main.py
git commit -m "feat(stream): integrate history image processing in dialog mode"
```

---

## Task 9: Add Integration Tests

**Files:**
- Create: `tests/test_history_image_integration.py`

### Step 1: Write integration tests

```python
# tests/test_history_image_integration.py
"""Integration tests for history image processing."""
import pytest
import json


class TestHistoryImageIntegration:
    """Integration tests for the complete history image flow."""

    @pytest.fixture
    def mock_redis(self, mocker):
        """Mock Redis manager with both set and get."""
        mock = mocker.AsyncMock()
        mock._cache = {}

        async def mock_set_cache(key, value, **kwargs):
            mock._cache[key] = value
            return True

        async def mock_get_cache(key):
            return mock._cache.get(key, "")

        mock.set_cache = mock_set_cache
        mock.get_cache = mock_get_cache
        return mock

    @pytest.mark.asyncio
    async def test_full_flow_process_then_retrieve(self, mock_redis):
        """Should be able to process images and then retrieve them."""
        from helper.history_image import process_history_images
        from helper.tools import GetHistoryImageTool

        # Simulate a conversation with images
        messages = [
            {"type": "human", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,firstimage"}},
                {"type": "text", "text": "What is this?"},
            ]},
            {"type": "assistant", "content": "This is a cat."},
            {"type": "human", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,lastimage"}},
                {"type": "text", "text": "And this?"},
            ]},
        ]

        # Process messages
        processed = await process_history_images(messages, mock_redis)

        # Extract MD5 from placeholder
        import re
        placeholder = processed[0]["content"][0]["text"]
        match = re.search(r"\[Picture:history_([a-f0-9]+)\]", placeholder)
        assert match, f"No placeholder found in: {placeholder}"
        md5_hash = match.group(1)

        # Retrieve using tool
        tool = GetHistoryImageTool(redis_manager=mock_redis)
        result = await tool._arun(image_md5=md5_hash)

        content, artifact = result
        assert content[0]["type"] == "image"
        assert content[0]["base64"] == "firstimage"
        assert content[0]["mime_type"] == "image/jpeg"

    @pytest.mark.asyncio
    async def test_expired_image_returns_error(self, mock_redis, mocker):
        """Should return error when image has expired."""
        from helper.tools import GetHistoryImageTool

        # Simulate expired cache (returns empty)
        mock_redis.get_cache = mocker.AsyncMock(return_value="")

        tool = GetHistoryImageTool(redis_manager=mock_redis)
        result = await tool._arun(image_md5="expiredmd5hash12345678901234")

        content, _ = result
        assert "过期" in content[0]["text"]
```

### Step 2: Run integration tests

Run: `pytest tests/test_history_image_integration.py -v`
Expected: PASS

### Step 3: Commit

```bash
git add tests/test_history_image_integration.py
git commit -m "test: add integration tests for history image processing"
```

---

## Task 10: Final Verification and Cleanup

### Step 1: Run all tests

Run: `pytest tests/ -v`
Expected: All PASS

### Step 2: Check for linting issues

Run: `ruff check helper/history_image.py helper/tools.py`
Expected: No errors (or fix any issues)

### Step 3: Final commit

```bash
git add -A
git commit -m "feat: complete history image processing implementation

- Add helper/history_image.py for processing historical images
- Add helper/tools.py with get_history_image built-in tool
- Integrate with both /invoke/stream and /stream endpoints
- Images cached in Redis with 2-hour TTL
- AI can retrieve historical images on demand"
```

---

## Summary

| Task | Files | Description |
|------|-------|-------------|
| 1 | `helper/history_image.py` | Create module with extract_base64_and_mime |
| 2 | `helper/history_image.py` | Add find_last_human_index |
| 3 | `helper/history_image.py` | Add replace_images_with_placeholders |
| 4 | `helper/history_image.py` | Add process_history_images main function |
| 5 | `helper/tools.py` | Create GetHistoryImageTool |
| 6 | `helper/mcp.py` | Integrate built-in tools |
| 7 | `main.py` | Integrate in /invoke/stream endpoint |
| 8 | `main.py` | Integrate in /stream endpoint |
| 9 | `tests/` | Add integration tests |
| 10 | - | Final verification |
