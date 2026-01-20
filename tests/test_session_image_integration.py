# tests/test_session_image_integration.py
"""Integration tests for session image processing."""
import re

import pytest


class TestSessionImageIntegration:
    """Integration tests for the complete session image flow."""

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
        from helper.session_image import process_session_images
        from helper.tools import GetSessionImageTool

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
        processed = await process_session_images(messages, mock_redis)

        # Extract MD5 from placeholder
        placeholder = processed[0]["content"][0]["text"]
        match = re.search(r"\[picture:session_([a-f0-9]+)\]", placeholder)
        assert match, f"No placeholder found in: {placeholder}"
        md5_hash = match.group(1)

        # Retrieve using tool
        tool = GetSessionImageTool(redis_manager=mock_redis)
        result = await tool._arun(image_md5=md5_hash)

        content, artifact = result
        assert content[0]["type"] == "image"
        assert content[0]["base64"] == "firstimage"
        assert content[0]["mime_type"] == "image/jpeg"

    @pytest.mark.asyncio
    async def test_expired_image_returns_error(self, mock_redis, mocker):
        """Should return error when image has expired."""
        from helper.tools import GetSessionImageTool

        # Simulate expired cache (returns empty)
        mock_redis.get_cache = mocker.AsyncMock(return_value="")

        tool = GetSessionImageTool(redis_manager=mock_redis)
        result = await tool._arun(image_md5="expiredmd5hash12345678901234")

        content, _ = result
        assert "过期" in content[0]["text"]
