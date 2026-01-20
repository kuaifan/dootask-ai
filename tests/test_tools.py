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
