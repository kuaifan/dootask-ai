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
        import json

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
