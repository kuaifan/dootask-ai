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
