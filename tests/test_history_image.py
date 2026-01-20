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
