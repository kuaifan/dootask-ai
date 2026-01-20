"""
Vision configuration and image processing module for DooTask AI.

This module provides functionality for:
- Loading and saving vision configuration
- Checking vision capabilities for models
- Processing images for multimodal content or file storage
- Cleaning up old image files
"""

import base64
import json
import logging
import uuid
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from helper.config import (
    DEFAULT_MODELS,
    VISION_CONFIG_PATH,
    VISION_DATA_DIR,
    VISION_PREVIEW_URL_PREFIX,
    VISION_CLEANUP_DAYS,
)

logger = logging.getLogger(__name__)


class VisionConfigError(Exception):
    """Raised when vision configuration read/write fails."""


def _get_default_vision_config() -> Dict[str, Any]:
    """
    Returns the default vision configuration.

    Returns:
        Dict containing default vision settings:
        - enabled: Whether vision is globally enabled
        - supported_models: List of model names that support vision
        - max_image_size: Maximum image dimension (width/height)
        - max_file_size_mb: Maximum file size in MB
        - default_quality: Default JPEG quality for compression
    """
    return {
        "enabled": False,
        "supportedModels": [],
        "maxImageSize": 2048,
        "maxFileSize": 10,
        "compressionQuality": 80,
    }


def _normalize_vision_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize vision configuration.

    Ensures all required fields exist with proper types and valid ranges.

    Args:
        config: Raw configuration dict

    Returns:
        Normalized configuration dict
    """
    defaults = _get_default_vision_config()

    # Start with defaults and update with provided config
    normalized = dict(defaults)
    normalized.update(config)

    # Ensure enabled is boolean
    normalized["enabled"] = bool(normalized.get("enabled", False))

    # Ensure supportedModels is a list of proper format
    supported = normalized.get("supportedModels")
    if not isinstance(supported, list):
        normalized["supportedModels"] = []
    else:
        # Normalize each model entry
        valid_models = []
        for item in supported:
            if isinstance(item, dict) and item.get("id"):
                valid_models.append({
                    "id": str(item["id"]),
                    "name": str(item.get("name", item["id"])),
                })
            elif isinstance(item, str):
                valid_models.append({"id": item, "name": item})
        normalized["supportedModels"] = valid_models

    # Ensure numeric fields are within valid ranges
    normalized["maxImageSize"] = max(256, min(8192, int(normalized.get("maxImageSize", 2048))))
    normalized["maxFileSize"] = max(1, min(50, int(normalized.get("maxFileSize", 10))))
    normalized["compressionQuality"] = max(1, min(100, int(normalized.get("compressionQuality", 80))))

    return normalized


def load_vision_config() -> Dict[str, Any]:
    """
    Load vision configuration from JSON file.

    If the config file doesn't exist or is invalid, returns default config.

    Returns:
        Dict containing vision configuration
    """
    try:
        if VISION_CONFIG_PATH.exists():
            with open(VISION_CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                return _normalize_vision_config(config)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load vision config: {e}, using defaults")

    return _get_default_vision_config()


def save_vision_config(config: Dict[str, Any]) -> None:
    """
    Save vision configuration to JSON file.

    Args:
        config: Vision configuration dict to save

    Raises:
        VisionConfigError: If save fails
    """
    try:
        # Normalize config before saving
        normalized = _normalize_vision_config(config)

        # Ensure config directory exists
        VISION_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(VISION_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(normalized, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logger.error(f"Failed to save vision config: {e}")
        raise VisionConfigError(f"Failed to save vision config: {e}") from e


def ensure_default_vision_config() -> None:
    """
    Ensure vision configuration exists with default values.

    If the config file doesn't exist, creates it with:
    - enabled: True
    - supportedModels: all vision-capable models
    - default image processing settings
    """
    if VISION_CONFIG_PATH.exists():
        return

    # Collect all vision-capable models for default config
    vision_models = collect_vision_capable_models()
    default_supported = [
        {"id": m["id"], "name": m["name"]}
        for m in vision_models
    ]

    default_config = {
        "enabled": True,
        "supportedModels": default_supported,
        "maxImageSize": 2048,
        "maxFileSize": 10,
        "compressionQuality": 80,
    }

    try:
        save_vision_config(default_config)
        logger.info("✅ 已写入默认视觉识别配置")
    except Exception as e:
        logger.error(f"❌ 写入视觉识别配置失败: {e}")


def is_vision_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Check if vision is globally enabled.

    Args:
        config: Optional pre-loaded config. If None, loads from file.

    Returns:
        True if vision is enabled globally
    """
    if config is None:
        config = load_vision_config()
    return config.get("enabled", False)


def model_supports_vision_capability(model_name: str) -> bool:
    """
    Check if a model has vision capability based on DEFAULT_MODELS.

    Args:
        model_name: The model identifier to check

    Returns:
        True if the model has support_vision=True in DEFAULT_MODELS
    """
    if not model_name:
        return False

    for provider_models in DEFAULT_MODELS.values():
        for model in provider_models:
            if model.get("id") == model_name:
                return model.get("support_vision", False)

    return False


def model_in_vision_supported_list(
    model_name: str,
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Check if a model is in the configured vision supported list.

    Args:
        model_name: The model identifier to check
        config: Optional pre-loaded config. If None, loads from file.

    Returns:
        True if the model is in the supported_models list
    """
    if config is None:
        config = load_vision_config()

    supported_models = config.get("supportedModels", [])
    return any(m.get("id") == model_name for m in supported_models)


def should_use_vision_directly(
    model_name: str,
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Determine if vision should be used directly for a model.

    Returns True only if:
    1. Vision is globally enabled AND
    2. The model is in the configured supported models list

    Args:
        model_name: The model identifier to check
        config: Optional pre-loaded config. If None, loads from file.

    Returns:
        True if vision content should be sent directly to the model
    """
    if config is None:
        config = load_vision_config()

    return is_vision_enabled(config) and model_in_vision_supported_list(model_name, config)


def collect_vision_capable_models() -> List[Dict[str, Any]]:
    """
    Get all models that have vision capability from DEFAULT_MODELS.

    Returns:
        List of model dicts that have support_vision=True
    """
    vision_models = []

    for provider, models in DEFAULT_MODELS.items():
        for model in models:
            if model.get("support_vision", False):
                vision_models.append({
                    "provider": provider,
                    "id": model["id"],
                    "name": model.get("name", model["id"]),
                })

    return vision_models


def ensure_vision_data_dir() -> Path:
    """
    Create the vision data directory if it doesn't exist.

    Returns:
        Path to the vision data directory
    """
    VISION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return VISION_DATA_DIR


def save_image_to_file(image_data: bytes, ext: str = "jpg") -> str:
    """
    Save image bytes to a file in the vision data directory.

    Args:
        image_data: Raw image bytes to save
        ext: File extension (default: jpg)

    Returns:
        The generated filename (without path)
    """
    ensure_vision_data_dir()

    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    filename = f"{timestamp}_{unique_id}.{ext}"

    file_path = VISION_DATA_DIR / filename
    with open(file_path, "wb") as f:
        f.write(image_data)

    logger.debug(f"Saved image to {file_path}")
    return filename


def get_image_url(filename: str) -> str:
    """
    Get the full URL for an image file.

    Args:
        filename: The filename of the image

    Returns:
        Full URL for accessing the image
    """
    return f"{VISION_PREVIEW_URL_PREFIX}/{filename}"


def decode_base64_image(data_url: str) -> tuple[bytes, str]:
    """
    Decode a base64 data URL to raw bytes and determine extension.

    Args:
        data_url: Base64 data URL (e.g., "data:image/jpeg;base64,...")

    Returns:
        Tuple of (image_bytes, extension)

    Raises:
        ValueError: If the data URL format is invalid
    """
    if not data_url.startswith("data:"):
        raise ValueError("Invalid data URL format")

    # Parse the data URL
    try:
        header, base64_data = data_url.split(",", 1)
    except ValueError:
        raise ValueError("Invalid data URL format: missing comma separator")

    # Extract MIME type
    mime_part = header.split(";")[0]  # e.g., "data:image/jpeg"
    mime_type = mime_part.replace("data:", "")  # e.g., "image/jpeg"

    # Map MIME types to extensions
    mime_to_ext = {
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp",
        "image/bmp": "bmp",
    }

    ext = mime_to_ext.get(mime_type, "jpg")

    # Decode base64
    try:
        image_bytes = base64.b64decode(base64_data)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 data: {e}")

    return image_bytes, ext


def process_image(
    image_bytes: bytes,
    max_size: int = 2048,
    max_file_size_mb: float = 5.0,
    quality: int = 85
) -> bytes:
    """
    Process an image: resize if too large and compress if needed.

    Args:
        image_bytes: Raw image bytes
        max_size: Maximum dimension (width or height)
        max_file_size_mb: Maximum file size in MB
        quality: JPEG quality for compression (1-100)

    Returns:
        Processed image bytes (JPEG format)
    """
    try:
        img = Image.open(BytesIO(image_bytes))

        # Convert to RGB if necessary (for PNG with alpha, etc.)
        if img.mode in ("RGBA", "P", "LA"):
            # Create white background for transparency
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # Resize if larger than max_size
        width, height = img.size
        if width > max_size or height > max_size:
            ratio = min(max_size / width, max_size / height)
            new_size = (int(width * ratio), int(height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"Resized image from {width}x{height} to {new_size}")

        # Save to bytes with compression
        output = BytesIO()
        img.save(output, format="JPEG", quality=quality, optimize=True)
        result = output.getvalue()

        # If still too large, reduce quality progressively
        max_bytes = int(max_file_size_mb * 1024 * 1024)
        current_quality = quality

        while len(result) > max_bytes and current_quality > 20:
            current_quality -= 10
            output = BytesIO()
            img.save(output, format="JPEG", quality=current_quality, optimize=True)
            result = output.getvalue()
            logger.debug(f"Compressed image to quality {current_quality}, size: {len(result)} bytes")

        return result

    except Exception as e:
        logger.error(f"Failed to process image: {e}")
        # Return original if processing fails
        return image_bytes


def encode_image_to_base64(image_bytes: bytes, ext: str = "jpg") -> str:
    """
    Encode image bytes to a base64 data URL.

    Args:
        image_bytes: Raw image bytes
        ext: File extension to determine MIME type

    Returns:
        Base64 data URL string
    """
    ext_to_mime = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
    }

    mime_type = ext_to_mime.get(ext.lower(), "image/jpeg")
    base64_data = base64.b64encode(image_bytes).decode("utf-8")

    return f"data:{mime_type};base64,{base64_data}"


async def process_vision_content(
    content: Union[str, List[Dict[str, Any]]],
    model_name: str,
    config: Optional[Dict[str, Any]] = None
) -> Union[str, List[Dict[str, Any]]]:
    """
    Process content that may contain images for vision-capable models.

    This function handles two scenarios:
    1. If vision is enabled AND model is in supported list:
       - Process images and return multimodal content
    2. Otherwise:
       - Save images to files and return URLs as text

    Args:
        content: Message content - either a string or list of content items
        model_name: The model being used
        config: Optional pre-loaded vision config

    Returns:
        Processed content - string or list depending on input and vision support
    """
    # If content is a string, return as-is
    if isinstance(content, str):
        return content

    # Load config if not provided
    if config is None:
        config = load_vision_config()

    # Determine if we should use vision directly
    use_vision = should_use_vision_directly(model_name, config)

    # Get processing parameters from config
    max_size = config.get("maxImageSize", 2048)
    max_file_size_mb = config.get("maxFileSize", 10)
    quality = config.get("compressionQuality", 80)

    # Process each content item
    processed_items = []
    text_parts = []

    for item in content:
        if not isinstance(item, dict):
            # Handle string items
            if isinstance(item, str):
                if use_vision:
                    processed_items.append({"type": "text", "text": item})
                else:
                    text_parts.append(item)
            continue

        item_type = item.get("type", "")

        if item_type == "text":
            # Text content
            text_content = item.get("text", "")
            if use_vision:
                processed_items.append({"type": "text", "text": text_content})
            else:
                text_parts.append(text_content)

        elif item_type == "image_url":
            # Image content
            image_url_data = item.get("image_url", {})
            url = image_url_data.get("url", "") if isinstance(image_url_data, dict) else str(image_url_data)

            if not url:
                continue

            try:
                # Check if it's a base64 data URL
                if url.startswith("data:"):
                    # Decode and process the image
                    image_bytes, ext = decode_base64_image(url)
                    processed_bytes = process_image(
                        image_bytes,
                        max_size=max_size,
                        max_file_size_mb=max_file_size_mb,
                        quality=quality
                    )

                    if use_vision:
                        # Re-encode and keep as multimodal content
                        new_data_url = encode_image_to_base64(processed_bytes, "jpg")
                        processed_items.append({
                            "type": "image_url",
                            "image_url": {"url": new_data_url}
                        })
                    else:
                        # Save to file and create URL text
                        filename = save_image_to_file(processed_bytes, "jpg")
                        image_url = get_image_url(filename)
                        text_parts.append(f"[图片: {image_url}]")
                else:
                    # Regular URL - keep as-is for vision, or convert to text
                    if use_vision:
                        processed_items.append({
                            "type": "image_url",
                            "image_url": {"url": url}
                        })
                    else:
                        text_parts.append(f"[图片: {url}]")

            except Exception as e:
                logger.error(f"Failed to process image: {e}")
                # On error, add as text reference
                text_parts.append("[图片: 处理失败]")
        else:
            # Unknown type - keep as-is for vision, ignore for text
            if use_vision:
                processed_items.append(item)

    # Return appropriate format
    if use_vision:
        return processed_items if processed_items else ""
    else:
        return " ".join(text_parts) if text_parts else ""


def cleanup_old_images(days: Optional[int] = None) -> int:
    """
    Delete images older than the specified number of days.

    Args:
        days: Number of days to keep images. Defaults to VISION_CLEANUP_DAYS.

    Returns:
        Number of files deleted
    """
    if days is None:
        days = VISION_CLEANUP_DAYS

    if not VISION_DATA_DIR.exists():
        return 0

    cutoff_time = datetime.now() - timedelta(days=days)
    deleted_count = 0

    try:
        for file_path in VISION_DATA_DIR.iterdir():
            if file_path.is_file():
                # Get file modification time
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

                if mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted old image: {file_path}")
                    except OSError as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")

    except Exception as e:
        logger.error(f"Error during image cleanup: {e}")

    if deleted_count > 0:
        logger.info(f"Cleaned up {deleted_count} old image files")

    return deleted_count
