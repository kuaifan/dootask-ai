# Vision Config Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add vision recognition configuration feature, allowing users to configure which models can receive images, with automatic fallback to URL + MCP OCR for unsupported models.

**Architecture:** Global configuration stored in `config/vision-config.json`, backend processing in `helper/vision.py`, frontend components mirroring MCP pattern, image storage in `data/vision/` with 7-day auto-cleanup.

**Tech Stack:** Python/FastAPI (backend), React/TypeScript/shadcn-ui (frontend), Pillow (image processing)

---

## Task 1: Add support_vision to DEFAULT_MODELS

**Files:**
- Modify: `helper/config.py:40-153`

**Step 1: Add support_vision field to all models**

Update each model in `DEFAULT_MODELS` to include `support_vision` field. Models with vision capability get `True`, others get `False`.

```python
# helper/config.py - Update DEFAULT_MODELS
# For openai section (line 41-75):
DEFAULT_MODELS = {
    "openai": [
        {"id": "gpt-5.2-chat-latest", "name": "GPT-5.2 Chat", "support_mcp": True, "support_vision": True},
        {"id": "gpt-5.2-codex", "name": "GPT-5.2 Codex", "support_mcp": True, "support_vision": True},
        {"id": "gpt-5.2-pro", "name": "GPT-5.2 Pro", "support_mcp": True, "support_vision": True},
        {"id": "gpt-5.2", "name": "GPT-5.2", "support_mcp": True, "support_vision": True},

        {"id": "gpt-5.1-chat-latest", "name": "GPT-5.1 Chat", "support_mcp": True, "support_vision": True},
        {"id": "gpt-5.1-codex-mini", "name": "GPT-5.1 Codex mini", "support_mcp": True, "support_vision": True},
        {"id": "gpt-5.1-codex", "name": "GPT-5.1 Codex", "support_mcp": True, "support_vision": True},
        {"id": "gpt-5.1", "name": "GPT-5.1", "support_mcp": True, "support_vision": True},

        {"id": "gpt-5-chat-latest", "name": "GPT-5 Chat", "support_mcp": True, "support_vision": True},
        {"id": "gpt-5-codex", "name": "GPT-5 Codex", "support_mcp": True, "support_vision": True},
        {"id": "gpt-5-mini", "name": "GPT-5 mini", "support_mcp": True, "support_vision": True},
        {"id": "gpt-5-nano", "name": "GPT-5 nano", "support_mcp": True, "support_vision": True},
        {"id": "gpt-5-pro", "name": "GPT-5 pro", "support_mcp": True, "support_vision": True},
        {"id": "gpt-5", "name": "GPT-5", "support_mcp": True, "support_vision": True},

        {"id": "gpt-4.1", "name": "GPT-4.1", "support_mcp": True, "support_vision": True},
        {"id": "gpt-4o", "name": "GPT-4o", "support_mcp": True, "support_vision": True},
        {"id": "gpt-4", "name": "GPT-4", "support_mcp": True, "support_vision": True},
        {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "support_mcp": True, "support_vision": True},
        {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "support_mcp": True, "support_vision": True},

        {"id": "o3", "name": "GPT-o3", "support_mcp": True, "support_vision": True},
        {"id": "o1", "name": "GPT-o1", "support_mcp": False, "support_vision": False},
        {"id": "o4-mini", "name": "GPT-o4 Mini", "support_mcp": True, "support_vision": True},
        {"id": "o3-mini", "name": "GPT-o3 Mini", "support_mcp": True, "support_vision": True},
        {"id": "o1-mini", "name": "GPT-o1 Mini", "support_mcp": False, "support_vision": False},

        {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "support_mcp": False, "support_vision": False},
        {"id": "gpt-3.5-turbo-16k", "name": "GPT-3.5 Turbo 16K", "support_mcp": False, "support_vision": False},
        {"id": "gpt-3.5-turbo-0125", "name": "GPT-3.5 Turbo 0125", "support_mcp": False, "support_vision": False},
        {"id": "gpt-3.5-turbo-1106", "name": "GPT-3.5 Turbo 1106", "support_mcp": False, "support_vision": False},
    ],
    "claude": [
        {"id": "claude-opus-4-5 (thinking)", "name": "Claude Opus 4.5", "support_mcp": True, "support_vision": True},
        {"id": "claude-sonnet-4-5 (thinking)", "name": "Claude Sonnet 4.5", "support_mcp": True, "support_vision": True},
        {"id": "claude-haiku-4-5 (thinking)", "name": "Claude Haiku 4.5", "support_mcp": True, "support_vision": True},

        {"id": "claude-opus-4-1 (thinking)", "name": "Claude Opus 4.1", "support_mcp": True, "support_vision": True},

        {"id": "claude-sonnet-4-0 (thinking)", "name": "Claude Sonnet 4.0", "support_mcp": True, "support_vision": True},
        {"id": "claude-opus-4-0 (thinking)", "name": "Claude Opus 4.0", "support_mcp": True, "support_vision": True},

        {"id": "claude-3-7-sonnet-latest (thinking)", "name": "Claude Sonnet 3.7", "support_mcp": True, "support_vision": True},
        {"id": "claude-3-5-haiku-latest", "name": "Claude Haiku 3.5", "support_mcp": True, "support_vision": True},
    ],
    "deepseek": [
        {"id": "deepseek-chat", "name": "DeepSeek-V3.2", "support_mcp": True, "support_vision": False},
        {"id": "deepseek-reasoner", "name": "DeepSeek-V3.2-Reasoner", "support_mcp": True, "support_vision": False},
    ],
    "gemini": [
        {"id": "gemini-3-pro-preview", "name": "Gemini 3 Pro", "support_mcp": True, "support_vision": True},
        {"id": "gemini-3-flash-preview", "name": "Gemini 3 Flash", "support_mcp": True, "support_vision": True},

        {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "support_mcp": True, "support_vision": True},
        {"id": "gemini-2.5-pro-tts", "name": "Gemini 2.5 Pro TTS", "support_mcp": True, "support_vision": True},
        {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "support_mcp": True, "support_vision": True},
        {"id": "gemini-2.5-flash-tts", "name": "Gemini 2.5 Flash TTS", "support_mcp": True, "support_vision": True},
        {"id": "gemini-2.5-flash-live", "name": "Gemini 2.5 Flash Live", "support_mcp": True, "support_vision": True},
        {"id": "gemini-2.5-flash-image", "name": "Gemini 2.5 Flash Image", "support_mcp": True, "support_vision": True},

        {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash", "support_mcp": False, "support_vision": True},
        {"id": "gemini-2.0-flash-lite", "name": "Gemini 2.0 Flash Lite", "support_mcp": False, "support_vision": True},
    ],
    "grok": [
        {"id": "grok-4-1-fast-reasoning", "name": "Grok 4.1 Fast Reasoning", "support_mcp": True, "support_vision": True},
        {"id": "grok-4-1-fast-non-reasoning", "name": "Grok 4.1 Fast", "support_mcp": True, "support_vision": True},
        {"id": "grok-code-fast-1", "name": "Grok Code Fast 1", "support_mcp": True, "support_vision": False},
        {"id": "grok-4-fast-reasoning", "name": "Grok 4 Fast Reasoning", "support_mcp": True, "support_vision": True},
        {"id": "grok-4-fast-non-reasoning", "name": "Grok 4 Fast", "support_mcp": True, "support_vision": True},
        {"id": "grok-4-0709", "name": "Grok 4", "support_mcp": True, "support_vision": True},
        {"id": "grok-3-latest", "name": "Grok 3", "support_mcp": False, "support_vision": True},
        {"id": "grok-3-fast-latest", "name": "Grok 3 Fast", "support_mcp": False, "support_vision": True},
    ],
    "zhipu": [
        {"id": "glm-4", "name": "GLM-4", "support_mcp": True, "support_vision": False},
        {"id": "glm-4-plus", "name": "GLM-4 Plus", "support_mcp": True, "support_vision": False},
        {"id": "glm-4-air", "name": "GLM-4 Air", "support_mcp": True, "support_vision": False},
        {"id": "glm-4-airx", "name": "GLM-4 AirX", "support_mcp": True, "support_vision": False},
        {"id": "glm-4-long", "name": "GLM-4 Long", "support_mcp": True, "support_vision": False},
        {"id": "glm-4-flash", "name": "GLM-4 Flash", "support_mcp": True, "support_vision": False},
        {"id": "glm-4v", "name": "GLM-4V", "support_mcp": False, "support_vision": True},
        {"id": "glm-4v-plus", "name": "GLM-4V Plus", "support_mcp": False, "support_vision": True},
        {"id": "glm-3-turbo", "name": "GLM-3 Turbo", "support_mcp": False, "support_vision": False},
    ],
    "qianwen": [
        {"id": "qwen-max", "name": "QWEN Max", "support_mcp": True, "support_vision": False},
        {"id": "qwen-max-latest", "name": "QWEN Max Latest", "support_mcp": False, "support_vision": False},
        {"id": "qwen-turbo", "name": "QWEN Turbo", "support_mcp": False, "support_vision": False},
        {"id": "qwen-turbo-latest", "name": "QWEN Turbo Latest", "support_mcp": False, "support_vision": False},
        {"id": "qwen-plus", "name": "QWEN Plus", "support_mcp": False, "support_vision": False},
        {"id": "qwen-plus-latest", "name": "QWEN Plus Latest", "support_mcp": False, "support_vision": False},
        {"id": "qwen-long", "name": "QWEN Long", "support_mcp": False, "support_vision": False},
    ],
    "wenxin": [
        {"id": "ernie-4.5-turbo-128k", "name": "Ernie 4.5 Turbo 128K", "support_mcp": True, "support_vision": False},
        {"id": "ernie-4.5-turbo-32k", "name": "Ernie 4.5 Turbo 32K", "support_mcp": True, "support_vision": False},
        {"id": "ernie-4.5-turbo-latest", "name": "Ernie 4.5 Turbo Latest", "support_mcp": True, "support_vision": False},
        {"id": "ernie-4.5-turbo-vl", "name": "Ernie 4.5 Turbo VL", "support_mcp": True, "support_vision": True},
        {"id": "ernie-4.0-8k", "name": "Ernie 4.0 8K", "support_mcp": False, "support_vision": False},
        {"id": "ernie-4.0-8k-latest", "name": "Ernie 4.0 8K Latest", "support_mcp": False, "support_vision": False},
        {"id": "ernie-4.0-turbo-128k", "name": "Ernie 4.0 Turbo 128K", "support_mcp": False, "support_vision": False},
        {"id": "ernie-4.0-turbo-8k", "name": "Ernie 4.0 Turbo 8K", "support_mcp": False, "support_vision": False},
        {"id": "ernie-3.5-128k", "name": "Ernie 3.5 128K", "support_mcp": False, "support_vision": False},
        {"id": "ernie-3.5-8k", "name": "Ernie 3.5 8K", "support_mcp": False, "support_vision": False},
        {"id": "ernie-speed-128k", "name": "Ernie Speed 128K", "support_mcp": False, "support_vision": False},
        {"id": "ernie-speed-8k", "name": "Ernie Speed 8K", "support_mcp": False, "support_vision": False},
        {"id": "ernie-lite-8k", "name": "Ernie Lite 8K", "support_mcp": False, "support_vision": False},
        {"id": "ernie-tiny-8k", "name": "Ernie Tiny 8K", "support_mcp": False, "support_vision": False},
    ],
}
```

**Step 2: Add vision config path constant**

Add after line 29 (after MCP config constants):

```python
# Vision 配置文件路径
VISION_CONFIG_PATH = BASE_DIR / "config" / "vision-config.json"
VISION_DATA_DIR = BASE_DIR / "data" / "vision"
VISION_PREVIEW_URL_PREFIX = "http://nginx/ai/vision/preview"
VISION_CLEANUP_DAYS = 7
VISION_CLEANUP_INTERVAL = 86400  # 24 hours in seconds
```

**Step 3: Commit**

```bash
git add helper/config.py
git commit -m "feat(vision): add support_vision field to DEFAULT_MODELS and vision config constants"
```

---

## Task 2: Create vision.py module

**Files:**
- Create: `helper/vision.py`

**Step 1: Create the vision module with core functions**

```python
# helper/vision.py
"""
Vision configuration and image processing module.
"""
import base64
import json
import logging
import os
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

logger = logging.getLogger("ai")

# Type definitions
VisionConfig = Dict[str, Any]
SupportedModel = Dict[str, str]


def _get_default_vision_config() -> VisionConfig:
    """Return default vision configuration."""
    return {
        "enabled": False,
        "supportedModels": [],
        "maxImageSize": 2048,
        "maxFileSize": 10,
        "compressionQuality": 80,
    }


def load_vision_config() -> VisionConfig:
    """Load vision configuration from file."""
    try:
        if VISION_CONFIG_PATH.exists():
            with open(VISION_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Merge with defaults to ensure all fields exist
                defaults = _get_default_vision_config()
                defaults.update(data)
                return defaults
    except Exception as e:
        logger.error(f"Failed to load vision config: {e}")
    return _get_default_vision_config()


def save_vision_config(config: VisionConfig) -> bool:
    """Save vision configuration to file."""
    try:
        VISION_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(VISION_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save vision config: {e}")
        return False


def is_vision_enabled() -> bool:
    """Check if vision feature is globally enabled."""
    config = load_vision_config()
    return config.get("enabled", False)


def model_supports_vision_capability(model_name: str) -> bool:
    """Check if model has vision capability in DEFAULT_MODELS."""
    for provider_models in DEFAULT_MODELS.values():
        for model in provider_models:
            if model["id"] == model_name:
                return model.get("support_vision", False)
    return False


def model_in_vision_supported_list(model_name: str, config: Optional[VisionConfig] = None) -> bool:
    """Check if model is in the configured supported models list."""
    if config is None:
        config = load_vision_config()
    supported_models = config.get("supportedModels", [])
    return any(m.get("id") == model_name for m in supported_models)


def should_use_vision_directly(model_name: str, config: Optional[VisionConfig] = None) -> bool:
    """
    Determine if vision should be used directly for this model.
    Returns True if:
    - Vision is globally enabled
    - Model is in the supported models list
    """
    if config is None:
        config = load_vision_config()
    if not config.get("enabled", False):
        return False
    return model_in_vision_supported_list(model_name, config)


def collect_vision_capable_models() -> List[SupportedModel]:
    """Collect all models with vision capability from DEFAULT_MODELS."""
    result = []
    for provider_models in DEFAULT_MODELS.values():
        for model in provider_models:
            if model.get("support_vision", False):
                result.append({"id": model["id"], "name": model["name"]})
    return result


def ensure_vision_data_dir() -> Path:
    """Ensure vision data directory exists and return its path."""
    VISION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return VISION_DATA_DIR


def save_image_to_file(image_data: bytes, ext: str = "jpg") -> str:
    """
    Save image bytes to file and return filename.

    Args:
        image_data: Raw image bytes
        ext: File extension (jpg, png, etc.)

    Returns:
        Filename (without path)
    """
    ensure_vision_data_dir()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = VISION_DATA_DIR / filename
    with open(filepath, "wb") as f:
        f.write(image_data)
    return filename


def get_image_url(filename: str) -> str:
    """Get the full URL for an image file."""
    return f"{VISION_PREVIEW_URL_PREFIX}/{filename}"


def decode_base64_image(data_url: str) -> tuple[bytes, str]:
    """
    Decode base64 image from data URL.

    Args:
        data_url: Data URL string (data:image/jpeg;base64,...)

    Returns:
        Tuple of (image_bytes, extension)
    """
    # Handle both data URL and raw base64
    if data_url.startswith("data:"):
        # Extract mime type and base64 data
        header, base64_data = data_url.split(",", 1)
        mime_type = header.split(";")[0].split(":")[1]
        ext = mime_type.split("/")[1]
        if ext == "jpeg":
            ext = "jpg"
    else:
        base64_data = data_url
        ext = "jpg"  # Default to jpg

    image_bytes = base64.b64decode(base64_data)
    return image_bytes, ext


def process_image(
    image_bytes: bytes,
    max_size: int = 2048,
    max_file_size_mb: int = 10,
    quality: int = 80,
) -> tuple[bytes, str]:
    """
    Process image: resize if needed and compress.

    Args:
        image_bytes: Raw image bytes
        max_size: Maximum dimension (width or height)
        max_file_size_mb: Maximum file size in MB
        quality: JPEG compression quality (1-100)

    Returns:
        Tuple of (processed_bytes, extension)
    """
    img = Image.open(BytesIO(image_bytes))

    # Convert RGBA to RGB if needed (for JPEG)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    # Resize if needed
    width, height = img.size
    if width > max_size or height > max_size:
        ratio = min(max_size / width, max_size / height)
        new_size = (int(width * ratio), int(height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Compress
    output = BytesIO()
    img.save(output, format="JPEG", quality=quality, optimize=True)
    result_bytes = output.getvalue()

    # If still too large, reduce quality further
    max_bytes = max_file_size_mb * 1024 * 1024
    while len(result_bytes) > max_bytes and quality > 20:
        quality -= 10
        output = BytesIO()
        img.save(output, format="JPEG", quality=quality, optimize=True)
        result_bytes = output.getvalue()

    return result_bytes, "jpg"


def encode_image_to_base64(image_bytes: bytes, ext: str = "jpg") -> str:
    """Encode image bytes to base64 data URL."""
    mime_type = f"image/{'jpeg' if ext == 'jpg' else ext}"
    base64_data = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{base64_data}"


async def process_vision_content(
    content: Union[str, List[Dict[str, Any]]],
    model_name: str,
    config: Optional[VisionConfig] = None,
) -> Union[str, List[Dict[str, Any]]]:
    """
    Process vision content based on model capability and config.

    If model supports vision directly: process and return multimodal content
    If not: save images to files and replace with URL text

    Args:
        content: Message content (string or multimodal list)
        model_name: Name of the model being used
        config: Optional vision config (loaded if not provided)

    Returns:
        Processed content
    """
    # If content is string, no images to process
    if isinstance(content, str):
        return content

    if config is None:
        config = load_vision_config()

    use_vision = should_use_vision_directly(model_name, config)
    max_size = config.get("maxImageSize", 2048)
    max_file_size = config.get("maxFileSize", 10)
    quality = config.get("compressionQuality", 80)

    processed_content = []
    text_parts = []

    for item in content:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")

        if item_type == "text":
            if use_vision:
                processed_content.append(item)
            else:
                text_parts.append(item.get("text", ""))

        elif item_type == "image_url":
            image_url_data = item.get("image_url", {})
            url = image_url_data.get("url", "")

            if not url:
                continue

            try:
                # Decode base64 image
                image_bytes, ext = decode_base64_image(url)

                # Process image (resize/compress)
                processed_bytes, ext = process_image(
                    image_bytes, max_size, max_file_size, quality
                )

                if use_vision:
                    # Re-encode and keep as multimodal
                    new_url = encode_image_to_base64(processed_bytes, ext)
                    processed_content.append({
                        "type": "image_url",
                        "image_url": {"url": new_url}
                    })
                else:
                    # Save to file and convert to text
                    filename = save_image_to_file(processed_bytes, ext)
                    image_url = get_image_url(filename)
                    text_parts.append(f"[图片: {image_url}]")

            except Exception as e:
                logger.error(f"Failed to process image: {e}")
                text_parts.append("[图片处理失败]")

    if use_vision:
        return processed_content if processed_content else ""
    else:
        # Return combined text
        return "\n".join(text_parts) if text_parts else ""


def cleanup_old_images(days: int = VISION_CLEANUP_DAYS) -> int:
    """
    Remove images older than specified days.

    Args:
        days: Number of days to keep images

    Returns:
        Number of files deleted
    """
    if not VISION_DATA_DIR.exists():
        return 0

    cutoff = datetime.now() - timedelta(days=days)
    deleted = 0

    for filepath in VISION_DATA_DIR.iterdir():
        if filepath.is_file():
            try:
                mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
                if mtime < cutoff:
                    filepath.unlink()
                    deleted += 1
            except Exception as e:
                logger.error(f"Failed to delete {filepath}: {e}")

    if deleted > 0:
        logger.info(f"Vision cleanup: deleted {deleted} old images")

    return deleted
```

**Step 2: Commit**

```bash
git add helper/vision.py
git commit -m "feat(vision): create vision module with config and image processing"
```

---

## Task 3: Create default vision config file

**Files:**
- Create: `config/vision-config.json`

**Step 1: Create the default config file**

```json
{
  "enabled": false,
  "supportedModels": [],
  "maxImageSize": 2048,
  "maxFileSize": 10,
  "compressionQuality": 80
}
```

**Step 2: Create data/vision directory with .gitkeep**

```bash
mkdir -p data/vision
touch data/vision/.gitkeep
```

**Step 3: Update .gitignore if needed**

Add to `.gitignore`:
```
data/vision/*.jpg
data/vision/*.png
data/vision/*.jpeg
data/vision/*.webp
data/vision/*.gif
!data/vision/.gitkeep
```

**Step 4: Commit**

```bash
git add config/vision-config.json data/vision/.gitkeep .gitignore
git commit -m "feat(vision): add default vision config and data directory"
```

---

## Task 4: Add vision API endpoints

**Files:**
- Modify: `main.py`

**Step 1: Add imports**

Add to imports section (around line 44):

```python
from helper.vision import (
    load_vision_config,
    save_vision_config,
    collect_vision_capable_models,
    cleanup_old_images,
    VISION_DATA_DIR,
)
```

**Step 2: Add GET /vision/config endpoint**

Add after MCP config endpoints (after line 904):

```python
@app.get('/vision/config')
async def get_vision_config():
    """获取视觉识别配置"""
    try:
        config = load_vision_config()
        # Also return available models for UI
        config["availableModels"] = collect_vision_capable_models()
        return JSONResponse(content={"code": 200, "data": config})
    except Exception as e:
        logger.error(f"Failed to get vision config: {e}")
        return JSONResponse(content={"code": 500, "error": str(e)})


@app.post('/vision/config')
async def save_vision_config_endpoint(request: Request):
    """保存视觉识别配置"""
    try:
        data = await request.json()
        # Remove availableModels if present (it's computed, not stored)
        data.pop("availableModels", None)
        if save_vision_config(data):
            return JSONResponse(content={"code": 200, "data": {"message": "ok"}})
        else:
            return JSONResponse(content={"code": 500, "error": "Failed to save config"})
    except Exception as e:
        logger.error(f"Failed to save vision config: {e}")
        return JSONResponse(content={"code": 500, "error": str(e)})


@app.get('/vision/preview/{filename}')
async def vision_preview(filename: str):
    """图片预览接口"""
    # Security: only allow specific extensions and no path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        return JSONResponse(content={"code": 400, "error": "Invalid filename"}, status_code=400)

    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        return JSONResponse(content={"code": 400, "error": "Invalid file type"}, status_code=400)

    filepath = VISION_DATA_DIR / filename
    if not filepath.exists():
        return JSONResponse(content={"code": 404, "error": "File not found"}, status_code=404)

    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "application/octet-stream")

    return FileResponse(filepath, media_type=media_type)
```

**Step 3: Commit**

```bash
git add main.py
git commit -m "feat(vision): add vision config and preview API endpoints"
```

---

## Task 5: Add vision cleanup periodic task

**Files:**
- Modify: `helper/lifespan.py`

**Step 1: Add imports**

```python
from helper.vision import cleanup_old_images
from helper.config import VISION_CLEANUP_INTERVAL
```

**Step 2: Add cleanup task function**

Add after `periodic_mcp_check` function:

```python
async def periodic_vision_cleanup(interval: int = VISION_CLEANUP_INTERVAL) -> None:
    """Periodically cleanup old vision images."""
    # Run once at startup
    cleanup_old_images()
    # Then run periodically
    while True:
        await asyncio.sleep(interval)
        cleanup_old_images()
```

**Step 3: Update lifespan_context to start cleanup task**

Update the `lifespan_context` function:

```python
@asynccontextmanager
async def lifespan_context(app: FastAPI):
    """FastAPI 生命周期钩子，负责启动/停止 Redis 和周期任务。"""
    mcp_task = None
    vision_task = None
    try:
        mcp_task = asyncio.create_task(periodic_mcp_check(app))
        vision_task = asyncio.create_task(periodic_vision_cleanup())
        redis_manager = RedisManager()
        app.state.redis_manager = redis_manager
        logger.info("✅ 初始化成功")
    except Exception as exc:
        logger.info(f"❌ 初始化失败: {str(exc)}")
    try:
        yield
    finally:
        for task in [mcp_task, vision_task]:
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("✅ 定时任务已停止")
        logger.info("🛑 AI服务正在关闭...")
```

**Step 4: Commit**

```bash
git add helper/lifespan.py
git commit -m "feat(vision): add periodic cleanup task for old images"
```

---

## Task 6: Integrate vision processing into /chat endpoint

**Files:**
- Modify: `main.py`

**Step 1: Add vision import**

Add to imports:

```python
from helper.vision import process_vision_content, load_vision_config
```

**Step 2: Update /chat endpoint to handle multimodal text**

In the `/chat` endpoint, after `text = process_html_content(text)` (around line 192), add vision processing:

```python
    # 处理HTML内容（图片标签）
    text = process_html_content(text)

    # 处理多模态内容（图片）
    vision_config = load_vision_config()
    if isinstance(text, list):
        # Multimodal content - process images
        text = await process_vision_content(text, model_name, vision_config)
```

**Step 3: Update stream_generate to handle multimodal messages**

In the `stream_generate` function, update the message creation (around line 349):

```python
            # 添加用户的新消息
            user_content = data["text"]
            if isinstance(user_content, list):
                # Multimodal content
                end_context = [HumanMessage(content=user_content)]
            else:
                end_context = [HumanMessage(content=user_content)]
```

**Step 4: Commit**

```bash
git add main.py
git commit -m "feat(vision): integrate vision processing into /chat endpoint"
```

---

## Task 7: Update invoke.py for multimodal support

**Files:**
- Modify: `helper/invoke.py`

**Step 1: Update _normalize_message to handle multimodal content**

Replace the `_normalize_message` function:

```python
def _normalize_message(item: Any) -> Optional[BaseMessage]:
    role = None
    content = None
    if isinstance(item, dict):
        role = item.get("role") or item.get("type") or "human"
        content = item.get("content") or item.get("text")
    elif isinstance(item, (list, tuple)) and len(item) >= 2:
        role, content = item[0], item[1]
    elif isinstance(item, str):
        role, content = "human", item
    elif item is not None:
        role, content = "human", str(item)

    # 无内容则返回 None
    if content is None:
        return None

    # 如果 content 是空字符串，也返回 None
    if isinstance(content, str) and not content:
        return None

    # 角色标准化
    normalized_role = _normalize_role(role)

    # 处理 content - 支持字符串或多模态列表
    # 多模态列表格式: [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {...}}]
    final_content = content
    if isinstance(content, list):
        # Keep list content as-is for multimodal support
        final_content = content
    elif not isinstance(content, str):
        final_content = str(content)

    # 构造 LangChain 消息对象
    if normalized_role == "human":
        return HumanMessage(content=final_content)
    elif normalized_role == "assistant":
        return AIMessage(content=final_content if isinstance(final_content, str) else str(final_content))
    elif normalized_role == "system":
        return SystemMessage(content=final_content if isinstance(final_content, str) else str(final_content))
    else:
        # 默认安全回退
        return HumanMessage(content=final_content)
```

**Step 2: Commit**

```bash
git add helper/invoke.py
git commit -m "feat(vision): update invoke.py to support multimodal content"
```

---

## Task 8: Integrate vision into /invoke/stream endpoint

**Files:**
- Modify: `main.py`

**Step 1: Update /invoke/stream to process vision content**

In the `/invoke/stream` endpoint, after getting `parsed_context` (around line 673), add vision processing:

```python
    stored_context = data.get("final_context") or []
    parsed_context = parse_context(stored_context)
    if not parsed_context:
        async def no_context_stream():
            yield f"id: {stream_key}\nevent: done\ndata: {json_error('No context found')}\n\n"
        return StreamingResponse(
            no_context_stream(),
            media_type='text/event-stream'
        )

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

**Step 2: Commit**

```bash
git add main.py
git commit -m "feat(vision): integrate vision processing into /invoke/stream endpoint"
```

---

## Task 9: Create frontend TypeScript types

**Files:**
- Create: `ui/src/data/vision-config.ts`

**Step 1: Create the types file**

```typescript
// ui/src/data/vision-config.ts
export interface SupportedModel {
  id: string
  name: string
}

export interface VisionConfig {
  enabled: boolean
  supportedModels: SupportedModel[]
  maxImageSize: number
  maxFileSize: number
  compressionQuality: number
  availableModels?: SupportedModel[] // Computed, returned by API
}

export const DEFAULT_VISION_CONFIG: VisionConfig = {
  enabled: false,
  supportedModels: [],
  maxImageSize: 2048,
  maxFileSize: 10,
  compressionQuality: 80,
}
```

**Step 2: Commit**

```bash
git add ui/src/data/vision-config.ts
git commit -m "feat(vision): add frontend TypeScript types for vision config"
```

---

## Task 10: Create frontend storage functions

**Files:**
- Create: `ui/src/lib/vision-storage.ts`

**Step 1: Create the storage module**

```typescript
// ui/src/lib/vision-storage.ts
import { type VisionConfig, DEFAULT_VISION_CONFIG } from "@/data/vision-config"

/**
 * Load vision configuration from API
 */
export const loadVisionConfig = async (): Promise<VisionConfig> => {
  try {
    const response = await fetch(`/ai/vision/config`)
    if (!response.ok) {
      if (response.status === 404) {
        return DEFAULT_VISION_CONFIG
      }
      throw new Error(`Failed to load vision config: ${response.statusText}`)
    }
    const result = await response.json()
    if (result.code === 200 && result.data) {
      return {
        ...DEFAULT_VISION_CONFIG,
        ...result.data,
      }
    }
    return DEFAULT_VISION_CONFIG
  } catch (error) {
    console.error("Error loading vision config:", error)
    return DEFAULT_VISION_CONFIG
  }
}

/**
 * Save vision configuration to API
 */
export const saveVisionConfig = async (config: VisionConfig): Promise<boolean> => {
  try {
    // Remove availableModels as it's computed
    const { availableModels, ...saveData } = config
    const response = await fetch(`/ai/vision/config`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(saveData),
    })
    if (!response.ok) {
      throw new Error(`Failed to save vision config: ${response.statusText}`)
    }
    const result = await response.json()
    return result.code === 200
  } catch (error) {
    console.error("Error saving vision config:", error)
    throw error
  }
}
```

**Step 2: Commit**

```bash
git add ui/src/lib/vision-storage.ts
git commit -m "feat(vision): add frontend storage functions for vision config"
```

---

## Task 11: Add vision i18n translations

**Files:**
- Modify: `ui/src/lib/i18n-core.ts`

**Step 1: Add vision translations to zh section**

Add after the `mcp` section in the `zh` translations (after line 92):

```typescript
    vision: {
      title: "视觉识别",
      description: "配置视觉识别功能以支持图片理解",
      enabled: "启用视觉识别",
      supportedModels: "支持的模型",
      supportedModelsTip: "选择可以接收图片的AI模型",
      maxImageSize: "最大图片尺寸",
      maxImageSizeTip: "像素，宽高中较大值",
      maxFileSize: "最大文件大小",
      maxFileSizeTip: "MB",
      compressionQuality: "压缩质量",
      compressionQualityTip: "1-100，值越大质量越高",
      statusEnabled: "已启用",
      statusDisabled: "已禁用",
      editTitle: "视觉识别配置",
      edit: "编辑",
      cancel: "取消",
      save: "保存",
      imageLimit: "图片限制",
    },
```

**Step 2: Add vision translations to en section**

Add after the `mcp` section in the `en` translations (after line 173):

```typescript
    vision: {
      title: "Vision Recognition",
      description: "Configure vision recognition to support image understanding",
      enabled: "Enable Vision",
      supportedModels: "Supported Models",
      supportedModelsTip: "Select AI models that can receive images",
      maxImageSize: "Max Image Size",
      maxImageSizeTip: "pixels, max of width/height",
      maxFileSize: "Max File Size",
      maxFileSizeTip: "MB",
      compressionQuality: "Compression Quality",
      compressionQualityTip: "1-100, higher means better quality",
      statusEnabled: "Enabled",
      statusDisabled: "Disabled",
      editTitle: "Vision Configuration",
      edit: "Edit",
      cancel: "Cancel",
      save: "Save",
      imageLimit: "Image Limits",
    },
```

**Step 3: Commit**

```bash
git add ui/src/lib/i18n-core.ts
git commit -m "feat(vision): add i18n translations for vision config"
```

---

## Task 12: Update aibots.ts with support_vision

**Files:**
- Modify: `ui/src/data/aibots.ts`

**Step 1: Update AIBotModelOption interface**

Update the interface (around line 25-29):

```typescript
export interface AIBotModelOption {
  value: string
  label: string
  support_mcp: boolean
  support_vision: boolean
}
```

**Step 2: Commit**

```bash
git add ui/src/data/aibots.ts
git commit -m "feat(vision): add support_vision to AIBotModelOption interface"
```

---

## Task 13: Create VisionConfigCard component

**Files:**
- Create: `ui/src/components/aibot/VisionConfigCard.tsx`

**Step 1: Create the card component**

```tsx
// ui/src/components/aibot/VisionConfigCard.tsx
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import type { VisionConfig } from "@/data/vision-config"
import { Eye, Settings } from "lucide-react"

interface VisionConfigCardProps {
  config: VisionConfig
  onEdit: () => void
  t: (key: string) => string
}

export function VisionConfigCard({ config, onEdit, t }: VisionConfigCardProps) {
  const modelCount = config.supportedModels.length
  const displayModels = config.supportedModels.slice(0, 3)
  const extraCount = modelCount - 3

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Eye className="h-5 w-5" />
            <CardTitle className="text-base">{t("vision.title")}</CardTitle>
          </div>
          <Button variant="ghost" size="sm" onClick={onEdit}>
            <Settings className="h-4 w-4 mr-1" />
            {t("vision.edit")}
          </Button>
        </div>
        <CardDescription>{t("vision.description")}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">{t("vision.enabled")}:</span>
          <Badge variant={config.enabled ? "default" : "secondary"}>
            {config.enabled ? t("vision.statusEnabled") : t("vision.statusDisabled")}
          </Badge>
        </div>
        {modelCount > 0 && (
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm text-muted-foreground">{t("vision.supportedModels")}:</span>
            {displayModels.map((model) => (
              <Badge key={model.id} variant="outline" className="text-xs">
                {model.name}
              </Badge>
            ))}
            {extraCount > 0 && (
              <Badge variant="outline" className="text-xs">
                +{extraCount}
              </Badge>
            )}
          </div>
        )}
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <span>{t("vision.imageLimit")}:</span>
          <span>
            {config.maxImageSize}px / {config.maxFileSize}MB / {config.compressionQuality}%
          </span>
        </div>
      </CardContent>
    </Card>
  )
}
```

**Step 2: Commit**

```bash
git add ui/src/components/aibot/VisionConfigCard.tsx
git commit -m "feat(vision): create VisionConfigCard component"
```

---

## Task 14: Create VisionEditorSheet component

**Files:**
- Create: `ui/src/components/aibot/VisionEditorSheet.tsx`

**Step 1: Create the editor component**

```tsx
// ui/src/components/aibot/VisionEditorSheet.tsx
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { Switch } from "@/components/ui/switch"
import type { SupportedModel, VisionConfig } from "@/data/vision-config"
import type { AIBotItem } from "@/data/aibots"
import { useEffect, useState } from "react"

interface VisionEditorSheetProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  config: VisionConfig
  onSave: (config: VisionConfig) => Promise<void>
  aiBots: AIBotItem[]
  t: (key: string) => string
}

export function VisionEditorSheet({
  open,
  onOpenChange,
  config,
  onSave,
  aiBots,
  t,
}: VisionEditorSheetProps) {
  const [editConfig, setEditConfig] = useState<VisionConfig>(config)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (open) {
      setEditConfig(config)
    }
  }, [open, config])

  const handleSave = async () => {
    setSaving(true)
    try {
      await onSave(editConfig)
      onOpenChange(false)
    } finally {
      setSaving(false)
    }
  }

  const isModelSelected = (modelId: string) => {
    return editConfig.supportedModels.some((m) => m.id === modelId)
  }

  const toggleModel = (model: { value: string; label: string; support_vision: boolean }) => {
    if (!model.support_vision) return

    const isSelected = isModelSelected(model.value)
    if (isSelected) {
      setEditConfig({
        ...editConfig,
        supportedModels: editConfig.supportedModels.filter((m) => m.id !== model.value),
      })
    } else {
      setEditConfig({
        ...editConfig,
        supportedModels: [
          ...editConfig.supportedModels,
          { id: model.value, name: model.label },
        ],
      })
    }
  }

  // Group models by provider, only show those with support_vision capability
  const modelsByProvider = aiBots
    .map((bot) => ({
      provider: bot.value,
      label: bot.label,
      models: (bot.models || []).filter((m) => m.support_vision),
    }))
    .filter((group) => group.models.length > 0)

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="sm:max-w-lg overflow-y-auto">
        <SheetHeader>
          <SheetTitle>{t("vision.editTitle")}</SheetTitle>
          <SheetDescription>{t("vision.description")}</SheetDescription>
        </SheetHeader>

        <div className="space-y-6 py-4">
          {/* Enable Switch */}
          <div className="flex items-center justify-between">
            <Label htmlFor="vision-enabled">{t("vision.enabled")}</Label>
            <Switch
              id="vision-enabled"
              checked={editConfig.enabled}
              onCheckedChange={(checked) =>
                setEditConfig({ ...editConfig, enabled: checked })
              }
            />
          </div>

          {/* Supported Models */}
          <div className="space-y-3">
            <Label>{t("vision.supportedModels")}</Label>
            <p className="text-sm text-muted-foreground">{t("vision.supportedModelsTip")}</p>
            <div className="space-y-4 max-h-64 overflow-y-auto border rounded-md p-3">
              {modelsByProvider.map((group) => (
                <div key={group.provider} className="space-y-2">
                  <div className="text-sm font-medium text-muted-foreground">
                    {group.label}
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    {group.models.map((model) => (
                      <div key={model.value} className="flex items-center space-x-2">
                        <Checkbox
                          id={`model-${model.value}`}
                          checked={isModelSelected(model.value)}
                          onCheckedChange={() => toggleModel(model)}
                        />
                        <label
                          htmlFor={`model-${model.value}`}
                          className="text-sm cursor-pointer"
                        >
                          {model.label}
                        </label>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Max Image Size */}
          <div className="space-y-2">
            <Label htmlFor="max-image-size">
              {t("vision.maxImageSize")} ({t("vision.maxImageSizeTip")})
            </Label>
            <Input
              id="max-image-size"
              type="number"
              min={256}
              max={8192}
              value={editConfig.maxImageSize}
              onChange={(e) =>
                setEditConfig({
                  ...editConfig,
                  maxImageSize: parseInt(e.target.value) || 2048,
                })
              }
            />
          </div>

          {/* Max File Size */}
          <div className="space-y-2">
            <Label htmlFor="max-file-size">
              {t("vision.maxFileSize")} ({t("vision.maxFileSizeTip")})
            </Label>
            <Input
              id="max-file-size"
              type="number"
              min={1}
              max={50}
              value={editConfig.maxFileSize}
              onChange={(e) =>
                setEditConfig({
                  ...editConfig,
                  maxFileSize: parseInt(e.target.value) || 10,
                })
              }
            />
          </div>

          {/* Compression Quality */}
          <div className="space-y-2">
            <Label htmlFor="compression-quality">
              {t("vision.compressionQuality")} ({t("vision.compressionQualityTip")})
            </Label>
            <Input
              id="compression-quality"
              type="number"
              min={1}
              max={100}
              value={editConfig.compressionQuality}
              onChange={(e) =>
                setEditConfig({
                  ...editConfig,
                  compressionQuality: parseInt(e.target.value) || 80,
                })
              }
            />
          </div>
        </div>

        <SheetFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t("vision.cancel")}
          </Button>
          <Button onClick={handleSave} disabled={saving}>
            {t("vision.save")}
          </Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  )
}
```

**Step 2: Commit**

```bash
git add ui/src/components/aibot/VisionEditorSheet.tsx
git commit -m "feat(vision): create VisionEditorSheet component"
```

---

## Task 15: Integrate vision components into App.tsx

**Files:**
- Modify: `ui/src/App.tsx`

**Step 1: Add imports**

Add to imports:

```typescript
import { VisionConfigCard } from "@/components/aibot/VisionConfigCard"
import { VisionEditorSheet } from "@/components/aibot/VisionEditorSheet"
import { loadVisionConfig, saveVisionConfig } from "@/lib/vision-storage"
import { type VisionConfig, DEFAULT_VISION_CONFIG } from "@/data/vision-config"
```

**Step 2: Add vision state**

Add state variables (near other state declarations):

```typescript
const [visionConfig, setVisionConfig] = useState<VisionConfig>(DEFAULT_VISION_CONFIG)
const [visionEditorOpen, setVisionEditorOpen] = useState(false)
```

**Step 3: Add vision config loading**

Add to the useEffect that loads configs (or create new one):

```typescript
useEffect(() => {
  const loadConfigs = async () => {
    const config = await loadVisionConfig()
    setVisionConfig(config)
  }
  loadConfigs()
}, [])
```

**Step 4: Add vision handlers**

Add handler functions:

```typescript
const handleEditVision = () => {
  if (!isAdmin) {
    messageError(t("errors.adminOnly"))
    return
  }
  setVisionEditorOpen(true)
}

const handleSaveVision = async (config: VisionConfig) => {
  await saveVisionConfig(config)
  setVisionConfig(config)
  messageSuccess(t("success.save"))
}
```

**Step 5: Add VisionConfigCard to render**

Add after MCPListCard in the settings section:

```tsx
<VisionConfigCard
  config={visionConfig}
  onEdit={handleEditVision}
  t={t}
/>
```

**Step 6: Add VisionEditorSheet**

Add near other sheets/dialogs:

```tsx
<VisionEditorSheet
  open={visionEditorOpen}
  onOpenChange={setVisionEditorOpen}
  config={visionConfig}
  onSave={handleSaveVision}
  aiBots={aiBots}
  t={t}
/>
```

**Step 7: Commit**

```bash
git add ui/src/App.tsx
git commit -m "feat(vision): integrate vision components into App"
```

---

## Task 16: Final verification and cleanup

**Step 1: Install Pillow dependency**

```bash
pip install Pillow
```

Update requirements.txt if it exists:
```bash
echo "Pillow>=10.0.0" >> requirements.txt
```

**Step 2: Build frontend**

```bash
cd ui && npm run build && cd ..
```

**Step 3: Test the API endpoints**

```bash
# Start the server
python main.py &

# Test vision config endpoints
curl http://localhost:5001/vision/config
curl -X POST http://localhost:5001/vision/config \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "supportedModels": [], "maxImageSize": 2048, "maxFileSize": 10, "compressionQuality": 80}'
```

**Step 4: Final commit**

```bash
git add requirements.txt
git commit -m "feat(vision): add Pillow dependency for image processing"
```

**Step 5: Create summary commit (optional)**

```bash
git log --oneline -15
```

---

## Summary

This implementation adds:

1. **Backend:**
   - `support_vision` field in DEFAULT_MODELS
   - `helper/vision.py` module for config and image processing
   - Vision API endpoints (`/vision/config`, `/vision/preview/{filename}`)
   - Periodic cleanup task for old images
   - Integration into `/chat` and `/invoke/stream` endpoints

2. **Frontend:**
   - TypeScript types for vision config
   - Storage functions for API communication
   - VisionConfigCard component
   - VisionEditorSheet component
   - i18n translations

3. **Key behaviors:**
   - If vision enabled + model in supported list → use multimodal directly
   - Otherwise → save image to file, send URL text for MCP OCR
   - Images stored in `data/vision/`, auto-cleaned after 7 days
