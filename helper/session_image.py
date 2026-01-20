"""
Session Image Processing Module

Handles caching of session images and replacing them with placeholders
to reduce token usage while allowing AI to retrieve them on demand.
"""

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("ai")

# Cache TTL: 2 hours
SESSION_IMAGE_TTL = 7200

# Placeholder pattern
PLACEHOLDER_PATTERN = re.compile(r"\[picture:session_([a-f0-9]{32})\]")


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
        cache_key = f"session_image_{md5_hash}"
        cache_value = json.dumps({"data": base64_data, "mime_type": mime_type})

        try:
            await redis_manager.set_cache(cache_key, cache_value, ex=SESSION_IMAGE_TTL)
        except Exception as e:
            logger.warning(f"Failed to cache session image: {e}")
            # Fallback: keep original image
            new_content.append(item)
            continue

        # Replace with placeholder
        new_content.append({
            "type": "text",
            "text": f"[picture:session_{md5_hash}]"
        })

    return new_content


async def process_session_images(
    messages: List[Any],
    redis_manager: Any,
) -> List[Any]:
    """Process messages to replace session images with placeholders.

    The last human message keeps its images intact. All other human messages
    have their images replaced with placeholders.

    Args:
        messages: List of messages (dict or tuple format)
        redis_manager: Redis manager instance for caching

    Returns:
        Processed messages with session images replaced
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
