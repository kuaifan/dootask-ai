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
