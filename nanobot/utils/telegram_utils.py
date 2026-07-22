"""
Utility functions for persisting Telegram topic profile mappings.

Schema: {chat_id: {topic_id: {"profile": str|null, "name": str|null, "native_audio": bool|null}}}
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

PINS_FILE = Path("sessions/tg_pins.json")

# Type alias: {chat_id_str: {topic_id_int: {"profile": str|None, "name": str|None, "native_audio": bool|None}}}
TopicPinEntry = Dict[str, Any]


def load_topic_pins() -> Dict[str, Dict[int, TopicPinEntry]]:
    """Load the topic profile mappings from the JSON file.

    Returns:
        Nested dict {chat_id: {topic_id: {"profile": ..., "name": ..., "native_audio": ...}}}
    """
    if not PINS_FILE.exists():
        return {}

    try:
        with open(PINS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert keys to correct types, and migrate legacy flat format if needed
        result: Dict[str, Dict[int, TopicPinEntry]] = {}
        for chat_id_str, topics in data.items():
            chat_id = str(chat_id_str)
            result[chat_id] = {}
            for topic_id_str, value in topics.items():
                topic_id = int(topic_id_str)
                if isinstance(value, dict):
                    # New format: {"profile": ..., "name": ..., "native_audio": ...}
                    result[chat_id][topic_id] = {
                        "profile": value.get("profile"),
                        "name": value.get("name"),
                        "native_audio": value.get("native_audio"),
                    }
                else:
                    # Legacy flat format: just the profile string (or null)
                    result[chat_id][topic_id] = {
                        "profile": value,
                        "name": None,
                        "native_audio": None,
                    }

        return result
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to load topic pins: {e}")
        return {}


def save_topic_pins(pins: Dict[str, Dict[int, TopicPinEntry]]) -> None:
    """Save the topic profile mappings to the JSON file."""
    serializable: Dict[str, Any] = {}
    for chat_id, topics in pins.items():
        serializable[chat_id] = {}
        for topic_id, entry in topics.items():
            serializable[chat_id][str(topic_id)] = entry

    PINS_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(PINS_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"Failed to save topic pins: {e}")


def get_cached_profile(
    pins: Dict[str, Dict[int, TopicPinEntry]], chat_id: str, topic_id: int
) -> Optional[str]:
    """Get the cached profile for a chat/topic from the in-memory pins dict."""
    entry = pins.get(chat_id, {}).get(topic_id)
    return entry.get("profile") if isinstance(entry, dict) else entry


def get_cached_topic_name(
    pins: Dict[str, Dict[int, TopicPinEntry]], chat_id: str, topic_id: int
) -> Optional[str]:
    """Get the cached topic name for a chat/topic from the in-memory pins dict."""
    entry = pins.get(chat_id, {}).get(topic_id)
    return entry.get("name") if isinstance(entry, dict) else None


def set_cached_profile(
    pins: Dict[str, Dict[int, TopicPinEntry]],
    chat_id: str,
    topic_id: int,
    profile: Optional[str],
    topic_name: Optional[str] = None,
    native_audio: Optional[bool] = None,
) -> None:
    """Set the cached profile for a chat/topic in the in-memory pins dict."""
    if chat_id not in pins:
        pins[chat_id] = {}
    if topic_id not in pins[chat_id]:
        pins[chat_id][topic_id] = {}
    pins[chat_id][topic_id]["profile"] = profile
    if topic_name is not None:
        pins[chat_id][topic_id]["name"] = topic_name
    elif "name" not in pins[chat_id][topic_id]:
        pins[chat_id][topic_id]["name"] = None
    if native_audio is not None:
        pins[chat_id][topic_id]["native_audio"] = native_audio
    elif "native_audio" not in pins[chat_id][topic_id]:
        pins[chat_id][topic_id]["native_audio"] = None


def set_cached_topic_name(
    pins: Dict[str, Dict[int, TopicPinEntry]],
    chat_id: str,
    topic_id: int,
    topic_name: Optional[str],
) -> None:
    """Set or update the cached topic name."""
    if chat_id not in pins:
        pins[chat_id] = {}
    if topic_id not in pins[chat_id]:
        pins[chat_id][topic_id] = {}
    pins[chat_id][topic_id]["name"] = topic_name


def remove_cached_profile(
    pins: Dict[str, Dict[int, TopicPinEntry]], chat_id: str, topic_id: int
) -> None:
    """Remove the cached profile for a chat/topic from the in-memory pins dict."""
    if chat_id in pins and topic_id in pins[chat_id]:
        del pins[chat_id][topic_id]
        if not pins[chat_id]:
            del pins[chat_id]


def add_lazy_topic_entry(
    pins: Dict[str, Dict[int, TopicPinEntry]], chat_id: str, topic_id: int
) -> None:
    """Add a lazy topic entry to the pins dict if it doesn't exist yet."""
    if chat_id not in pins:
        pins[chat_id] = {}
    if topic_id not in pins[chat_id]:
        pins[chat_id][topic_id] = {"profile": None, "name": None, "native_audio": None}
