"""Protocol tokens for channel-level message handling (e.g. Silent Ack, Reactions)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

SilentAckAction = Literal["reaction", "silent", "text"]

# Common markdown/Slack/Discord shortcode to Unicode emoji mapping
EMOJI_SHORTCODES: dict[str, str] = {
    ":white_check_mark:": "✅",
    ":check_mark:": "✅",
    ":heavy_check_mark:": "✔️",
    ":thumbsup:": "👍",
    ":thumbs_up:": "👍",
    ":+1:": "👍",
    ":thumbsdown:": "👎",
    ":thumbs_down:": "👎",
    ":-1:": "👎",
    ":heart:": "❤️",
    ":fire:": "🔥",
    ":eyes:": "👀",
    ":clap:": "👏",
    ":ok_hand:": "👌",
    ":pray:": "🙏",
    ":tada:": "🎉",
    ":rocket:": "🚀",
    ":thinking:": "🤔",
    ":smile:": "😄",
    ":grin:": "😁",
    ":joy:": "😂",
    ":salute:": "🫡",
    ":heart_eyes:": "😍",
    ":star:": "⭐",
    ":100:": "💯",
    ":wave:": "👋",
    ":sparkles:": "✨",
    ":raised_hands:": "🙌",
    ":bulb:": "💡",
    ":x:": "❌",
}

_REACTION_RE = re.compile(r"^\s*\[REACTION:\s*([^\]]+?)\s*\]", re.IGNORECASE)
_SILENT_RE = re.compile(r"^\s*\[(?:SILENT|NONE|NO[_-]?REPLY|NO[_-]?OP)\]", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SilentAckSignal:
    action: SilentAckAction
    emoji: str | None = None
    remaining_text: str = ""


def parse_silent_ack(content: str | None) -> SilentAckSignal:
    """Parse a message for silent ack tokens: [REACTION: <emoji>] or [SILENT].

    If a token is present at the start of the message, any trailing text
    is truncated from group delivery to guarantee zero noise.
    """
    if not content:
        return SilentAckSignal(action="text", remaining_text="")

    text = content.strip()
    # Strip optional surrounding markdown code ticks or emphasis
    clean_text = text.strip("`*")

    # 1. Check for [SILENT] / [NONE] / [NO_REPLY] / [NOOP] (and variants like [NO-REPLY], [NO-OP])
    if _SILENT_RE.match(clean_text):
        return SilentAckSignal(action="silent", emoji=None, remaining_text="")

    # 2. Check for [REACTION: <emoji>]
    if match := _REACTION_RE.match(clean_text):
        raw_emoji = match.group(1).strip()
        lower_emoji = raw_emoji.lower()

        # Handle explicit clearing tokens (e.g. [REACTION: none], [REACTION: silent])
        if lower_emoji in ("none", "null", "clear", "cancel", "remove", "off", "no", "silent"):
            return SilentAckSignal(action="silent", emoji=None, remaining_text="")

        coloned = lower_emoji if lower_emoji.startswith(":") and lower_emoji.endswith(":") else f":{lower_emoji}:"
        normalized = EMOJI_SHORTCODES.get(lower_emoji, EMOJI_SHORTCODES.get(coloned, raw_emoji))
        return SilentAckSignal(action="reaction", emoji=normalized, remaining_text="")

    return SilentAckSignal(action="text", remaining_text=content)


def could_be_silent_ack_prefix(text: str) -> bool:
    """Check if accumulated text could potentially be the start of a silent ack token."""
    s = text.lstrip()
    if not s:
        return True
    s = s.lstrip("`*")
    if not s:
        return True
    return s.startswith("[")


class SilentAckStreamGate:
    """Gates and buffers the initial chunks of a stream to intercept silent ack tokens.

    When an LLM response streams token-by-token in group chats:
    - If the response starts with normal text, deltas are immediately forwarded without buffering.
    - If the response starts with '[', deltas are buffered until ']' is encountered or the buffer
      exceeds `max_buffer_len`.
    - If a valid silent token ([REACTION: ...] or [SILENT]) is detected, the stream is marked as
      suppressed, and all deltas for that stream are dropped.
    - When the turn finishes, `check_and_consume_chat_suppressed` signals to ChannelManager that
      the final turn message should be delivered via `channel.send(msg)` rather than dropped,
      allowing the channel's silent ack / reaction handler to execute cleanly.
    """

    def __init__(self, max_buffer_len: int = 60, ttl_seconds: float = 120.0) -> None:
        self._max_buffer_len = max_buffer_len
        self._ttl_seconds = ttl_seconds
        # Key: (channel, chat_id, stream_id) -> list of accumulated delta chunks
        self._buffers: dict[tuple[str, str, str | None], list[str]] = {}
        # Streams that were confirmed to be silent ack and must be suppressed
        self._suppressed: set[tuple[str, str, str | None]] = set()
        # Streams that were confirmed NOT to be silent ack and flow freely
        self._released: set[tuple[str, str, str | None]] = set()
        # Most recently suppressed stream per (channel, chat_id) with timestamp
        self._chat_suppressed: dict[tuple[str, str], float] = {}

    def process_delta(
        self,
        channel_name: str,
        chat_id: str,
        stream_id: str | None,
        delta: str,
    ) -> tuple[bool, list[str]]:
        """Process a stream delta.

        Returns:
            tuple of (is_suppressed, deltas_to_flush)
            - is_suppressed: True if this stream is suppressed (no channel.send_delta calls).
            - deltas_to_flush: List of buffered delta strings to flush to channel.send_delta now.
        """
        import time

        key = (channel_name, chat_id, stream_id)
        chat_key = (channel_name, chat_id)

        if key in self._suppressed:
            return True, []

        if key in self._released:
            return False, [delta]

        chunks = self._buffers.setdefault(key, [])
        chunks.append(delta)
        accumulated = "".join(chunks)

        if not could_be_silent_ack_prefix(accumulated):
            # Definitely not a silent ack token (starts with normal text/numbers/punctuation)
            self._released.add(key)
            flushed = self._buffers.pop(key, [])
            self._chat_suppressed.pop(chat_key, None)
            return False, flushed

        # Starts with [ (or `* [)
        if "]" in accumulated:
            # Token bracket closed! Parse it.
            sig = parse_silent_ack(accumulated)
            if sig.action != "text":
                # Valid silent ack token!
                self._suppressed.add(key)
                self._buffers.pop(key, None)
                self._chat_suppressed[chat_key] = time.time()
                return True, []
            else:
                # Closed bracket but not a silent ack (e.g. [1], [citation], [link])
                self._released.add(key)
                flushed = self._buffers.pop(key, [])
                self._chat_suppressed.pop(chat_key, None)
                return False, flushed

        if len(accumulated) >= self._max_buffer_len:
            # Exceeded maximum token length without closing ']'
            self._released.add(key)
            flushed = self._buffers.pop(key, [])
            self._chat_suppressed.pop(chat_key, None)
            return False, flushed

        # Still buffering and waiting for closing ']'
        return False, []

    def process_end(
        self,
        channel_name: str,
        chat_id: str,
        stream_id: str | None,
    ) -> tuple[bool, list[str]]:
        """Process a stream end event.

        Returns:
            tuple of (is_suppressed, final_deltas_to_flush)
        """
        import time

        key = (channel_name, chat_id, stream_id)
        chat_key = (channel_name, chat_id)

        if key in self._suppressed:
            self._suppressed.discard(key)
            self._buffers.pop(key, None)
            self._released.discard(key)
            return True, []

        self._released.discard(key)
        chunks = self._buffers.pop(key, None)
        if chunks:
            accumulated = "".join(chunks)
            sig = parse_silent_ack(accumulated)
            if sig.action != "text":
                self._chat_suppressed[chat_key] = time.time()
                return True, []
            return False, chunks

        return False, []

    def check_and_consume_chat_suppressed(self, channel_name: str, chat_id: str) -> bool:
        """Check and consume whether the last stream for this chat was suppressed."""
        import time

        chat_key = (channel_name, chat_id)
        now = time.time()
        # Clean up any expired entries across chats
        expired = [k for k, ts in self._chat_suppressed.items() if (now - ts) > self._ttl_seconds]
        for k in expired:
            self._chat_suppressed.pop(k, None)

        timestamp = self._chat_suppressed.pop(chat_key, None)
        if timestamp is not None and (now - timestamp) <= self._ttl_seconds:
            return True
        return False
