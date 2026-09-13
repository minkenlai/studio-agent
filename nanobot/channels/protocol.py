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
