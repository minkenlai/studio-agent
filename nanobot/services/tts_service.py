from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Optional

from nanobot.config.paths import get_media_dir


class TTSService:
    """Service to handle text-to-speech synthesis logic."""

    def __init__(self, loop):
        self.loop = loop

    async def speak(
        self,
        text: Optional[str],
        output_path: Optional[str] = None,
        session_key: Optional[str] = None,
        msg_address: Optional[any] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Synthesize text to speech.

        Args:
            text: Content to speak. If None, fetches the last assistant message from the session.
            output_path: Specific path to save the file. If None, generates a unique name in the media dir.
            session_key: Key to identify the session for history retrieval.
            msg_address: The address of the message for channel-specific media directory resolution.

        Returns:
            A tuple of (success_boolean, file_path_or_error_message).
        """
        # 1. Determine text source
        if text is None:
            if not session_key:
                return False, "No session key provided to fetch context."

            session = self.loop.sessions.get_or_create(session_key)
            last_asst = ""
            for m in reversed(session.messages):
                if m.get("role") == "assistant":
                    last_asst = m.get("content", "")
                    break

            if not last_asst:
                return False, "No prior agent message found to speak."
            text = last_asst

        # 2. Clean text
        clean = self._clean_text(text)
        if not clean:
            return False, "Nothing to speak (empty after cleanup)."

        # 3. Fetch TTS client (needed for output format)
        tts_client = await self.loop.get_tts_client()
        if tts_client is None:
            return False, "🔇 TTS is not configured."

        # 4. Determine output path (use provider's native extension)
        if output_path:
            final_path = Path(output_path)
        else:
            # Fallback to default media dir
            channel = msg_address.channel if msg_address else "default"
            out_dir = get_media_dir(channel)
            out_dir.mkdir(parents=True, exist_ok=True)
            ext = getattr(tts_client, "output_ext", ".wav")
            filename = f"tts_{uuid.uuid4().hex[:12]}{ext}"
            final_path = out_dir / filename

        # 5. Synthesize

        ok = await tts_client.synthesize(clean, str(final_path))
        if not ok:
            return False, "🔇 TTS synthesis failed."

        return True, str(final_path)

    def _clean_text(self, text: str) -> str:
        """Strip markdown and normalize whitespace for cleaner TTS output."""
        # Remove fenced code blocks (``` ... ```) — must come before inline patterns
        clean = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        # Bold (use .+? to handle ***bold-italic*** gracefully)
        clean = re.sub(r"\*\*(.+?)\*\*", r"\1", clean)
        # Inline code
        clean = re.sub(r"`([^`]+)`", r"\1", clean)
        # Italics (*text* or _text_)
        clean = re.sub(r"\*([^*]+)\*", r"\1", clean)
        clean = re.sub(r"_([^_]+)_", r"\1", clean)
        # Strike-throughs (~~text~~)
        clean = re.sub(r"~~([^~]+)~~", r"\1", clean)
        # Links [text](url) → text
        clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)
        # Headings (# text)
        clean = re.sub(r"^#{1,6}\s+", "", clean, flags=re.MULTILINE)
        # List markers (-, *, +, or 1. )
        clean = re.sub(r"^[ \t]*[-*+]\s+", "", clean, flags=re.MULTILINE)
        clean = re.sub(r"^[ \t]*\d+\.\s+", "", clean, flags=re.MULTILINE)
        # Blockquotes (> text)
        clean = re.sub(r"^>\s?", "", clean, flags=re.MULTILINE)
        # Normalize whitespace
        clean = re.sub(r"\n{2,}", "\n", clean)
        return clean.strip()
