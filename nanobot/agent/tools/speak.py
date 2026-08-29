from __future__ import annotations

from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import Address
from nanobot.services.tts_service import TTSService


class SpeakTool(Tool):
    """Tool to synthesize speech from text."""

    def __init__(self, loop):
        self._loop = loop
        self.tts_service = TTSService(loop)
        self._address: Address | None = None
        self._session_key: str | None = None

    def set_context(self, address: Address, session_key: str | None = None) -> None:
        """Set the current message context."""
        self._address = address
        self._session_key = session_key

    @property
    def name(self) -> str:
        return "speak"

    @property
    def description(self) -> str:
        return (
            "Synthesize speech from text. "
            "If no text is provided, it will synthesize the last assistant message from the session. "
            "Provide an 'output_path' to save the audio to a specific location; otherwise, a unique file is created in the default media directory."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": ["string", "null"],
                    "description": "The content to synthesize into speech. If null, the last assistant message is used.",
                },
                "output_path": {
                    "type": ["string", "null"],
                    "description": "The specific file path to save the generated audio (e.g., /path/to/file.wav). If null, a unique file is created.",
                },
            },
            "required": [],
        }

    async def execute(
        self,
        text: str | None = None,
        output_path: str | None = None,
        **kwargs: Any,
    ) -> str:
        session_key = self._session_key
        msg_address = self._address

        ok, result = await self.tts_service.speak(
            text=text,
            output_path=output_path,
            session_key=session_key,
            msg_address=msg_address,
        )

        if not ok:
            return f"Error: {result}"

        return f"Speech synthesized successfully and saved to: {result}"
