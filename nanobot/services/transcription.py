"""Service for handling audio transcription across the system."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from nanobot.config.schema import Config
from nanobot.providers.transcription.base import BaseTranscriptionProvider
from nanobot.providers.transcription.groq import GroqTranscriptionProvider


class TranscriptionService:
    """
    Orchestrates audio transcription using a configured provider.
    """

    def __init__(self, config: Config):
        self.config = config
        self._provider: BaseTranscriptionProvider | None = None

    def _get_provider(self) -> BaseTranscriptionProvider:
        """
        Resolve the transcription provider based on configuration.
        """
        # Future: Add logic to check config.providers.transcription.backend
        # For now, default to Groq.
        try:
            # Pull API key from the groq provider config
            api_key = self.config.providers.groq.api_key
            return GroqTranscriptionProvider(api_key=api_key)
        except Exception as e:
            logger.error("Failed to resolve transcription provider: {}", e)
            # Fallback to a dummy provider or raise
            raise

    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe an audio file.
        """
        if not self._provider:
            self._provider = self._get_provider()

        return await self._provider.transcribe(file_path)
