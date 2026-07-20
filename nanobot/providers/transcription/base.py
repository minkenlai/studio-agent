"""Base interface for audio transcription providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseTranscriptionProvider(ABC):
    """
    Abstract base class for audio transcription providers.
    """

    @abstractmethod
    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe an audio file to text.

        Args:
            file_path: Path to the audio file.

        Returns:
            Transcribed text or empty string on failure.
        """
        pass
