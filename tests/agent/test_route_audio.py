"""Tests for AgentLoop._route_audio audio routing logic."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.providers.base import ModelCapabilities


def _make_mock_caps(audio: bool = True) -> ModelCapabilities:
    return ModelCapabilities(audio=audio)


def _make_audio_block(data: str = "test-audio-data", mime: str = "audio/ogg") -> dict:
    return {
        "type": "audio",
        "data": f"data:{mime};base64,{base64.b64encode(data.encode()).decode()}",
        "mime_type": mime,
    }


def _make_text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def _make_mock_provider(audio_support: bool = True) -> MagicMock:
    """Create a mock provider with a `supports_native_audio` property."""
    provider = MagicMock()
    provider.supports_native_audio = audio_support
    return provider


def _make_message_with_audio(text: str = "Hello", num_audio: int = 1) -> dict:
    content = [_make_text_block(text)]
    for _ in range(num_audio):
        content.append(_make_audio_block())
    return {"role": "user", "content": content}


class TestRouteAudioNativePassthrough:
    """When model supports native audio, messages pass through unchanged."""

    @pytest.mark.asyncio
    @patch("nanobot.providers.registry.get_capabilities", return_value=_make_mock_caps(audio=True))
    async def test_native_audio_model_passes_through_unchanged(self, mock_caps):
        from nanobot.agent.loop import AgentLoop

        loop = MagicMock(spec=AgentLoop)
        loop._get_transcription_provider = AsyncMock()  # Should never be called
        provider = _make_mock_provider(audio_support=True)

        messages = [_make_message_with_audio()]
        result = await AgentLoop._route_audio(
            loop, messages, "claude-sonnet-4-20250514", provider=provider
        )

        assert len(result) == 1
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][1]["type"] == "audio"
        loop._get_transcription_provider.assert_not_called()

    @pytest.mark.asyncio
    @patch("nanobot.providers.registry.get_capabilities", return_value=_make_mock_caps(audio=True))
    async def test_native_audio_preserves_all_content_types(self, mock_caps):
        from nanobot.agent.loop import AgentLoop

        loop = MagicMock(spec=AgentLoop)
        provider = _make_mock_provider(audio_support=True)
        content = [_make_text_block("Hi"), _make_audio_block(), _make_text_block("Bye")]
        messages = [{"role": "user", "content": content}]

        result = await AgentLoop._route_audio(
            loop, messages, "claude-sonnet-4-20250514", provider=provider
        )

        assert len(result[0]["content"]) == 3
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "audio"
        assert result[0]["content"][2]["type"] == "text"

    @pytest.mark.asyncio
    @patch("nanobot.providers.registry.get_capabilities", return_value=_make_mock_caps(audio=True))
    async def test_native_audio_multiple_messages(self, mock_caps):
        from nanobot.agent.loop import AgentLoop

        loop = MagicMock(spec=AgentLoop)
        provider = _make_mock_provider(audio_support=True)
        messages = [_make_message_with_audio("First"), _make_message_with_audio("Second")]

        result = await AgentLoop._route_audio(
            loop, messages, "claude-sonnet-4-20250514", provider=provider
        )

        assert len(result) == 2
        for msg in result:
            audio_blocks = [c for c in msg["content"] if c["type"] == "audio"]
            assert len(audio_blocks) == 1


class TestRouteAudioTranscription:
    """When model lacks native audio, transcribe via Groq Whisper."""

    @pytest.mark.asyncio
    @patch("nanobot.providers.registry.get_capabilities", return_value=_make_mock_caps(audio=False))
    async def test_transcribes_audio_when_model_lacks_support(self, mock_caps):
        from nanobot.agent.loop import AgentLoop

        loop = MagicMock(spec=AgentLoop)
        mock_provider = _make_mock_provider(audio_support=False)
        trans_provider = MagicMock()
        trans_provider.transcribe = AsyncMock(return_value="Hello, how are you?")
        loop._get_transcription_provider = AsyncMock(return_value=trans_provider)
        loop._transcribe_audio_blocks = AgentLoop._transcribe_audio_blocks.__get__(loop, AgentLoop)

        messages = [_make_message_with_audio()]
        result = await AgentLoop._route_audio(loop, messages, "gpt-4o", provider=mock_provider)

        assert len(result) == 1
        audio_blocks = [c for c in result[0]["content"] if c["type"] == "audio"]
        assert len(audio_blocks) == 0
        text_blocks = [c for c in result[0]["content"] if c["type"] == "text"]
        transcribed = [t for t in text_blocks if t["text"].startswith("[Transcribed Audio]")]
        assert len(transcribed) == 1
        assert "Hello, how are you?" in transcribed[0]["text"]
        trans_provider.transcribe.assert_called_once()

    @pytest.mark.asyncio
    @patch("nanobot.providers.registry.get_capabilities", return_value=_make_mock_caps(audio=False))
    async def test_transcribes_multiple_audio_blocks_concurrently(self, mock_caps):
        from nanobot.agent.loop import AgentLoop

        loop = MagicMock(spec=AgentLoop)
        mock_provider = _make_mock_provider(audio_support=False)
        trans_provider = MagicMock()
        trans_provider.transcribe = AsyncMock(side_effect=["First audio", "Second audio"])
        loop._get_transcription_provider = AsyncMock(return_value=trans_provider)
        loop._transcribe_audio_blocks = AgentLoop._transcribe_audio_blocks.__get__(loop, AgentLoop)

        messages = [_make_message_with_audio(num_audio=2)]
        result = await AgentLoop._route_audio(loop, messages, "gpt-4o", provider=mock_provider)

        assert trans_provider.transcribe.await_count == 2
        text_blocks = [c for c in result[0]["content"] if c["type"] == "text"]
        transcribed = [t for t in text_blocks if t["text"].startswith("[Transcribed Audio]")]
        assert len(transcribed) == 2
        assert "First audio" in transcribed[0]["text"]
        assert "Second audio" in transcribed[1]["text"]

    @pytest.mark.asyncio
    @patch("nanobot.providers.registry.get_capabilities", return_value=_make_mock_caps(audio=False))
    async def test_transcription_across_multiple_messages(self, mock_caps):
        from nanobot.agent.loop import AgentLoop

        loop = MagicMock(spec=AgentLoop)
        mock_provider = _make_mock_provider(audio_support=False)
        trans_provider = MagicMock()
        trans_provider.transcribe = AsyncMock(side_effect=["Msg1 audio", "Msg2 audio"])
        loop._get_transcription_provider = AsyncMock(return_value=trans_provider)
        loop._transcribe_audio_blocks = AgentLoop._transcribe_audio_blocks.__get__(loop, AgentLoop)

        messages = [
            _make_message_with_audio("Msg1"),
            _make_message_with_audio("Msg2"),
        ]
        result = await AgentLoop._route_audio(loop, messages, "gpt-4o", provider=mock_provider)

        assert trans_provider.transcribe.await_count == 2
        assert len(result) == 2

    @pytest.mark.asyncio
    @patch("nanobot.providers.registry.get_capabilities", return_value=_make_mock_caps(audio=False))
    async def test_empty_transcription_result_still_replaces_audio_block(self, mock_caps):
        from nanobot.agent.loop import AgentLoop

        loop = MagicMock(spec=AgentLoop)
        mock_provider = _make_mock_provider(audio_support=False)
        trans_provider = MagicMock()
        trans_provider.transcribe = AsyncMock(return_value="")
        loop._get_transcription_provider = AsyncMock(return_value=trans_provider)
        loop._transcribe_audio_blocks = AgentLoop._transcribe_audio_blocks.__get__(loop, AgentLoop)

        messages = [_make_message_with_audio()]
        result = await AgentLoop._route_audio(loop, messages, "gpt-4o", provider=mock_provider)

        audio_blocks = [c for c in result[0]["content"] if c["type"] == "audio"]
        assert len(audio_blocks) == 0
        text_blocks = [c for c in result[0]["content"] if c["type"] == "text"]
        transcribed = [t for t in text_blocks if t["text"].startswith("[Transcribed Audio]")]
        assert len(transcribed) == 1


class TestRouteAudioFallback:
    """When no transcription provider, strip audio blocks gracefully."""

    @pytest.mark.asyncio
    @patch("nanobot.providers.registry.get_capabilities", return_value=_make_mock_caps(audio=False))
    async def test_strips_audio_when_no_transcription_provider(self, mock_caps):
        from nanobot.agent.loop import AgentLoop

        loop = MagicMock(spec=AgentLoop)
        loop._get_transcription_provider = AsyncMock(return_value=None)

        messages = [_make_message_with_audio("Hello")]
        result = await AgentLoop._route_audio(loop, messages, "gpt-4o")

        assert len(result) == 1
        audio_blocks = [c for c in result[0]["content"] if c["type"] == "audio"]
        assert len(audio_blocks) == 0
        text_blocks = [c for c in result[0]["content"] if c["type"] == "text"]
        assert any("Hello" in t["text"] for t in text_blocks)


class TestRouteAudioEdgeCases:
    """Edge cases and early returns."""

    @pytest.mark.asyncio
    async def test_no_audio_returns_unchanged(self):
        from nanobot.agent.loop import AgentLoop

        loop = MagicMock(spec=AgentLoop)
        messages = [{"role": "user", "content": [_make_text_block("Hello")]}]

        result = await AgentLoop._route_audio(loop, messages, "gpt-4o")

        assert result == messages
        loop._get_transcription_provider.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_messages_list(self):
        from nanobot.agent.loop import AgentLoop

        loop = MagicMock(spec=AgentLoop)
        result = await AgentLoop._route_audio(loop, [], "gpt-4o")

        assert result == []

    @pytest.mark.asyncio
    @patch("nanobot.providers.registry.get_capabilities", return_value=_make_mock_caps(audio=False))
    async def test_mixed_audio_and_text_messages(self, mock_caps):
        from nanobot.agent.loop import AgentLoop

        loop = MagicMock(spec=AgentLoop)
        provider = _make_mock_provider(audio_support=False)
        trans_provider = MagicMock()
        trans_provider.transcribe = AsyncMock(return_value="Audio text")
        loop._get_transcription_provider = AsyncMock(return_value=trans_provider)
        loop._transcribe_audio_blocks = AgentLoop._transcribe_audio_blocks.__get__(loop, AgentLoop)

        messages = [
            {"role": "user", "content": [_make_text_block("Text only")]},
            _make_message_with_audio("With audio"),
        ]
        result = await AgentLoop._route_audio(loop, messages, "gpt-4o", provider=provider)

        assert len(result) == 2
        # First message unchanged (no audio)
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "text"
        # Second message has transcribed audio
        audio_blocks = [c for c in result[1]["content"] if c["type"] == "audio"]
        assert len(audio_blocks) == 0

    @pytest.mark.asyncio
    @patch("nanobot.providers.registry.get_capabilities", return_value=_make_mock_caps(audio=False))
    async def test_preserves_message_role_and_other_fields(self, mock_caps):
        from nanobot.agent.loop import AgentLoop

        loop = MagicMock(spec=AgentLoop)
        provider = _make_mock_provider(audio_support=False)
        trans_provider = MagicMock()
        trans_provider.transcribe = AsyncMock(return_value="Transcribed")
        loop._get_transcription_provider = AsyncMock(return_value=trans_provider)
        loop._transcribe_audio_blocks = AgentLoop._transcribe_audio_blocks.__get__(loop, AgentLoop)

        msg = {
            "role": "user",
            "content": [_make_audio_block()],
            "name": "test_user",
            "metadata": {"key": "value"},
        }
        result = await AgentLoop._route_audio(loop, [msg], "gpt-4o", provider=provider)

        assert result[0]["role"] == "user"
        assert result[0]["name"] == "test_user"
        assert result[0]["metadata"] == {"key": "value"}

    @pytest.mark.asyncio
    @patch("nanobot.providers.registry.get_capabilities", return_value=_make_mock_caps(audio=False))
    async def test_various_audio_mime_types(self, mock_caps):
        from nanobot.agent.loop import AgentLoop

        loop = MagicMock(spec=AgentLoop)
        provider = MagicMock()
        provider.transcribe = AsyncMock(return_value="Transcribed")
        loop._get_transcription_provider = AsyncMock(return_value=provider)

        for mime in ["audio/ogg", "audio/wav", "audio/mpeg", "audio/mp4"]:
            content = [_make_audio_block(mime=mime)]
            messages = [{"role": "user", "content": content}]
            result = await AgentLoop._route_audio(loop, messages, "gpt-4o")
            audio_blocks = [c for c in result[0]["content"] if c["type"] == "audio"]
            assert len(audio_blocks) == 0, f"Failed for mime type {mime}"
