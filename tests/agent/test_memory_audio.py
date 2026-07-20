import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.memory import MemoryConsolidator


def _make_audio_block(mime_type="audio/wav", data=b"fake-audio-data"):
    data_uri = f"data:{mime_type};base64,{base64.b64encode(data).decode()}"
    return {"type": "audio", "data": data_uri, "mime_type": mime_type}


def _make_text_block(text="hello"):
    return {"type": "text", "text": text}


def _make_consolidator():
    """Create a MemoryConsolidator instance with mocked dependencies."""
    return MemoryConsolidator(
        workspace=Path("/tmp/test"),
        provider=MagicMock(),
        model="test",
        sessions=MagicMock(),
        context_window_tokens=8192,
        build_messages=MagicMock(return_value=[]),
        get_tool_definitions=MagicMock(return_value=[]),
    )


@pytest.mark.asyncio
async def test_transcribe_audio_no_audio_blocks():
    consolidator = _make_consolidator()
    messages = [{"role": "user", "content": "hello"}]
    result = await consolidator._transcribe_audio(messages)
    assert result == messages


@pytest.mark.asyncio
async def test_transcribe_audio_string_content():
    consolidator = _make_consolidator()
    messages = [{"role": "user", "content": "hello world"}]
    result = await consolidator._transcribe_audio(messages)
    assert result == messages


@pytest.mark.asyncio
async def test_transcribe_audio_no_provider_strips():
    mock_no_key = MagicMock()
    mock_no_key.config = MagicMock()
    mock_no_key.config.providers.groq.api_key = None

    with patch(
        "nanobot.services.transcription.TranscriptionService._get_provider",
        side_effect=Exception("No key"),
    ):
        consolidator = _make_consolidator()
        # Manually ensure the config matches what the service expects to fail
        consolidator.provider.generation.config = mock_no_key.config

        messages = [{"role": "user", "content": [_make_text_block("hello"), _make_audio_block()]}]
        result = await consolidator._transcribe_audio(messages)
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "hello"


@pytest.mark.asyncio
async def test_transcribe_audio_success():
    mock_service = MagicMock()
    mock_service.transcribe = AsyncMock(return_value="transcribed text")

    with patch(
        "nanobot.services.transcription.TranscriptionService.transcribe",
        side_effect=mock_service.transcribe,
    ):
        consolidator = _make_consolidator()
        messages = [
            {
                "role": "user",
                "content": [
                    _make_text_block("before"),
                    _make_audio_block(),
                    _make_text_block("after"),
                ],
            }
        ]
        result = await consolidator._transcribe_audio(messages)

        content = result[0]["content"]
        assert len(content) == 3
        assert content[0] == {"type": "text", "text": "before"}
        assert content[1] == {"type": "text", "text": "[Transcribed Audio]: transcribed text"}
        assert content[2] == {"type": "text", "text": "after"}


@pytest.mark.asyncio
async def test_transcribe_audio_multiple_blocks():
    mock_service = MagicMock()
    mock_service.transcribe = AsyncMock(side_effect=["first", "second"])

    with patch(
        "nanobot.services.transcription.TranscriptionService.transcribe",
        side_effect=mock_service.transcribe,
    ):
        consolidator = _make_consolidator()
        messages = [
            {"role": "user", "content": [_make_audio_block()]},
            {"role": "assistant", "content": [_make_text_block("response"), _make_audio_block()]},
        ]
        result = await consolidator._transcribe_audio(messages)

        assert result[0]["content"][0] == {"type": "text", "text": "[Transcribed Audio]: first"}
        assert result[1]["content"][1] == {"type": "text", "text": "[Transcribed Audio]: second"}


@pytest.mark.asyncio
async def test_transcribe_audio_ogg():
    mock_service = MagicMock()
    mock_service.transcribe = AsyncMock(return_value="ogg text")

    with patch(
        "nanobot.services.transcription.TranscriptionService.transcribe",
        side_effect=mock_service.transcribe,
    ):
        consolidator = _make_consolidator()
        messages = [
            {"role": "user", "content": [_make_audio_block(mime_type="audio/ogg; codecs=opus")]}
        ]
        await consolidator._transcribe_audio(messages)
        tmp_path = mock_service.transcribe.call_args[0][0]
        assert tmp_path.suffix == ".ogg"


@pytest.mark.asyncio
async def test_transcribe_audio_mp3():
    mock_service = MagicMock()
    mock_service.transcribe = AsyncMock(return_value="mp3 text")

    with patch(
        "nanobot.services.transcription.TranscriptionService.transcribe",
        side_effect=mock_service.transcribe,
    ):
        consolidator = _make_consolidator()
        messages = [{"role": "user", "content": [_make_audio_block(mime_type="audio/mpeg")]}]
        await consolidator._transcribe_audio(messages)
        tmp_path = mock_service.transcribe.call_args[0][0]
        assert tmp_path.suffix == ".mp3"


@pytest.mark.asyncio
async def test_transcribe_audio_invalid_data():
    mock_service = MagicMock()
    mock_service.transcribe = AsyncMock(return_value="text")

    with patch(
        "nanobot.services.transcription.TranscriptionService.transcribe",
        side_effect=mock_service.transcribe,
    ):
        consolidator = _make_consolidator()
        messages = [
            {
                "role": "user",
                "content": [{"type": "audio", "data": "not-valid!!!", "mime_type": "audio/wav"}],
            }
        ]
        result = await consolidator._transcribe_audio(messages)
        assert "[Transcription failed" in result[0]["content"][0]["text"]
        mock_service.transcribe.assert_not_called()


@pytest.mark.asyncio
async def test_transcribe_audio_provider_error():
    mock_service = MagicMock()
    mock_service.transcribe = AsyncMock(side_effect=Exception("API error"))

    with patch(
        "nanobot.services.transcription.TranscriptionService.transcribe",
        side_effect=mock_service.transcribe,
    ):
        consolidator = _make_consolidator()
        messages = [{"role": "user", "content": [_make_audio_block()]}]
        result = await consolidator._transcribe_audio(messages)
        assert "[Transcription failed" in result[0]["content"][0]["text"]
        assert "API error" in result[0]["content"][0]["text"]


@pytest.mark.asyncio
async def test_transcribe_audio_empty_result():
    mock_service = MagicMock()
    mock_service.transcribe = AsyncMock(return_value="")

    with patch(
        "nanobot.services.transcription.TranscriptionService.transcribe",
        side_effect=mock_service.transcribe,
    ):
        consolidator = _make_consolidator()
        messages = [{"role": "user", "content": [_make_audio_block()]}]
        result = await consolidator._transcribe_audio(messages)
        assert (
            result[0]["content"][0]["text"]
            == "[Transcribed Audio]: [Transcription returned empty result]"
        )


@pytest.mark.asyncio
async def test_transcribe_audio_temp_file_cleanup():
    mock_service = MagicMock()
    mock_service.transcribe = AsyncMock(return_value="text")

    with patch(
        "nanobot.services.transcription.TranscriptionService.transcribe",
        side_effect=mock_service.transcribe,
    ):
        consolidator = _make_consolidator()
        messages = [{"role": "user", "content": [_make_audio_block()]}]
        await consolidator._transcribe_audio(messages)
        tmp_path = mock_service.transcribe.call_args[0][0]
        assert not tmp_path.exists()


@pytest.mark.asyncio
async def test_strip_audio_preserves_non_audio():
    messages = [
        {
            "role": "user",
            "content": [
                _make_text_block("hello"),
                _make_audio_block(),
                {"type": "image", "source": {"type": "base64", "data": "img"}},
            ],
        }
    ]
    consolidator = _make_consolidator()
    result = consolidator._strip_audio(messages)
    content = result[0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image"


@pytest.mark.asyncio
async def test_consolidate_messages_calls_transcribe():
    consolidator = _make_consolidator()
    consolidator.store.consolidate = AsyncMock(return_value=True)

    messages = [{"role": "user", "content": "hello"}]
    result = await consolidator.consolidate_messages(messages)

    assert result is True
    consolidator.store.consolidate.assert_called_once()
