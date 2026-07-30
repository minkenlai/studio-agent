"""Tests for ChatterboxTTSProvider.

Validates:
- Init validation (invalid model_variant raises ValueError)
- Empty text returns False immediately
- Lazy loading (from_pretrained called only on first synthesize)
- Model caching (from_pretrained not called on subsequent synthesize calls)
- Success path (torchaudio.save called, returns True)
- Error path (returns False on any exception)
- Model params (from_pretrained called with correct device/nano)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nanobot.providers.tts import ChatterboxTTSProvider

# ---------------------------------------------------------------------------
# Init validation
# ---------------------------------------------------------------------------


def test_init_invalid_model_variant():
    """Invalid model_variant raises ValueError."""
    with pytest.raises(ValueError, match="super-nano"):
        ChatterboxTTSProvider(model_variant="super-nano")


# ---------------------------------------------------------------------------
# Empty text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_empty_text():
    """Empty text returns False immediately, no model load."""
    provider = ChatterboxTTSProvider()
    result = await provider.synthesize("", Path("test.wav"))
    assert result is False
    # Model must never have been loaded
    assert provider._loaded is False


# ---------------------------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------------------------


def test_from_pretrained_not_called_on_init():
    """from_pretrained is NOT called during __init__."""
    with patch("nanobot.providers.tts.ChatterboxTurboTTS.from_pretrained") as mock_from_pretrained:
        ChatterboxTTSProvider(model_variant="nano", device="cpu")
        mock_from_pretrained.assert_not_called()


@pytest.mark.asyncio
async def test_from_pretrained_called_on_first_synthesize():
    """from_pretrained IS called on the first synthesize call."""
    mock_model = MagicMock()
    mock_model.sr = 24000
    mock_model.generate.return_value = MagicMock()

    with patch(
        "nanobot.providers.tts.ChatterboxTurboTTS.from_pretrained",
        return_value=mock_model,
    ) as mock_from_pretrained:
        with patch("torchaudio.save"):
            provider = ChatterboxTTSProvider(model_variant="nano", device="cpu")
            await provider.synthesize("Hello", Path("test.wav"))

            mock_from_pretrained.assert_called_once()


@pytest.mark.asyncio
async def test_from_pretrained_not_called_on_second_synthesize():
    """from_pretrained is NOT called on subsequent synthesize calls."""
    mock_model = MagicMock()
    mock_model.sr = 24000
    mock_model.generate.return_value = MagicMock()

    with patch(
        "nanobot.providers.tts.ChatterboxTurboTTS.from_pretrained",
        return_value=mock_model,
    ) as mock_from_pretrained:
        with patch("torchaudio.save"):
            provider = ChatterboxTTSProvider(model_variant="nano", device="cpu")

            # First call loads the model
            await provider.synthesize("Hello", Path("test.wav"))
            assert mock_from_pretrained.call_count == 1

            # Second call should reuse cached model
            await provider.synthesize("World", Path("test2.wav"))
            assert mock_from_pretrained.call_count == 1


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_success():
    """synthesize returns True and calls torchaudio.save."""
    mock_model = MagicMock()
    mock_model.sr = 24000
    mock_model.generate.return_value = MagicMock()

    with patch(
        "nanobot.providers.tts.ChatterboxTurboTTS.from_pretrained",
        return_value=mock_model,
    ):
        with patch("torchaudio.save") as mock_save:
            provider = ChatterboxTTSProvider(model_variant="nano", device="cpu")
            result = await provider.synthesize("Hello world", Path("output.wav"))

            assert result is True
            mock_model.generate.assert_called_once_with("Hello world", audio_prompt_path=None)
            mock_save.assert_called_once()
            # Verify the arguments: path, waveform tensor, sample rate
            call_args = mock_save.call_args
            assert str(Path("output.wav")) == call_args[0][0]
            assert mock_model.sr == call_args[0][2]


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_error_on_generate():
    """synthesize returns False when model.generate raises."""
    mock_model = MagicMock()
    mock_model.generate.side_effect = RuntimeError("model crashed")

    with patch(
        "nanobot.providers.tts.ChatterboxTurboTTS.from_pretrained",
        return_value=mock_model,
    ):
        provider = ChatterboxTTSProvider()
        result = await provider.synthesize("Hello", Path("test.wav"))
        assert result is False


@pytest.mark.asyncio
async def test_synthesize_error_on_save():
    """synthesize returns False when torchaudio.save raises."""
    mock_model = MagicMock()
    mock_model.sr = 24000
    mock_model.generate.return_value = MagicMock()

    with patch(
        "nanobot.providers.tts.ChatterboxTurboTTS.from_pretrained",
        return_value=mock_model,
    ):
        with patch("torchaudio.save", side_effect=OSError("disk full")):
            provider = ChatterboxTTSProvider()
            result = await provider.synthesize("Hello", Path("test.wav"))
            assert result is False


@pytest.mark.asyncio
async def test_synthesize_missing_library():
    """synthesize returns False when chatterbox-tts is not installed."""
    with patch("nanobot.providers.tts.ChatterboxTurboTTS", None):
        provider = ChatterboxTTSProvider()
        result = await provider.synthesize("Hello", Path("test.wav"))
        assert result is False


# ---------------------------------------------------------------------------
# Model params
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_params_nano():
    """from_pretrained is called with nano=True and correct device for 'nano' variant."""
    mock_model = MagicMock()
    mock_model.sr = 24000
    mock_model.generate.return_value = MagicMock()

    with patch(
        "nanobot.providers.tts.ChatterboxTurboTTS.from_pretrained",
        return_value=mock_model,
    ) as mock_from_pretrained:
        with patch("torchaudio.save"):
            provider = ChatterboxTTSProvider(model_variant="nano", device="cpu")
            await provider.synthesize("Hello", Path("test.wav"))

            mock_from_pretrained.assert_called_once_with(device="cpu", nano=True)


@pytest.mark.asyncio
async def test_model_params_turbo():
    """from_pretrained is called with nano=False for 'turbo' variant."""
    mock_model = MagicMock()
    mock_model.sr = 24000
    mock_model.generate.return_value = MagicMock()

    with patch(
        "nanobot.providers.tts.ChatterboxTurboTTS.from_pretrained",
        return_value=mock_model,
    ) as mock_from_pretrained:
        with patch("torchaudio.save"):
            provider = ChatterboxTTSProvider(model_variant="turbo", device="cuda")
            await provider.synthesize("Hello", Path("test.wav"))

            mock_from_pretrained.assert_called_once_with(device="cuda", nano=False)


@pytest.mark.asyncio
async def test_audio_prompt_passed_to_generate():
    """audio_prompt_path is forwarded to model.generate()."""
    mock_model = MagicMock()
    mock_model.sr = 24000
    mock_model.generate.return_value = MagicMock()

    with patch(
        "nanobot.providers.tts.ChatterboxTurboTTS.from_pretrained",
        return_value=mock_model,
    ):
        with patch("torchaudio.save"):
            provider = ChatterboxTTSProvider(
                model_variant="nano",
                device="cpu",
                audio_prompt_path="/path/to/prompt.wav",
            )
            await provider.synthesize("Hello", Path("test.wav"))

            mock_model.generate.assert_called_once_with(
                "Hello", audio_prompt_path="/path/to/prompt.wav"
            )
