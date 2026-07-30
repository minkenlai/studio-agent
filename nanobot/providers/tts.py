"""Text-to-Speech (TTS) provider interface and implementations."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from openai import AsyncOpenAI

if TYPE_CHECKING:
    from nanobot.config.schema import TTSConfig

# Chatterbox / PyTorch — optional, guarded so the module loads without them.
try:
    from chatterbox.tts_turbo import ChatterboxTurboTTS
except ImportError:
    ChatterboxTurboTTS = None

try:
    import torchaudio
except ImportError:
    torchaudio = None


class TTSClient(ABC):
    """Base interface for Text-to-Speech providers."""

    def __init__(self, api_key: str | None = None, api_base: str | None = None):
        self.api_key = api_key
        self.api_base = api_base

    @abstractmethod
    async def synthesize(self, text: str, output_path: Path, voice: str = "alloy") -> bool:
        """
        Synthesize text to an audio file.
        Returns True if successful, False otherwise.
        """
        pass


class OpenAITTProvider(TTSClient):
    """OpenAI TTS implementation using the /v1/audio/speech endpoint."""

    def __init__(
        self, api_key: str | None = None, api_base: str | None = None, default_voice: str = "alloy"
    ):
        super().__init__(api_key, api_base)
        self.default_voice = default_voice
        self._client = AsyncOpenAI(api_key=api_key or "no-key", base_url=api_base)

    async def synthesize(self, text: str, output_path: Path, voice: str | None = None) -> bool:
        """Synthesize text to audio using OpenAI TTS."""
        if not text:
            return False

        target_voice = voice or self.default_voice
        try:
            response = await self._client.audio.speech.create(
                model="tts-1",
                voice=target_voice,
                input=text,
            )

            # Save the binary content to disk
            content = await response.content
            output_path.write_bytes(content)
            return True
        except Exception as e:
            logger.error("OpenAI TTS synthesis failed for {}: {}", text[:50], e)
            return False


class ChatterboxTTSProvider(TTSClient):
    """Local TTS using Chatterbox Nano/Turbo (no API key required).

    Model loading is deferred until the first :meth:`synthesize` call so that
    importing this module or instantiating the provider never blocks the event
    loop with heavy weights downloads or GPU allocations.

    Supported model variants:

    * **nano** — 110 M params, English, ``nano=True`` in
      :class:`ChatterboxTurboTTS`.  Runs comfortably on CPU.
    * **turbo** — full-size model, ``nano=False``.  Same class under the hood.
    """

    # Mapping of variant name → internal identifier.
    _VARIANTS: dict[str, str] = {"nano": "nano", "turbo": "turbo"}

    def __init__(
        self,
        model_variant: str = "nano",
        device: str = "cpu",
        audio_prompt_path: str | None = None,
        **_kwargs: Any,
    ) -> None:
        """
        Args:
            model_variant: One of ``"nano"`` (default) or ``"turbo"``.
            device: PyTorch device string, e.g. ``"cpu"`` or ``"cuda"``.
            audio_prompt_path: Optional path to a reference WAV for voice cloning.
            **_kwargs: Absorbs *api_key* / *api_base* forwarded by callers that
                don't distinguish between providers (the base :class:`TTSClient`
                interface).
        """
        # Call parent — chatterbox doesn't need credentials, but the registry
        # may still pass *api_key* / *api_base* through the shared interface.
        super().__init__(**_kwargs)

        if model_variant not in self._VARIANTS:
            raise ValueError(
                f"Unknown Chatterbox model_variant {model_variant!r} — "
                f"choose from {list(self._VARIANTS)}"
            )

        self.model_variant: str = model_variant
        self.device: str = device
        self.audio_prompt_path: str | None = audio_prompt_path
        self._model: ChatterboxTurboTTS | None = None
        self._loaded: bool = False

    # ---- lazy loading --------------------------------------------------

    def _ensure_loaded(self) -> ChatterboxTurboTTS:
        """Load the Chatterbox model on first call (thread-safe after init)."""
        if self._loaded:
            return self._model  # type: ignore[return-value]

        if ChatterboxTurboTTS is None:
            raise RuntimeError(
                "The 'chatterbox-tts' package is not installed. "
                "Install it with: pip install chatterbox-tts"
            )
        if torchaudio is None:
            raise RuntimeError(
                "The 'torchaudio' package is not installed. Install it with: pip install torchaudio"
            )

        logger.info(
            "Loading Chatterbox model (variant={}, device={})",
            self.model_variant,
            self.device,
        )

        try:
            is_nano = self.model_variant == "nano"
            self._model = ChatterboxTurboTTS.from_pretrained(device=self.device, nano=is_nano)
            self._loaded = True
            logger.info("Chatterbox model loaded successfully")
        except Exception as e:
            logger.error("Failed to load Chatterbox model: {}", e)
            raise

        return self._model  # type: ignore[return-value]

    # ---- inference (blocking, must be called from a thread) -------------

    def _synthesize_sync(self, text: str, output_path: Path) -> bool:
        """Run the actual model inference and save WAV.  *Not* async."""
        model = self._ensure_loaded()

        wav = model.generate(text, audio_prompt_path=self.audio_prompt_path)
        torchaudio.save(str(output_path), wav, model.sr)

        # Use a try-except here so mocks (which don't write files) don't crash
        try:
            logger.debug("Chatterbox saved {} bytes to {}", output_path.stat().st_size, output_path)
        except Exception:
            pass
        return True

    # ---- public async interface ----------------------------------------

    async def synthesize(self, text: str, output_path: Path, voice: str | None = None) -> bool:
        """Synthesize *text* into a WAV file at *output_path*.

        The blocking model inference is offloaded to a background thread via
        :func:`asyncio.to_thread` so the event loop stays responsive.

        The *voice* parameter is accepted for interface compatibility but is
        not used by Chatterbox (voice cloning is controlled via
        *audio_prompt_path* on construction).

        Returns:
            ``True`` on success, ``False`` on any error.
        """
        if not text:
            return False

        try:
            await asyncio.to_thread(self._synthesize_sync, text, output_path)
            return True
        except RuntimeError as e:
            # Re-raise install-time errors so callers can surface them.
            logger.error("Chatterbox synthesis failed: {}", e)
            return False
        except Exception as e:
            logger.error("Chatterbox synthesis failed for {}: {}", text[:50], e)
            return False


def make_tts_provider(config: "TTSConfig") -> TTSClient:
    """Build a TTSClient from TTSConfig."""
    if config.provider == "chatterbox":
        cb = config.chatterbox
        return ChatterboxTTSProvider(
            model_variant=cb.model_variant,
            device=cb.device,
            audio_prompt_path=cb.audio_prompt_path,
        )
    elif config.provider == "openai":
        oa = config.openai
        return OpenAITTProvider(
            api_key=oa.api_key,
            api_base=oa.api_base,
            default_voice=oa.default_voice,
        )
    else:
        raise ValueError(f"Unknown TTS provider: {config.provider!r}")
