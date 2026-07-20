"""Base channel interface for chat platforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import Address, InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus


class BaseChannel(ABC):
    """
    Abstract base class for chat channel implementations.

    Each channel (Telegram, Discord, etc.) should implement this interface
    to integrate with the nanobot message bus.
    """

    name: str = "base"
    display_name: str = "Base"
    aliases: list[str] = []

    def __init__(self, config: Any, bus: MessageBus):
        """
        Initialize the channel.

        Args:
            config: Channel-specific configuration.
            bus: The message bus for communication.
        """
        self.config = config
        self.bus = bus
        self._running = False

    async def transcribe_audio(self, file_path: str | Path) -> str:
        """Transcribe an audio file using the system transcription service. Returns empty string on failure."""
        try:
            from nanobot.services.transcription import TranscriptionService

            service = TranscriptionService(self.global_config)
            return await service.transcribe(file_path)
        except Exception as e:
            logger.warning("{}: audio transcription failed: {}", self.name, e)
            return ""

    async def login(self, force: bool = False) -> bool:
        """
        Perform channel-specific interactive login (e.g. QR code scan).

        Args:
            force: If True, ignore existing credentials and force re-authentication.

        Returns True if already authenticated or login succeeds.
        Override in subclasses that support interactive login.
        """
        return True

    @abstractmethod
    async def start(self) -> None:
        """
        Start the channel and begin listening for messages.

        This should be a long-running async task that:
        1. Connects to the chat platform
        2. Listens for incoming messages
        3. Forwards messages to the bus via _handle_message()
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel and clean up resources."""
        pass

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """
        Send a message through this channel.

        Args:
            msg: The message to send.

        Implementations should raise on delivery failure so the channel manager
        can apply any retry policy in one place.
        """
        pass

    async def send_delta(self, msg: OutboundMessage) -> None:
        """Deliver a streaming text chunk.

        Override in subclasses to enable streaming. Implementations should
        raise on delivery failure so the channel manager can retry.

        Streaming contract: ``_stream_delta`` is a chunk, ``_stream_end`` ends
        the current segment, and stateful implementations must key buffers by
        ``msg.address.to_uri()`` rather than only by ``chat_id``.
        """
        pass

    @property
    def supports_streaming(self) -> bool:
        """True when config enables streaming AND this subclass implements send_delta."""
        cfg = self.config
        streaming = (
            cfg.get("streaming", False)
            if isinstance(cfg, dict)
            else getattr(cfg, "streaming", False)
        )
        return bool(streaming) and type(self).send_delta is not BaseChannel.send_delta

    def is_allowed(self, sender_id: str) -> bool:
        """Check if *sender_id* is permitted.  Empty list → deny all; ``"*"`` → allow all."""
        allow_list = getattr(self.config, "allow_from", [])
        if not allow_list:
            logger.warning("{}: allow_from is empty — all access denied", self.name)
            return False
        if "*" in allow_list:
            return True
        return str(sender_id) in allow_list

    async def _handle_message(
        self,
        sender_id: str,
        content: str,
        address: Address | None = None,
        media: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        # Legacy fields for backward compatibility
        chat_id: str | None = None,
        message_thread_id: int | None = None,
        session_key: str | None = None,
    ) -> None:
        """
        Handle an incoming message from the chat platform.

        Args:
            sender_id: The sender's identifier.
            content: Message text content.
            address: The unified conversation address.
            media: Optional list of media URLs.
            metadata: Optional channel-specific metadata.
        """
        # Construction of address from legacy fields if missing
        if address is None:
            if chat_id is not None:
                segments = [str(chat_id)]
                if message_thread_id is not None:
                    segments.append(str(message_thread_id))
                address = Address(channel=self.name, segments=tuple(segments))
            elif session_key is not None:
                address = Address.from_uri(session_key)

        if not address:
            logger.error("Missing address in _handle_message for channel {}", self.name)
            return

        if not self.is_allowed(sender_id):
            logger.warning(
                "Access denied for sender {} on channel {}. "
                "Add them to allowFrom list in config to grant access.",
                sender_id,
                self.name,
            )
            return

        meta = metadata or {}
        if self.supports_streaming:
            meta = {**meta, "_wants_stream": True}

        msg = InboundMessage(
            address=address,
            sender_id=str(sender_id),
            content=content,
            media=media or [],
            metadata=meta,
        )

        await self.bus.publish_inbound(msg)

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """Return default config for onboard. Override in plugins to auto-populate config.json."""
        return {"enabled": False}

    @property
    def is_running(self) -> bool:
        """Check if the channel is running."""
        return self._running
