"""Tests for Silent Ack protocol tokens and channel integrations."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.protocol import parse_silent_ack
from nanobot.channels.telegram.runtime import TelegramChannel, TelegramConfig
from nanobot.channels.whatsapp.runtime import WhatsAppChannel, WhatsAppConfig


def test_parse_silent_ack() -> None:
    # 1. Unicode reaction tokens
    sig1 = parse_silent_ack("[REACTION: ✅]")
    assert sig1.action == "reaction"
    assert sig1.emoji == "✅"

    # 2. Shortcode reaction tokens
    sig2 = parse_silent_ack("[REACTION: :white_check_mark:]")
    assert sig2.action == "reaction"
    assert sig2.emoji == "✅"

    sig3 = parse_silent_ack("[REACTION: :thumbsup:]")
    assert sig3.action == "reaction"
    assert sig3.emoji == "👍"

    # 3. Trailing text truncation
    sig4 = parse_silent_ack("[REACTION: 🚀] Here is some trailing LLM explanation that must not be sent.")
    assert sig4.action == "reaction"
    assert sig4.emoji == "🚀"
    assert sig4.remaining_text == ""

    # 4. Silent / No-op tokens
    sig5 = parse_silent_ack("[SILENT]")
    assert sig5.action == "silent"
    assert sig5.emoji is None

    sig6 = parse_silent_ack("[NO_REPLY] I will not answer.")
    assert sig6.action == "silent"
    assert sig6.emoji is None

    sig7 = parse_silent_ack("[NOOP]")
    assert sig7.action == "silent"
    assert sig7.emoji is None

    sig7_none = parse_silent_ack("[NONE]")
    assert sig7_none.action == "silent"
    assert sig7_none.emoji is None

    sig7_reaction_silent = parse_silent_ack("[REACTION: silent]")
    assert sig7_reaction_silent.action == "silent"
    assert sig7_reaction_silent.emoji is None

    # 5. Normal text messages
    sig8 = parse_silent_ack("Can someone check the class schedule?")
    assert sig8.action == "text"
    assert sig8.emoji is None
    assert sig8.remaining_text == "Can someone check the class schedule?"

    # 6. None or empty
    sig9 = parse_silent_ack(None)
    assert sig9.action == "text"
    sig10 = parse_silent_ack("")
    assert sig10.action == "text"


@pytest.mark.asyncio
async def test_telegram_channel_silent_ack_reaction() -> None:
    bus = MessageBus()
    config = TelegramConfig(token="fake:token", allowed_chats=["12345"])
    chan = TelegramChannel(config=config, bus=bus)

    mock_app = MagicMock()
    mock_bot = MagicMock()
    mock_bot.set_message_reaction = AsyncMock()
    mock_bot.send_message = AsyncMock()
    chan._app = mock_app
    chan._wait_for_app = AsyncMock(return_value=mock_app)
    chan._get_bot = MagicMock(return_value=mock_bot)
    chan._stop_typing = MagicMock()

    # Outbound with reaction token
    msg = OutboundMessage(
        channel="telegram",
        chat_id="12345",
        content="[REACTION: ✅] Optional commentary",
        metadata={"message_id": "999"},
    )
    await chan.send(msg)

    # Verify reaction added, typing stopped, no text sent
    chan._stop_typing.assert_called_once_with("12345", metadata=msg.metadata)
    mock_bot.set_message_reaction.assert_called_once()
    assert mock_bot.set_message_reaction.call_args.kwargs["message_id"] == 999
    reaction_arg = mock_bot.set_message_reaction.call_args.kwargs["reaction"]
    assert len(reaction_arg) == 1
    assert reaction_arg[0].emoji == "✅"
    mock_bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_telegram_channel_silent_ack_silent() -> None:
    bus = MessageBus()
    config = TelegramConfig(token="fake:token", allowed_chats=["12345"])
    chan = TelegramChannel(config=config, bus=bus)

    mock_app = MagicMock()
    mock_bot = MagicMock()
    mock_bot.set_message_reaction = AsyncMock()
    mock_bot.send_message = AsyncMock()
    chan._app = mock_app
    chan._wait_for_app = AsyncMock(return_value=mock_app)
    chan._get_bot = MagicMock(return_value=mock_bot)
    chan._stop_typing = MagicMock()

    # Outbound with [SILENT]
    msg = OutboundMessage(
        channel="telegram",
        chat_id="12345",
        content="[SILENT]",
        metadata={"message_id": "888"},
    )
    await chan.send(msg)

    # Verify reaction removed (reaction=[]), typing stopped, no text sent
    chan._stop_typing.assert_called_once_with("12345", metadata=msg.metadata)
    mock_bot.set_message_reaction.assert_called_once_with(
        chat_id=12345,
        message_id=888,
        reaction=[],
    )
    mock_bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_whatsapp_channel_silent_ack() -> None:
    bus = MessageBus()
    config = WhatsAppConfig(session_dir="/tmp/fake_wa")
    chan = WhatsAppChannel(config=config, bus=bus)
    chan._connected = True
    chan._build_jid = MagicMock(return_value="12345@s.whatsapp.net")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()
    mock_client.send_reaction = AsyncMock()
    chan._client = mock_client
    chan._send_chat_presence = AsyncMock()

    # Outbound with [SILENT]
    msg_silent = OutboundMessage(
        channel="whatsapp",
        chat_id="12345@s.whatsapp.net",
        content="[SILENT]",
        metadata={"message_id": "wa_msg_1"},
    )
    await chan.send(msg_silent)
    chan._send_chat_presence.assert_called_once()
    mock_client.send_message.assert_not_called()

    # Outbound with [REACTION: 👍]
    msg_reaction = OutboundMessage(
        channel="whatsapp",
        chat_id="12345@s.whatsapp.net",
        content="[REACTION: 👍]",
        metadata={"message_id": "wa_msg_2"},
    )
    await chan.send(msg_reaction)
    mock_client.send_reaction.assert_called_once()
    assert mock_client.send_reaction.call_args[0][1] == "wa_msg_2"
    assert mock_client.send_reaction.call_args[0][2] == "👍"
    mock_client.send_message.assert_not_called()


def test_parse_silent_ack_edge_cases() -> None:
    # Shortcode without colons
    sig1 = parse_silent_ack("[REACTION: thumbsup]")
    assert sig1.action == "reaction"
    assert sig1.emoji == "👍"

    sig2 = parse_silent_ack("[REACTION: +1]")
    assert sig2.action == "reaction"
    assert sig2.emoji == "👍"

    # Common aliases
    sig3 = parse_silent_ack("[REACTION: :thumbs_up:]")
    assert sig3.action == "reaction"
    assert sig3.emoji == "👍"

    # Explicit clear / none reaction
    sig4 = parse_silent_ack("[REACTION: none]")
    assert sig4.action == "silent"
    assert sig4.emoji is None

    # Variants of silent tokens
    for variant in ["[NO-REPLY]", "[NOREPLY]", "[NO-OP]", "[NO_OP]", "[silent]", "[noop]"]:
        sig = parse_silent_ack(variant)
        assert sig.action == "silent", f"Failed for variant {variant}"

    # Backtick-wrapped tokens
    sig5 = parse_silent_ack("`[SILENT]`")
    assert sig5.action == "silent"

    sig6 = parse_silent_ack("`[REACTION: 👍]`")
    assert sig6.action == "reaction"
    assert sig6.emoji == "👍"


@pytest.mark.asyncio
async def test_telegram_channel_silent_ack_edge_cases() -> None:
    bus = MessageBus()
    config = TelegramConfig(token="fake:token", allowed_chats=["12345"])
    chan = TelegramChannel(config=config, bus=bus)

    mock_app = MagicMock()
    mock_bot = MagicMock()
    mock_bot.set_message_reaction = AsyncMock(side_effect=Exception("Reaction API error"))
    mock_bot.send_message = AsyncMock()
    chan._app = mock_app
    chan._wait_for_app = AsyncMock(return_value=mock_app)
    chan._get_bot = MagicMock(return_value=mock_bot)
    chan._stop_typing = MagicMock()

    # 1. Invalid non-integer message_id: should suppress ValueError/TypeError and send no text
    msg_invalid = OutboundMessage(
        channel="telegram",
        chat_id="12345",
        content="[REACTION: 👍]",
        metadata={"message_id": "not_an_int"},
    )
    await chan.send(msg_invalid)
    chan._stop_typing.assert_called_with("12345", metadata=msg_invalid.metadata)
    mock_bot.set_message_reaction.assert_not_called()
    mock_bot.send_message.assert_not_called()

    # 2. Missing metadata entirely
    msg_empty = OutboundMessage(
        channel="telegram",
        chat_id="12345",
        content="[SILENT]",
        metadata={},
    )
    await chan.send(msg_empty)
    mock_bot.send_message.assert_not_called()

    # 3. Exception in set_message_reaction should be suppressed, zero text sent
    msg_err = OutboundMessage(
        channel="telegram",
        chat_id="12345",
        content="[REACTION: 👍]",
        metadata={"message_id": "111"},
    )
    await chan.send(msg_err)
    mock_bot.set_message_reaction.assert_called_once()
    mock_bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_whatsapp_channel_silent_ack_edge_cases() -> None:
    bus = MessageBus()
    config = WhatsAppConfig(session_dir="/tmp/fake_wa")
    chan = WhatsAppChannel(config=config, bus=bus)
    chan._connected = True
    chan._build_jid = MagicMock(return_value="12345@s.whatsapp.net")

    # 1. Client without send_reaction attribute
    mock_client_no_reaction = MagicMock(spec=["send_message"])
    mock_client_no_reaction.send_message = AsyncMock()
    chan._client = mock_client_no_reaction
    chan._send_chat_presence = AsyncMock()

    msg_reaction = OutboundMessage(
        channel="whatsapp",
        chat_id="12345@s.whatsapp.net",
        content="[REACTION: 👍]",
        metadata={"message_id": "wa_msg_err"},
    )
    # Should not raise AttributeError when send_reaction doesn't exist
    await chan.send(msg_reaction)
    mock_client_no_reaction.send_message.assert_not_called()

    # 2. send_reaction raises exception
    mock_client_err = MagicMock()
    mock_client_err.send_message = AsyncMock()
    mock_client_err.send_reaction = AsyncMock(side_effect=RuntimeError("WA reaction failure"))
    chan._client = mock_client_err

    # Should suppress exception and not send text
    await chan.send(msg_reaction)
    mock_client_err.send_message.assert_not_called()
