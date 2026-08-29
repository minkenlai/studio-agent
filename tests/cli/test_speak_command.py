"""Tests for /speak command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.bus.events import Address, InboundMessage
from nanobot.command.router import CommandContext


def _make_ctx(content, args, loop=None, session=None, metadata=None):
    """Helper to build a CommandContext for /speak tests."""
    msg = InboundMessage(
        address=Address(channel="telegram", segments=("c1",)),
        sender_id="tg:12345",
        content=content,
        metadata=metadata or {},
    )
    return CommandContext(
        msg=msg,
        session=session,
        key=msg.session_key,
        raw=content,
        loop=loop,
        args=args,
    )


@pytest.mark.asyncio
async def test_cmd_speak_with_text():
    """/speak with text argument synthesizes and returns audio file."""
    from nanobot.command.builtin import cmd_speak

    tts = MagicMock()
    tts.synthesize = AsyncMock(return_value=True)
    tts.output_ext = ".wav"

    loop = MagicMock()
    loop.get_tts_client = AsyncMock(return_value=tts)

    ctx = _make_ctx('/speak "Hello world"', '"Hello world"', loop=loop)

    with patch("nanobot.config.paths.get_media_dir") as mock_media_dir:
        mock_dir = MagicMock()
        mock_dir.__truediv__ = MagicMock(return_value=Path("/tmp/tts_abc12345.wav"))
        mock_media_dir.return_value = mock_dir

        response = await cmd_speak(ctx)

    assert response is not None
    assert response.content == ""
    assert len(response.media) == 1
    assert "tts_" in response.media[0]
    assert response.media[0].endswith(".wav")
    tts.synthesize.assert_called_once()
    assert tts.synthesize.call_args[0][0] == '"Hello world"'


@pytest.mark.asyncio
async def test_cmd_speak_no_args_uses_last_assistant():
    """/speak without args grabs last assistant message from session."""
    from nanobot.command.builtin import cmd_speak

    tts = MagicMock()
    tts.synthesize = AsyncMock(return_value=True)
    tts.output_ext = ".wav"

    loop = MagicMock()
    loop.get_tts_client = AsyncMock(return_value=tts)

    # TTSService.fetch_last_assistant() calls loop.sessions.get_or_create(session_key)
    session = MagicMock()
    session.messages = [
        {"role": "user", "content": "What is AI?"},
        {"role": "assistant", "content": "**AI** is `artificial` intelligence."},
    ]
    loop.sessions.get_or_create.return_value = session

    ctx = _make_ctx("/speak", "", loop=loop, session=session)

    with patch("nanobot.config.paths.get_media_dir") as mock_media_dir:
        mock_dir = MagicMock()
        mock_dir.__truediv__ = MagicMock(return_value=Path("/tmp/tts_xyz.wav"))
        mock_media_dir.return_value = mock_dir

        response = await cmd_speak(ctx)

    assert response is not None
    assert response.content == ""
    tts.synthesize.assert_called_once()
    # Markdown should be stripped
    call_text = tts.synthesize.call_args[0][0]
    assert "**" not in call_text
    assert "`" not in call_text
    assert "AI is artificial intelligence" in call_text


@pytest.mark.asyncio
async def test_cmd_speak_no_prior_message():
    """/speak with no args and no assistant messages returns an error."""
    from nanobot.command.builtin import cmd_speak

    tts = MagicMock()
    loop = MagicMock()
    loop.get_tts_client = AsyncMock(return_value=tts)

    session = MagicMock()
    session.messages = [{"role": "user", "content": "Hello?"}]

    ctx = _make_ctx("/speak", "", loop=loop, session=session)

    response = await cmd_speak(ctx)

    assert response is not None
    assert "No prior agent message found" in response.content
    tts.synthesize.assert_not_called()


@pytest.mark.asyncio
async def test_cmd_speak_no_loop():
    """When loop is None, returns an error."""
    from nanobot.command.builtin import cmd_speak

    ctx = _make_ctx("/speak hello", "hello", loop=None)

    response = await cmd_speak(ctx)

    assert response is not None
    assert "Error: agent loop not available" in response.content


@pytest.mark.asyncio
async def test_cmd_speak_no_tts_client():
    """When TTS client is None, returns a configuration error."""
    from nanobot.command.builtin import cmd_speak

    loop = MagicMock()
    loop.get_tts_client = AsyncMock(return_value=None)

    ctx = _make_ctx("/speak hello", "hello", loop=loop)

    response = await cmd_speak(ctx)

    assert response is not None
    assert "TTS is not configured" in response.content


@pytest.mark.asyncio
async def test_cmd_speak_synthesis_failure():
    """When synthesis fails, returns an error message."""
    from nanobot.command.builtin import cmd_speak

    tts = MagicMock()
    tts.synthesize = AsyncMock(return_value=False)

    loop = MagicMock()
    loop.get_tts_client = AsyncMock(return_value=tts)

    ctx = _make_ctx("/speak hello", "hello", loop=loop)

    with patch("nanobot.config.paths.get_media_dir") as mock_media_dir:
        mock_dir = MagicMock()
        mock_dir.__truediv__ = MagicMock(return_value=Path("/tmp/tts_fail.wav"))
        mock_media_dir.return_value = mock_dir

        response = await cmd_speak(ctx)

    assert response is not None
    assert "TTS synthesis failed" in response.content


@pytest.mark.asyncio
async def test_cmd_speak_strips_markdown():
    """Markdown formatting is stripped before synthesis."""
    from nanobot.command.builtin import cmd_speak

    tts = MagicMock()
    tts.synthesize = AsyncMock(return_value=True)

    loop = MagicMock()
    loop.get_tts_client = AsyncMock(return_value=tts)

    raw_text = "# Hello\n\n**bold** and `code` and [link](http://x.com)"
    ctx = _make_ctx(f"/speak {raw_text}", raw_text, loop=loop)

    with patch("nanobot.config.paths.get_media_dir") as mock_media_dir:
        mock_dir = MagicMock()
        mock_dir.__truediv__ = MagicMock(return_value=Path("/tmp/tts_md.wav"))
        mock_media_dir.return_value = mock_dir

        await cmd_speak(ctx)

    call_text = tts.synthesize.call_args[0][0]
    assert "**" not in call_text
    assert "`" not in call_text
    assert "http://x.com" not in call_text
    assert "# " not in call_text
    assert "Hello" in call_text
    assert "bold" in call_text
    assert "code" in call_text
    assert "link" in call_text


@pytest.mark.asyncio
async def test_cmd_speak_creates_session_if_none():
    """/speak without args and no session creates one from loop."""
    from nanobot.command.builtin import cmd_speak

    tts = MagicMock()
    tts.synthesize = AsyncMock(return_value=True)

    created_session = MagicMock()
    created_session.messages = [
        {"role": "assistant", "content": "Hello from assistant"},
    ]

    loop = MagicMock()
    loop.get_tts_client = AsyncMock(return_value=tts)
    loop.sessions.get_or_create.return_value = created_session

    ctx = _make_ctx("/speak", "", loop=loop, session=None)

    with patch("nanobot.config.paths.get_media_dir") as mock_media_dir:
        mock_dir = MagicMock()
        mock_dir.__truediv__ = MagicMock(return_value=Path("/tmp/tts_sess.wav"))
        mock_media_dir.return_value = mock_dir

        await cmd_speak(ctx)

    loop.sessions.get_or_create.assert_called_once()


@pytest.mark.asyncio
async def test_get_tts_client_returns_cached():
    """Second call to get_tts_client returns the same client."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = MagicMock(max_tokens=8192, context_max=200000)
    workspace = MagicMock()
    workspace.__truediv__ = MagicMock(return_value=MagicMock())

    with (
        patch("nanobot.agent.loop.ContextBuilder"),
        patch("nanobot.agent.loop.SessionManager"),
        patch("nanobot.agent.loop.SubagentManager"),
        patch("nanobot.agent.loop.MemoryConsolidator"),
    ):
        loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)

    # Pre-set the client
    mock_tts = MagicMock()
    loop._tts_client = mock_tts

    client = await loop.get_tts_client()
    assert client is mock_tts


@pytest.mark.asyncio
async def test_get_tts_client_no_config_returns_none():
    """get_tts_client returns None when config.tts is None."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = MagicMock(max_tokens=8192, context_max=200000)
    workspace = MagicMock()
    workspace.__truediv__ = MagicMock(return_value=MagicMock())

    with (
        patch("nanobot.agent.loop.ContextBuilder"),
        patch("nanobot.agent.loop.SessionManager"),
        patch("nanobot.agent.loop.SubagentManager"),
        patch("nanobot.agent.loop.MemoryConsolidator"),
    ):
        loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)

    loop.config.tts = None

    client = await loop.get_tts_client()
    assert client is None
