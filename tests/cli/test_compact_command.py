"""Tests for /compact slash command."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.bus.events import Address, InboundMessage, OutboundMessage
from nanobot.session.manager import Session


def _make_loop():
    """Create a minimal AgentLoop with mocked dependencies."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    workspace = MagicMock()
    workspace.__truediv__ = MagicMock(return_value=MagicMock())

    with (
        patch("nanobot.agent.loop.ContextBuilder"),
        patch("nanobot.agent.loop.SessionManager"),
        patch("nanobot.agent.loop.SubagentManager"),
        patch("nanobot.agent.loop.MemoryConsolidator") as mock_consolidator,
    ):
        mock_consolidator.return_value.estimate_session_prompt_tokens.return_value = (0, "mock")
        loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)
    return loop, bus


class TestCompactCommand:
    @pytest.mark.asyncio
    async def test_compact_default_keeps_0(self):
        """Default N=0 when no argument is provided — compact everything."""
        from nanobot.command.builtin import cmd_compact
        from nanobot.command.router import CommandContext

        loop, _bus = _make_loop()
        session = Session(key="test-session")
        for i in range(20):
            session.add_message("user", f"msg{i}")
        session.last_consolidated = 0

        loop.sessions.get_or_create.return_value = session

        msg = InboundMessage(
            address=Address(channel="cli", segments=("direct",)),
            sender_id="user",
            content="/compact",
        )
        ctx = CommandContext(
            msg=msg, session=session, key=msg.session_key, raw="/compact", loop=loop
        )

        loop.memory_consolidator.archive_messages = AsyncMock(
            return_value="Summary of the conversation"
        )

        out = await cmd_compact(ctx)
        assert isinstance(out, OutboundMessage)
        assert "all history" in out.content
        assert "Summary of the conversation" in out.content

        # All 20 messages should be archived
        loop.memory_consolidator.archive_messages.assert_called_once()
        archived = loop.memory_consolidator.archive_messages.call_args[0][0]
        assert len(archived) == 20
        # Chunk replaced by summary at start_idx (0), so last_consolidated = 0
        assert session.last_consolidated == 0
        assert len(session.messages) == 1  # only the summary remains
        assert session.messages[0].get("_compact_summary") is True

    @pytest.mark.asyncio
    async def test_compact_injects_summary_message(self):
        """The summary is injected as an assistant message in the session."""
        from nanobot.command.builtin import cmd_compact
        from nanobot.command.router import CommandContext

        loop, _bus = _make_loop()
        session = Session(key="test-session")
        session.add_message("user", "hello")
        session.last_consolidated = 0

        loop.sessions.get_or_create.return_value = session

        msg = InboundMessage(
            address=Address(channel="cli", segments=("direct",)),
            sender_id="user",
            content="/compact",
        )
        ctx = CommandContext(
            msg=msg, session=session, key=msg.session_key, raw="/compact", loop=loop
        )

        summary_text = "Summary: discussed hello"
        loop.memory_consolidator.archive_messages = AsyncMock(return_value=summary_text)

        await cmd_compact(ctx)

        # Chunk [0:1] replaced by summary at position 0
        assert len(session.messages) == 1  # only the summary remains
        summary_msg = session.messages[0]
        assert summary_msg["role"] == "assistant"
        assert summary_msg["content"] == summary_text
        assert summary_msg.get("_compact_summary") is True

    @pytest.mark.asyncio
    async def test_compact_no_summary_preserves_messages(self):
        """If archive_messages returns None, messages are preserved and last_consolidated is NOT advanced."""
        from nanobot.command.builtin import cmd_compact
        from nanobot.command.router import CommandContext

        loop, _bus = _make_loop()
        session = Session(key="test-session")
        session.add_message("user", "hello")
        session.last_consolidated = 0

        loop.sessions.get_or_create.return_value = session

        msg = InboundMessage(
            address=Address(channel="cli", segments=("direct",)),
            sender_id="user",
            content="/compact",
        )
        ctx = CommandContext(
            msg=msg, session=session, key=msg.session_key, raw="/compact", loop=loop
        )

        loop.memory_consolidator.archive_messages = AsyncMock(return_value=None)

        out = await cmd_compact(ctx)

        # Summary failed — messages preserved, last_consolidated NOT advanced
        assert "failed" in out.content.lower()
        assert "preserved" in out.content.lower()
        assert len(session.messages) == 1  # original message still there
        assert session.messages[0]["content"] == "hello"
        assert session.last_consolidated == 0  # unchanged

    @pytest.mark.asyncio
    async def test_compact_positive_n(self):
        """Positive N argument keeps that many messages."""
        from nanobot.command.builtin import cmd_compact
        from nanobot.command.router import CommandContext

        loop, _bus = _make_loop()
        session = Session(key="test-session")
        for i in range(20):
            session.add_message("user", f"msg{i}")
        session.last_consolidated = 0

        loop.sessions.get_or_create.return_value = session

        msg = InboundMessage(
            address=Address(channel="cli", segments=("direct",)),
            sender_id="user",
            content="/compact 10",
        )
        ctx = CommandContext(
            msg=msg, session=session, key=msg.session_key, raw="/compact 10", loop=loop
        )

        loop.memory_consolidator.archive_messages = AsyncMock(
            return_value="Summary of the conversation"
        )

        out = await cmd_compact(ctx)
        assert isinstance(out, OutboundMessage)
        assert "10" in out.content
        assert "Summary of the conversation" in out.content

        archived = loop.memory_consolidator.archive_messages.call_args[0][0]
        assert len(archived) == 10  # 20 - 10 = 10
        # Chunk [0:10] replaced by summary at start_idx (0), so last_consolidated = 0
        assert session.last_consolidated == 0
        assert len(session.messages) == 11  # 1 summary + 10 kept

    @pytest.mark.asyncio
    async def test_compact_negative_n_uses_absolute(self):
        """Negative N is treated as its absolute value."""
        from nanobot.command.builtin import cmd_compact
        from nanobot.command.router import CommandContext

        loop, _bus = _make_loop()
        session = Session(key="test-session")
        for i in range(20):
            session.add_message("user", f"msg{i}")
        session.last_consolidated = 0

        loop.sessions.get_or_create.return_value = session

        msg = InboundMessage(
            address=Address(channel="cli", segments=("direct",)),
            sender_id="user",
            content="/compact -10",
        )
        ctx = CommandContext(
            msg=msg, session=session, key=msg.session_key, raw="/compact -10", loop=loop
        )

        loop.memory_consolidator.archive_messages = AsyncMock(
            return_value="Summary of the conversation"
        )

        out = await cmd_compact(ctx)
        assert isinstance(out, OutboundMessage)
        assert "10" in out.content
        assert "Summary of the conversation" in out.content

        archived = loop.memory_consolidator.archive_messages.call_args[0][0]
        assert len(archived) == 10  # Same as positive 10
        assert session.last_consolidated == 0  # Chunk replaced at start_idx (0)
        assert len(session.messages) == 11  # 1 summary + 10 kept

    @pytest.mark.asyncio
    async def test_compact_nothing_to_summarize(self):
        """Returns a message when there's nothing to consolidate."""
        from nanobot.command.builtin import cmd_compact
        from nanobot.command.router import CommandContext

        loop, _bus = _make_loop()
        session = Session(key="test-session")
        for i in range(3):
            session.add_message("user", f"msg{i}")
        session.last_consolidated = 3  # Already fully compacted

        loop.sessions.get_or_create.return_value = session

        msg = InboundMessage(
            address=Address(channel="cli", segments=("direct",)),
            sender_id="user",
            content="/compact",
        )
        ctx = CommandContext(
            msg=msg, session=session, key=msg.session_key, raw="/compact", loop=loop
        )

        loop.memory_consolidator.archive_messages = AsyncMock()

        out = await cmd_compact(ctx)
        assert isinstance(out, OutboundMessage)
        assert "Nothing to summarize" in out.content
        loop.memory_consolidator.archive_messages.assert_not_called()

    @pytest.mark.asyncio
    async def test_compact_invalid_n_falls_back_to_default(self):
        """Invalid N argument falls back to default of 0 (compact all)."""
        from nanobot.command.builtin import cmd_compact
        from nanobot.command.router import CommandContext

        loop, _bus = _make_loop()
        session = Session(key="test-session")
        for i in range(20):
            session.add_message("user", f"msg{i}")
        session.last_consolidated = 0

        loop.sessions.get_or_create.return_value = session

        msg = InboundMessage(
            address=Address(channel="cli", segments=("direct",)),
            sender_id="user",
            content="/compact abc",
        )
        ctx = CommandContext(
            msg=msg, session=session, key=msg.session_key, raw="/compact abc", loop=loop
        )

        loop.memory_consolidator.archive_messages = AsyncMock(
            return_value="Summary of the conversation"
        )

        out = await cmd_compact(ctx)
        assert isinstance(out, OutboundMessage)
        assert "all history" in out.content
        assert "Summary of the conversation" in out.content

        archived = loop.memory_consolidator.archive_messages.call_args[0][0]
        assert len(archived) == 20  # all 20 (default 0)
        # Chunk [0:20] replaced by summary at start_idx (0)
        assert session.last_consolidated == 0
        assert len(session.messages) == 1  # only the summary remains

    @pytest.mark.asyncio
    async def test_compact_respects_last_consolidated(self):
        """Only consolidates messages after last_consolidated."""
        from nanobot.command.builtin import cmd_compact
        from nanobot.command.router import CommandContext

        loop, _bus = _make_loop()
        session = Session(key="test-session")
        for i in range(20):
            session.add_message("user", f"msg{i}")
        session.last_consolidated = 10  # First 10 already consolidated

        loop.sessions.get_or_create.return_value = session

        msg = InboundMessage(
            address=Address(channel="cli", segments=("direct",)),
            sender_id="user",
            content="/compact 5",
        )
        ctx = CommandContext(
            msg=msg, session=session, key=msg.session_key, raw="/compact 5", loop=loop
        )

        loop.memory_consolidator.archive_messages = AsyncMock(
            return_value="Summary of the conversation"
        )

        out = await cmd_compact(ctx)
        assert isinstance(out, OutboundMessage)
        assert "Summary of the conversation" in out.content

        archived = loop.memory_consolidator.archive_messages.call_args[0][0]
        assert len(archived) == 5  # chunk = messages[10:15]
        # Summary replaces messages[10:15] at position 10, so last_consolidated = 10
        # get_history() returns messages[10:] which includes the summary
        assert session.last_consolidated == 10
        assert len(session.messages) == 16  # 10 original + 1 summary + 5 kept
        assert session.messages[10].get("_compact_summary") is True
