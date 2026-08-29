"""Tests for the /new command (D8): only the unstaged tail is archived.

Ported from tests/agent/test_consolidate_offset.py (TestNewCommandArchival),
rewritten for the unified staged_boundary model: messages before SB were
already distilled into STAGING.md, so re-archiving them would duplicate.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.memory import MemoryConsolidator
from nanobot.command.builtin import cmd_new
from nanobot.session.manager import Session


def _ctx(session, raw="/new"):
    mock_loop = MagicMock()
    mock_loop.sessions = MagicMock()
    mock_loop.sessions.get_or_create.return_value = session
    # REAL consolidator so the session lock is a genuine asyncio.Lock (F1);
    # the background archive coroutine uses its real (mocked-LLM) pipeline.
    provider = MagicMock()
    provider.generation = None
    consolidator = MemoryConsolidator(
        workspace=Path(tempfile.gettempdir()),
        provider=provider,
        model="test-model",
        sessions=mock_loop.sessions,
        context_window_tokens=10_000_000,
        build_messages=lambda *args, **kwargs: [],
        get_tool_definitions=lambda: [],
        max_completion_tokens=128,
    )
    consolidator.consolidate_messages = AsyncMock(return_value=("summary", None))
    mock_loop.memory_consolidator = consolidator
    mock_loop._schedule_background = MagicMock()
    ctx = MagicMock()
    ctx.loop = mock_loop
    ctx.session = session
    ctx.key = session.key
    ctx.msg = MagicMock(address=MagicMock(channel="tg", sender_id="1"))
    ctx.raw = raw
    return ctx, mock_loop


class TestCmdNew:
    async def test_new_archives_only_unstaged_tail(self):
        """Snapshot is messages[SB:] — the staged region is NOT re-archived."""
        session = Session(key="t:1")
        for i in range(8):
            session.add_message("user", f"msg{i}")
        session.staged_boundary = 5  # [0:5] already in STAGING.md

        ctx, mock_loop = _ctx(session)
        resp = await cmd_new(ctx)

        assert resp.content == "New session started."
        mock_loop._schedule_background.assert_called_once()
        await mock_loop._schedule_background.call_args[0][0]  # drain the task

        # archive_messages (real) funnels to the mocked consolidate_messages seam.
        archived = mock_loop.memory_consolidator.consolidate_messages.call_args[0][0]
        assert [m["content"] for m in archived] == ["msg5", "msg6", "msg7"]

    async def test_new_clears_session_and_resets_sb(self):
        session = Session(key="t:2")
        for i in range(4):
            session.add_message("user", f"msg{i}")
        session.staged_boundary = 2

        ctx, mock_loop = _ctx(session)
        await cmd_new(ctx)

        assert session.messages == []
        assert session.staged_boundary == 0
        mock_loop.sessions.save.assert_called_with(session)
        mock_loop.sessions.invalidate.assert_called_with(session.key)
        await mock_loop._schedule_background.call_args[0][0]  # drain the task

    async def test_new_no_archival_when_nothing_unstaged(self):
        """Everything already staged -> no background archival scheduled."""
        session = Session(key="t:3")
        for i in range(4):
            session.add_message("user", f"msg{i}")
        session.staged_boundary = 4  # fully staged

        ctx, mock_loop = _ctx(session)
        await cmd_new(ctx)

        assert session.messages == []
        mock_loop.memory_consolidator.consolidate_messages.assert_not_awaited()
        mock_loop._schedule_background.assert_not_called()

    async def test_new_empty_session_no_archival(self):
        session = Session(key="t:4")

        ctx, mock_loop = _ctx(session)
        await cmd_new(ctx)

        assert session.messages == []
        mock_loop._schedule_background.assert_not_called()


@pytest.mark.asyncio
async def test_cmd_new_acquires_consolidator_lock():
    """F1: /new's snapshot+clear is serialized with background consolidation."""
    session = Session(key="test:new_lock")
    for i in range(4):
        session.add_message("user", f"m{i}")
    session.staged_boundary = 2

    ctx, mock_loop = _ctx(session)
    lock = mock_loop.memory_consolidator.get_lock(session.key)
    await lock.acquire()
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(cmd_new(ctx), 0.1)
    finally:
        lock.release()

    # Untouched — blocked on the lock before snapshot/clear.
    assert len(session.messages) == 4
    assert session.staged_boundary == 2
