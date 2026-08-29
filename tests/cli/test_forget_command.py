import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nanobot.agent.memory import MemoryConsolidator
from nanobot.command.builtin import cmd_forget
from nanobot.session.manager import Session


def _ctx(session, raw="/forget"):
    mock_loop = MagicMock()
    mock_loop.sessions = MagicMock()
    mock_loop.sessions.get_or_create.return_value = session
    # REAL consolidator so the session lock is a genuine asyncio.Lock (F1).
    provider = MagicMock()
    provider.generation = None
    mock_loop.memory_consolidator = MemoryConsolidator(
        workspace=Path(tempfile.gettempdir()),
        provider=provider,
        model="test-model",
        sessions=mock_loop.sessions,
        context_window_tokens=10_000_000,
        build_messages=lambda *args, **kwargs: [],
        get_tool_definitions=lambda: [],
        max_completion_tokens=128,
    )
    ctx = MagicMock()
    ctx.loop = mock_loop
    ctx.session = session
    ctx.key = session.key
    ctx.msg = MagicMock(address=MagicMock(channel="tg", sender_id="123"), sender_id="123")
    ctx.raw = raw
    return ctx, mock_loop


async def test_cmd_forget_keeps_pinned_context():
    session = Session(key="test:forget")
    session.messages = [
        {
            "role": "user",
            "content": "[Previous session context]: doing X",
            "_session_context": True,
            "_compact_summary": True,
        },
        {"role": "user", "content": "m1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "m2"},
        {"role": "assistant", "content": "a2"},
    ]
    session.staged_boundary = 2  # m1/a1 already staged

    ctx, mock_loop = _ctx(session)
    response = await cmd_forget(ctx)

    # Pinned ctx preserved; staged + fresh tail dropped (lossless: in STAGING.md).
    assert len(session.messages) == 1
    assert session.messages[0]["content"] == "[Previous session context]: doing X"
    assert session.staged_boundary == 1
    assert "Forgot 4 message(s)" in response.content
    mock_loop.sessions.save.assert_called_with(session)


async def test_cmd_forget_no_pinned_context():
    session = Session(key="test:forget_zero")
    session.messages = [
        {"role": "user", "content": "m1"},
        {"role": "assistant", "content": "a1"},
    ]
    session.staged_boundary = 0

    ctx, mock_loop = _ctx(session)
    response = await cmd_forget(ctx)

    assert len(session.messages) == 0
    assert session.staged_boundary == 0
    assert "Forgot 2 message(s)" in response.content
    mock_loop.sessions.save.assert_called_with(session)


async def test_cmd_forget_already_at_pinned_context():
    session = Session(key="test:forget_at_ctx")
    session.messages = [
        {
            "role": "user",
            "content": "[Previous session context]: doing X",
            "_session_context": True,
            "_compact_summary": True,
        }
    ]
    session.staged_boundary = 1

    ctx, mock_loop = _ctx(session)
    response = await cmd_forget(ctx)

    # Idempotent: nothing after ctx_end to forget.
    assert len(session.messages) == 1
    assert session.staged_boundary == 1
    assert "Forgot 0 message(s)" in response.content
    mock_loop.sessions.save.assert_called_with(session)


@pytest.mark.asyncio
async def test_cmd_forget_acquires_consolidator_lock():
    """F1: /forget waits on the consolidator session lock (no lost updates
    while background/token-budget consolidation is mid-flight)."""
    session = Session(key="test:forget_lock")
    for i in range(4):
        session.add_message("user", f"m{i}")
    session.staged_boundary = 2

    ctx, mock_loop = _ctx(session)
    lock = mock_loop.memory_consolidator.get_lock(session.key)
    await lock.acquire()
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(cmd_forget(ctx), 0.1)
    finally:
        lock.release()

    # Untouched — the command blocked on the lock, never mutated the session.
    assert len(session.messages) == 4
    assert session.staged_boundary == 2
