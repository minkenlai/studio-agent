"""Tests for the /stage command (D5) and its boundary picker."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.memory import MemoryConsolidator
from nanobot.command.builtin import _pick_stage_boundary, cmd_stage
from nanobot.session.manager import Session


def _tc(id_):
    return {"id": id_, "type": "function", "function": {"name": "f", "arguments": "{}"}}


def _ctx(session, raw, archive_result="summary text"):
    mock_loop = MagicMock()
    mock_loop.sessions = MagicMock()
    mock_loop.sessions.get_or_create.return_value = session
    # REAL consolidator so the session lock is a genuine asyncio.Lock (F1);
    # archive_messages (real) funnels to the mocked consolidate_messages seam.
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
    consolidator.consolidate_messages = AsyncMock(return_value=archive_result)
    mock_loop.memory_consolidator = consolidator
    ctx = MagicMock()
    ctx.loop = mock_loop
    ctx.session = session
    ctx.key = session.key
    ctx.msg = MagicMock(address=MagicMock(channel="tg", sender_id="1"))
    ctx.raw = raw
    return ctx, mock_loop


class TestPickStageBoundary:
    """_pick_stage_boundary must never let SB land mid tool-call exchange (D5)."""

    def test_user_turn_at_or_below_desired(self):
        msgs = [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a3"},
        ]
        assert _pick_stage_boundary(msgs, 0, 2) == 2
        # desired above the last user turn -> clamp down to it
        assert _pick_stage_boundary(msgs, 0, 4) == 2

    def test_skips_unbalanced_user_turn_clamps_to_legal(self):
        msgs = [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a1", "tool_calls": [_tc("t1")]},
            {"role": "user", "content": "u*"},  # mid exchange — unbalanced
            {"role": "tool", "content": "r1", "tool_call_id": "t1"},
        ]
        assert _pick_stage_boundary(msgs, 0, 2) == 1  # largest legal cut below

    def test_open_tail_clamps_back(self):
        msgs = [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a1", "tool_calls": [_tc("t9")]},  # open
        ]
        assert _pick_stage_boundary(msgs, 0, 2) == 1

    def test_desired_beyond_len_clamps_to_total(self):
        msgs = [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a1"},
        ]
        # desired clamped to total=2; no user turn in (0,2); backward clamp to 1
        assert _pick_stage_boundary(msgs, 0, 99) == 1
        # user turn at the end is found directly
        msgs2 = [
            {"role": "assistant", "content": "a0"},
            {"role": "user", "content": "u1"},
        ]
        assert _pick_stage_boundary(msgs2, 0, 99) == 1

    def test_desired_below_sb(self):
        msgs = [{"role": "user", "content": "u"}]
        assert _pick_stage_boundary(msgs, 1, 0) == 1

    def test_no_legal_cut_returns_none(self):
        # SB itself unbalanced (defensive: invariant broken upstream)
        msgs = [{"role": "assistant", "content": "a", "tool_calls": [_tc("t1")]}]
        assert _pick_stage_boundary(msgs, 1, 2) is None

    def test_multiple_tool_calls_same_message(self):
        msgs = [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a", "tool_calls": [_tc("t1"), _tc("t2")]},
            {"role": "tool", "content": "r1", "tool_call_id": "t1"},
            {"role": "tool", "content": "r2", "tool_call_id": "t2"},
            {"role": "user", "content": "u4"},
        ]
        # desired=4: u4 balanced (t1,t2 both resolved) -> 4
        assert _pick_stage_boundary(msgs, 0, 4) == 4
        # desired=3: mid exchange (t2 open) -> no user turn; clamp to 1
        assert _pick_stage_boundary(msgs, 0, 3) == 1
        # desired past the end: clamped to 6; no user turn in (0,6) except u4
        # -> backward scan finds u4 at 4
        msgs2 = msgs + [{"role": "assistant", "content": "a5"}]
        assert _pick_stage_boundary(msgs2, 0, 6) == 4


class TestCmdStage:
    async def test_stage_all_stages_up_to_legal_boundary(self):
        session = Session(key="t:1")
        session.messages = [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "u4"},
        ]
        session.staged_boundary = 0
        ctx, mock_loop = _ctx(session, "/stage")

        resp = await cmd_stage(ctx)

        assert session.staged_boundary == 4  # user turn; last msg kept live
        assert len(session.messages) == 5  # nothing dropped
        chunk = mock_loop.memory_consolidator.consolidate_messages.call_args[0][0]
        assert len(chunk) == 4 and chunk[0]["content"] == "u0"
        assert "Staged 4 message(s)" in resp.content
        assert "Nothing dropped" in resp.content
        mock_loop.sessions.save.assert_called_with(session)

    async def test_stage_keeps_n_live(self):
        session = Session(key="t:2")
        session.messages = [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "u4"},
            {"role": "assistant", "content": "a5"},
        ]
        session.staged_boundary = 0
        ctx, _ = _ctx(session, "/stage 2")

        resp = await cmd_stage(ctx)

        assert session.staged_boundary == 4  # len - 2 lands on user turn
        assert len(session.messages) == 6
        assert "clamped" not in resp.content

    async def test_nothing_to_stage(self):
        session = Session(key="t:3")
        session.messages = [{"role": "user", "content": "u0"}]
        session.staged_boundary = 1
        ctx, mock_loop = _ctx(session, "/stage")

        resp = await cmd_stage(ctx)

        assert "Nothing to stage" in resp.content
        mock_loop.memory_consolidator.consolidate_messages.assert_not_called()

    async def test_failure_does_not_advance_sb(self):
        session = Session(key="t:4")
        session.messages = [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a1"},
        ]
        session.staged_boundary = 0
        ctx, mock_loop = _ctx(session, "/stage", archive_result=None)

        resp = await cmd_stage(ctx)

        assert session.staged_boundary == 0
        assert len(session.messages) == 2
        assert "Stage failed" in resp.content
        mock_loop.sessions.save.assert_not_called()

    async def test_pinned_ctx_never_staged(self):
        session = Session(key="t:5")
        session.messages = [
            {
                "role": "user",
                "content": "[ctx]",
                "_session_context": True,
                "_compact_summary": True,
            },
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
        session.staged_boundary = 1
        ctx, mock_loop = _ctx(session, "/stage")

        await cmd_stage(ctx)

        # No user turn in (sb, desired); clamp stages [u1] only, keeps a1 live.
        assert session.staged_boundary == 2
        assert session.messages[0]["_session_context"] is True
        chunk = mock_loop.memory_consolidator.consolidate_messages.call_args[0][0]
        assert [m["content"] for m in chunk] == ["u1"]


@pytest.mark.asyncio
async def test_cmd_stage_acquires_consolidator_lock():
    """F1: /stage waits on the consolidator session lock for its distillation
    + SB-advance (the boundary pick itself is read-only, outside the lock)."""
    session = Session(key="test:stage_lock")
    for i in range(4):
        session.add_message("user", f"u{i}")
    session.staged_boundary = 0

    ctx, mock_loop = _ctx(session, "/stage")
    lock = mock_loop.memory_consolidator.get_lock(session.key)
    await lock.acquire()
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(cmd_stage(ctx), 0.1)
    finally:
        lock.release()

    # Untouched — the command blocked before distilling or advancing SB.
    assert len(session.messages) == 4
    assert session.staged_boundary == 0
