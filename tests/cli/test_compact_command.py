"""Tests for /compact slash command (unified dispatch, D4)."""

import asyncio
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.memory import MemoryConsolidator
from nanobot.command.builtin import cmd_compact
from nanobot.session.manager import Session

# Shared scratch workspace for the real consolidators created per test.
# (The LLM seam is mocked, so MemoryStore never touches these files.)
_TMPDIR = tempfile.mkdtemp(prefix="nanobot-cmd-compact-")


def _ctx(
    session,
    raw,
    archive_result="summary text",
    session_ctx=None,
    consolidator_context: int = 10_000_000,
):
    mock_loop = MagicMock()
    mock_loop.sessions = MagicMock()
    mock_loop.sessions.get_or_create.return_value = session
    # REAL consolidator: shared dispatch + lock + 80% guard are exercised
    # end-to-end; only the LLM seam (consolidate_messages) is mocked. Its
    # session store IS mock_loop.sessions, so save() assertions keep working.
    provider = MagicMock()
    provider.generation = None
    consolidator = MemoryConsolidator(
        workspace=__import__("pathlib").Path(_TMPDIR),
        provider=provider,
        model="test-model",
        sessions=mock_loop.sessions,
        context_window_tokens=consolidator_context,
        build_messages=lambda *args, **kwargs: [],
        get_tool_definitions=lambda: [],
        max_completion_tokens=128,
    )
    # Real MemoryConsolidator always has _last_session_context (None initially);
    # set it explicitly so MagicMock auto-attribution can't fake a truthy context.
    consolidator._last_session_context = session_ctx
    consolidator.consolidate_messages = AsyncMock(return_value=archive_result)
    mock_loop.memory_consolidator = consolidator
    ctx = MagicMock()
    ctx.loop = mock_loop
    ctx.session = session
    ctx.key = session.key
    ctx.msg = MagicMock(address=MagicMock(channel="tg", sender_id="1"))
    ctx.raw = raw
    return ctx, mock_loop


def _session(key, n, sb=0):
    s = Session(key=key)
    for i in range(n):
        s.add_message("user", f"msg{i}")
    s.staged_boundary = sb
    return s


class TestCompactNothingToSummarize:
    async def test_n_keep_exceeds_live_count(self):
        """n_keep >= len - ctx_end -> empty chunk -> nothing to summarize."""
        session = _session("t:1", 5)
        ctx, mock_loop = _ctx(session, "/compact 10")

        resp = await cmd_compact(ctx)

        assert "Nothing to summarize" in resp.content
        mock_loop.memory_consolidator.consolidate_messages.assert_not_awaited()
        assert len(session.messages) == 5

    async def test_fully_tagged_session(self):
        """All messages are pinned context (ctx_end == len) -> empty chunk."""
        session = Session(key="t:2")
        session.messages = [
            {
                "role": "user",
                "content": "[Previous session context]: done",
                "_session_context": True,
                "_compact_summary": True,
            }
        ]
        session.staged_boundary = 1
        ctx, mock_loop = _ctx(session, "/compact")

        resp = await cmd_compact(ctx)

        assert "Nothing to summarize" in resp.content
        mock_loop.memory_consolidator.consolidate_messages.assert_not_awaited()
        assert len(session.messages) == 1


class TestCompactFreeDrop:
    async def test_free_drop_when_range_already_staged(self):
        """end_idx <= SB: physically removed, SB shifted, NO LLM call (free drop)."""
        session = _session("t:3", 10, sb=8)  # [0:8] staged
        ctx, mock_loop = _ctx(session, "/compact 3")  # end_idx = 7 <= 8

        resp = await cmd_compact(ctx)

        assert "Dropped 7 already-staged" in resp.content
        mock_loop.memory_consolidator.consolidate_messages.assert_not_awaited()
        assert len(session.messages) == 3
        assert [m["content"] for m in session.messages] == ["msg7", "msg8", "msg9"]
        assert session.staged_boundary == 8 - 7  # ctx_end(0) + (SB - end_idx)
        mock_loop.sessions.save.assert_called_with(session)

    async def test_free_drop_with_pinned_prefix(self):
        """ctx_end > 0: the pinned prefix is kept, SB shifts relative to ctx_end."""
        session = Session(key="t:4")
        session.messages = [
            {
                "role": "user",
                "content": "[Previous session context]: old",
                "_session_context": True,
                "_compact_summary": True,
            }
        ]
        for i in range(9):
            session.add_message("user", f"msg{i}")
        session.staged_boundary = 7  # staged region covers msg0..msg5

        ctx, mock_loop = _ctx(session, "/compact 3")
        # len=10, ctx_end=1, end_idx=7 <= SB=7 -> free drop of messages[1:7]

        resp = await cmd_compact(ctx)

        assert "Dropped 6 already-staged" in resp.content
        mock_loop.memory_consolidator.consolidate_messages.assert_not_awaited()
        assert session.messages[0]["_session_context"] is True  # prefix kept
        assert [m["content"] for m in session.messages[1:]] == ["msg6", "msg7", "msg8"]
        assert session.staged_boundary == 1 + (7 - 7)  # ctx_end + (SB - end_idx)


class TestCompactCommitDrop:
    async def test_commit_drop_distills_only_fresh_part(self):
        """end_idx > SB: archive is awaited ONCE with exactly messages[SB:end_idx]."""
        session = _session("t:5", 10, sb=4)
        ctx, mock_loop = _ctx(session, "/compact 3")  # end_idx = 7 > SB=4

        resp = await cmd_compact(ctx)

        assert "Consolidated history, keeping last 3 messages" in resp.content
        archive = mock_loop.memory_consolidator.consolidate_messages
        archive.assert_awaited_once()
        chunk = archive.call_args[0][0]
        assert [m["content"] for m in chunk] == ["msg4", "msg5", "msg6"]  # [SB:end_idx]
        # [ctx_end:end_idx] removed; no context injection (no _last_session_context)
        assert [m["content"] for m in session.messages] == ["msg7", "msg8", "msg9"]
        assert session.staged_boundary == 0
        mock_loop.sessions.save.assert_called_with(session)

    async def test_commit_drop_injects_dual_tagged_context(self):
        """With _last_session_context set, a dual-tagged msg is inserted at ctx_end, SB+1."""
        session = _session("t:6", 10, sb=4)
        ctx, mock_loop = _ctx(
            session, "/compact", session_ctx="Working on the consolidation cutover."
        )

        resp = await cmd_compact(ctx)

        assert "Consolidated all history" in resp.content
        injected = session.messages[0]
        assert injected["role"] == "user"
        assert injected["content"] == (
            "[Previous session context]: Working on the consolidation cutover."
        )
        assert injected["_session_context"] is True
        assert injected["_compact_summary"] is True
        assert len(session.messages) == 1  # all originals dropped, only context remains
        assert session.staged_boundary == 1

    async def test_commit_drop_failure_drops_nothing(self):
        """archive_messages -> None: nothing dropped, SB unchanged, no save."""
        session = _session("t:7", 10, sb=4)
        ctx, mock_loop = _ctx(session, "/compact", archive_result=None)

        resp = await cmd_compact(ctx)

        assert "Compact failed" in resp.content
        assert len(session.messages) == 10
        assert session.staged_boundary == 4
        mock_loop.sessions.save.assert_not_called()

    async def test_staged_boundary_equivalent_replaces_last_consolidated(self):
        """Staged-boundary equivalent of the old 'respects last_consolidated' test:
        /compact 5 on 20 msgs with SB=10 distills only the fresh part [10:15]."""
        session = _session("t:8", 20, sb=10)
        ctx, mock_loop = _ctx(session, "/compact 5")  # end_idx = 15 > SB=10

        resp = await cmd_compact(ctx)

        assert "Consolidated history, keeping last 5 messages" in resp.content
        archive = mock_loop.memory_consolidator.consolidate_messages
        archive.assert_awaited_once()
        chunk = archive.call_args[0][0]
        assert [m["content"] for m in chunk] == ["msg10", "msg11", "msg12", "msg13", "msg14"]
        # [0:15] dropped (staged region lossless), 5 kept live
        assert [m["content"] for m in session.messages] == [
            "msg15",
            "msg16",
            "msg17",
            "msg18",
            "msg19",
        ]
        assert session.staged_boundary == 0


class TestCompactArgumentParsing:
    async def test_negative_n_uses_absolute_value(self):
        session = _session("t:9", 10)
        ctx, mock_loop = _ctx(session, "/compact -3")

        resp = await cmd_compact(ctx)

        assert "keeping last 3 messages" in resp.content
        archive = mock_loop.memory_consolidator.consolidate_messages
        archive.assert_awaited_once()
        assert len(archive.call_args[0][0]) == 7

    async def test_invalid_n_falls_back_to_zero(self):
        session = _session("t:10", 10)
        ctx, mock_loop = _ctx(session, "/compact abc")

        resp = await cmd_compact(ctx)

        assert "Consolidated all history" in resp.content
        assert len(mock_loop.memory_consolidator.consolidate_messages.call_args[0][0]) == 10


class TestCompactGuardAndLock:
    """F4: the 80% consolidator-context guard (now in the shared dispatch).
    F1: /compact acquires the consolidator session lock."""

    @pytest.mark.asyncio
    async def test_cmd_compact_rejects_chunk_over_80pct_of_consolidator_context(self, monkeypatch):
        import nanobot.agent.memory as memory_module

        monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _m: 100)
        # 10 msgs × 100 = 1000 fresh tokens > 500 × 0.8 = 400
        session = _session("t:guard", 10)
        ctx, mock_loop = _ctx(session, "/compact", consolidator_context=500)

        resp = await cmd_compact(ctx)

        assert "exceed the summarizer's context window" in resp.content
        mock_loop.memory_consolidator.consolidate_messages.assert_not_awaited()
        # Nothing dropped
        assert len(session.messages) == 10
        assert session.staged_boundary == 0
        mock_loop.sessions.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_cmd_compact_acquires_consolidator_lock(self):
        session = _session("t:lock", 10)
        ctx, mock_loop = _ctx(session, "/compact")

        lock = mock_loop.memory_consolidator.get_lock(session.key)
        await lock.acquire()
        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(cmd_compact(ctx), 0.1)
        finally:
            lock.release()

        assert len(session.messages) == 10
