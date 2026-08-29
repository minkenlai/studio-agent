"""Tests for CompactHistoryTool — session history consolidation via tool call.

Rewritten for the unified dispatch (D4):
- free drop      — end_idx <= SB: physical delete, SB shifted, NO LLM call
- commit + drop  — end_idx > SB: distill fresh [SB:end_idx], then drop [ctx_end:end_idx]
- 80% guard      — fresh chunk larger than 80% of the consolidator's context window
- injection      — dual-tagged user message only when _last_session_context is set
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import nanobot.agent.memory as memory_module
from nanobot.agent.memory import MemoryConsolidator
from nanobot.agent.tools.compact import CompactHistoryTool
from nanobot.bus.events import Address
from nanobot.session.manager import Session, SessionManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(key: str = "test:compact", message_pairs: int = 5) -> Session:
    """Build a session with `message_pairs` user/assistant exchanges."""
    s = Session(key=key)
    for i in range(message_pairs):
        s.add_message("user", f"user message {i}")
        s.add_message("assistant", f"assistant reply {i}")
    return s


def _make_tool(
    tmp_path: Path,
    session_manager: SessionManager,
    summary_return: str | None = "Summary of conversation so far.",
    session_ctx: str | None = None,
    consolidator_context: int | None = None,
) -> CompactHistoryTool:
    """Create a CompactHistoryTool backed by a REAL MemoryConsolidator (shared
    dispatch + lock + 80% guard) with only the LLM seam (consolidate_messages)
    mocked.  context_window_tokens is set explicitly so MagicMock
    auto-attribution can't fake an int window."""
    provider = MagicMock()
    provider.generation = None
    consolidator = MemoryConsolidator(
        workspace=tmp_path,
        provider=provider,
        model="test-model",
        sessions=session_manager,
        context_window_tokens=consolidator_context or 10_000_000,
        build_messages=lambda *args, **kwargs: [],
        get_tool_definitions=lambda: [],
        max_completion_tokens=128,
    )
    consolidator._last_session_context = session_ctx
    consolidator.consolidate_messages = AsyncMock(return_value=summary_return)
    return CompactHistoryTool(consolidator, session_manager)


def _ctx_tagged(content: str) -> dict:
    return {
        "role": "user",
        "content": content,
        "_session_context": True,
        "_compact_summary": True,
        "timestamp": "2026-01-01T00:00:00",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompactHistoryBasic:
    """Core functionality of CompactHistoryTool.execute (commit + drop path)."""

    @pytest.mark.asyncio
    async def test_n_keep_zero_consolidates_all(self, tmp_path: Path) -> None:
        """n_keep=0 with SB=0 distills everything and drops it all (no ctx set)."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=5)  # 10 messages
        sm.save(session)

        tool = _make_tool(tmp_path, sm)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        result = await tool.execute(n_keep=0)

        # archive was called once with ALL 10 messages (fresh = [SB=0 : end_idx=10])
        tool._consolidator.consolidate_messages.assert_awaited_once()
        chunk = tool._consolidator.consolidate_messages.call_args[0][0]
        assert len(chunk) == 10
        assert chunk[0]["content"] == "user message 0"
        assert chunk[-1]["content"] == "assistant reply 4"

        # [0:10] dropped; no session context → no injection; SB reset to ctx_end (0)
        reloaded = sm.get_or_create("test:compact")
        assert reloaded.messages == []
        assert reloaded.staged_boundary == 0

        assert "Consolidated all history" in result

    @pytest.mark.asyncio
    async def test_n_keep_partial(self, tmp_path: Path) -> None:
        """n_keep=2 distills [0:8] and keeps the last 2 messages live."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=5)  # 10 messages
        sm.save(session)

        tool = _make_tool(tmp_path, sm)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        result = await tool.execute(n_keep=2)

        chunk = tool._consolidator.consolidate_messages.call_args[0][0]
        # end_idx = 10 - 2 = 8, fresh = messages[SB=0 : 8]
        assert len(chunk) == 8
        assert chunk[-1]["content"] == "assistant reply 3"

        reloaded = sm.get_or_create("test:compact")
        # [0:8] dropped, last 2 kept, no injection
        assert [m["content"] for m in reloaded.messages] == ["user message 4", "assistant reply 4"]
        assert reloaded.staged_boundary == 0

        assert "keeping last 2 messages" in result

    @pytest.mark.asyncio
    async def test_n_keep_larger_than_history(self, tmp_path: Path) -> None:
        """n_keep > len(messages) should be a no-op."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=2)  # 4 messages
        sm.save(session)

        tool = _make_tool(tmp_path, sm)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        result = await tool.execute(n_keep=100)

        # end_idx = max(0, 4 - 100) = 0, chunk = messages[0:0] = []
        tool._consolidator.consolidate_messages.assert_not_awaited()
        assert "Nothing to summarize" in result

    @pytest.mark.asyncio
    async def test_empty_session(self, tmp_path: Path) -> None:
        """An empty session should return 'Nothing to summarize'."""
        sm = SessionManager(tmp_path)
        session = Session(key="test:empty")
        sm.save(session)

        tool = _make_tool(tmp_path, sm)
        tool.set_context(Address.from_uri("test:empty"), "test:empty")

        result = await tool.execute(n_keep=0)

        tool._consolidator.consolidate_messages.assert_not_awaited()
        assert "Nothing to summarize" in result

    @pytest.mark.asyncio
    async def test_default_n_keep_is_zero(self, tmp_path: Path) -> None:
        """Calling without n_keep should default to 0 (consolidate all)."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=3)
        sm.save(session)

        tool = _make_tool(tmp_path, sm)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        await tool.execute()

        chunk = tool._consolidator.consolidate_messages.call_args[0][0]
        assert len(chunk) == 6  # all messages

    @pytest.mark.asyncio
    async def test_fully_tagged_session_nothing_to_summarize(self, tmp_path: Path) -> None:
        """ctx_end == len (all pinned context) → empty chunk → nothing to summarize."""
        sm = SessionManager(tmp_path)
        session = Session(key="test:tagged")
        session.messages = [_ctx_tagged("[Previous session context]: done"), _ctx_tagged("b")]
        session.staged_boundary = 2
        sm.save(session)

        tool = _make_tool(tmp_path, sm)
        tool.set_context(Address.from_uri("test:tagged"), "test:tagged")

        result = await tool.execute(n_keep=0)

        tool._consolidator.consolidate_messages.assert_not_awaited()
        assert "Nothing to summarize" in result
        assert len(sm.get_or_create("test:tagged").messages) == 2


class TestCompactHistoryFreeDrop:
    """end_idx <= SB: reclaim already-staged messages without any LLM call."""

    @pytest.mark.asyncio
    async def test_free_drop_no_llm_call(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=5)  # 10 messages
        session.staged_boundary = 8  # [0:8] already staged
        sm.save(session)

        tool = _make_tool(tmp_path, sm)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        result = await tool.execute(n_keep=3)  # end_idx = 7 <= SB=8

        tool._consolidator.consolidate_messages.assert_not_awaited()
        reloaded = sm.get_or_create("test:compact")
        # [0:7] dropped → last 3 messages survive
        assert [m["content"] for m in reloaded.messages] == [
            "assistant reply 3",
            "user message 4",
            "assistant reply 4",
        ]
        # SB' = ctx_end + (SB - end_idx) = 0 + (8 - 7) = 1
        assert reloaded.staged_boundary == 1
        assert "Dropped 7 already-staged" in result

    @pytest.mark.asyncio
    async def test_free_drop_with_pinned_prefix(self, tmp_path: Path) -> None:
        """ctx_end > 0: the pinned prefix is kept, SB shifts relative to ctx_end."""
        sm = SessionManager(tmp_path)
        session = Session(key="test:pref")
        session.messages = [_ctx_tagged("[Previous session context]: old")]
        for i in range(9):
            session.add_message("user", f"user message {i}")
            session.add_message("assistant", f"assistant reply {i}")
        session.staged_boundary = 7  # staged region: user0..assistant2 (6 msgs)
        sm.save(session)

        tool = _make_tool(tmp_path, sm)
        tool.set_context(Address.from_uri("test:pref"), "test:pref")
        # len=19, ctx_end=1, n_keep=12 → end_idx=7 <= SB=7 → free drop of [1:7]
        result = await tool.execute(n_keep=12)

        assert "Dropped 6 already-staged" in result
        tool._consolidator.consolidate_messages.assert_not_awaited()
        reloaded = sm.get_or_create("test:pref")
        assert reloaded.messages[0]["_session_context"] is True  # prefix kept
        # [1:7] (user0..assistant2) dropped → user3..assistant8 remain
        assert [m["content"] for m in reloaded.messages[1:]] == [
            "user message 3",
            "assistant reply 3",
            "user message 4",
            "assistant reply 4",
            "user message 5",
            "assistant reply 5",
            "user message 6",
            "assistant reply 6",
            "user message 7",
            "assistant reply 7",
            "user message 8",
            "assistant reply 8",
        ]
        # SB' = ctx_end + (SB - end_idx) = 1 + 0 = 1
        assert reloaded.staged_boundary == 1


class TestCompactHistorySummaryInjection:
    """Dual-tagged session-context injection on commit + drop."""

    @pytest.mark.asyncio
    async def test_context_injected_into_session_messages(self, tmp_path: Path) -> None:
        """With _last_session_context set, a dual-tagged user msg is inserted at ctx_end."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=3)  # 6 messages
        sm.save(session)

        custom_ctx = "Key topics: pricing, architecture, deployment."
        tool = _make_tool(tmp_path, sm, session_ctx=custom_ctx)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        await tool.execute(n_keep=2)

        reloaded = sm.get_or_create("test:compact")
        # [0:4] dropped; injected ctx at 0; kept msgs follow
        assert len(reloaded.messages) == 3
        injected = reloaded.messages[0]
        assert injected["role"] == "user"
        assert injected["content"] == f"[Previous session context]: {custom_ctx}"
        assert injected["_session_context"] is True
        assert injected["_compact_summary"] is True
        assert "timestamp" in injected
        assert [m["content"] for m in reloaded.messages[1:]] == [
            "user message 2",
            "assistant reply 2",
        ]
        assert reloaded.staged_boundary == 1  # ctx_end + 1

    @pytest.mark.asyncio
    async def test_summary_included_in_tool_result(self, tmp_path: Path) -> None:
        """The tool result string should contain the generated summary."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=2)
        sm.save(session)

        summary_text = "Discussed API design and database schema."
        tool = _make_tool(tmp_path, sm, summary_return=summary_text)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        result = await tool.execute(n_keep=0)

        assert summary_text in result
        assert "---" in result  # separator before summary

    @pytest.mark.asyncio
    async def test_no_injection_when_context_none(self, tmp_path: Path) -> None:
        """Commit + drop without _last_session_context: no injected message."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=3)  # 6 messages
        sm.save(session)

        tool = _make_tool(tmp_path, sm)  # session_ctx defaults to None
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        await tool.execute(n_keep=2)

        reloaded = sm.get_or_create("test:compact")
        assert [m["content"] for m in reloaded.messages] == [
            "user message 2",
            "assistant reply 2",
        ]
        assert reloaded.staged_boundary == 0

    @pytest.mark.asyncio
    async def test_staged_boundary_resets_to_ctx_end(self, tmp_path: Path) -> None:
        """Commit + drop: only fresh [SB:end_idx] is distilled; SB resets to ctx_end."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=10)  # 20 messages
        session.staged_boundary = 4
        sm.save(session)

        tool = _make_tool(tmp_path, sm)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        await tool.execute(n_keep=4)

        chunk = tool._consolidator.consolidate_messages.call_args[0][0]
        # fresh = [SB=4 : end_idx=16] = 12 messages (staged part NOT re-archived)
        assert len(chunk) == 12
        assert chunk[0]["content"] == "user message 2"
        assert chunk[-1]["content"] == "assistant reply 7"

        reloaded = sm.get_or_create("test:compact")
        # [0:16] dropped (staged losslessly + fresh just distilled), 4 kept, SB=0
        assert [m["content"] for m in reloaded.messages] == [
            "user message 8",
            "assistant reply 8",
            "user message 9",
            "assistant reply 9",
        ]
        assert reloaded.staged_boundary == 0

    @pytest.mark.asyncio
    async def test_get_history_after_compact(self, tmp_path: Path) -> None:
        """get_history should return only the kept messages after compacting."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=5)  # 10 messages
        sm.save(session)

        tool = _make_tool(tmp_path, sm)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        await tool.execute(n_keep=4)

        reloaded = sm.get_or_create("test:compact")
        history = reloaded.get_history(max_messages=0)
        # [0:6] dropped, last 4 kept
        assert len(history) == 4
        assert history[0]["content"] == "user message 3"
        assert history[-1]["content"] == "assistant reply 4"


class TestCompactHistoryContextGuard:
    """80% consolidator-context early guard."""

    @pytest.mark.asyncio
    async def test_chunk_exceeding_80_percent_context_is_rejected(
        self, tmp_path, monkeypatch
    ) -> None:
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=5)  # 10 messages, SB=0
        sm.save(session)

        monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _m: 100)
        tool = _make_tool(tmp_path, sm, consolidator_context=500)  # 80% = 400 tokens
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        result = await tool.execute(n_keep=0)  # fresh = 10 msgs × 100 = 1000 tokens

        tool._consolidator.consolidate_messages.assert_not_awaited()
        assert "exceed the summarizer's context window" in result
        # Nothing dropped
        reloaded = sm.get_or_create("test:compact")
        assert len(reloaded.messages) == 10
        assert reloaded.staged_boundary == 0

    @pytest.mark.asyncio
    async def test_chunk_within_context_is_allowed(self, tmp_path, monkeypatch) -> None:
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=5)  # 10 messages, SB=0
        sm.save(session)

        monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _m: 10)
        tool = _make_tool(tmp_path, sm, consolidator_context=500)  # 10 × 10 = 100 < 400
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        result = await tool.execute(n_keep=0)

        tool._consolidator.consolidate_messages.assert_awaited_once()
        assert "Consolidated all history" in result


class TestCompactHistoryArchiveFailure:
    """Behavior when archive_messages returns None (consolidation failure)."""

    @pytest.mark.asyncio
    async def test_no_summary_on_archive_failure(self, tmp_path: Path) -> None:
        """When archive_messages returns None, no summary should be injected."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=3)
        sm.save(session)

        tool = _make_tool(tmp_path, sm, summary_return=None)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        result = await tool.execute(n_keep=0)

        reloaded = sm.get_or_create("test:compact")
        # All messages should still be original user/assistant pairs
        assert all(
            m["content"].startswith("user") or m["content"].startswith("assistant reply")
            for m in reloaded.messages
        )
        assert "Summary of conversation" not in result

    @pytest.mark.asyncio
    async def test_staged_boundary_not_advanced_on_failure(self, tmp_path: Path) -> None:
        """
        When archive_messages returns None (consolidation failure),
        SB must NOT change — messages stay in session to prevent data loss.
        """
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=3)
        session.staged_boundary = 2
        sm.save(session)

        tool = _make_tool(tmp_path, sm, summary_return=None)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        result = await tool.execute(n_keep=0)

        reloaded = sm.get_or_create("test:compact")
        # SB unchanged — messages preserved
        assert reloaded.staged_boundary == 2
        assert len(reloaded.messages) == 6  # all original messages intact
        assert "failed" in result.lower()


class TestCompactHistoryRepeatedCalls:
    """Multiple consecutive compact_history calls."""

    @pytest.mark.asyncio
    async def test_second_call_noop_when_fully_compacted(self, tmp_path: Path) -> None:
        """After compaction to an empty session, a second call finds nothing to do."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=5)
        sm.save(session)

        tool = _make_tool(tmp_path, sm)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        # First call: consolidate all (commit + drop, no ctx → empty session)
        result1 = await tool.execute(n_keep=0)
        assert "Consolidated all history" in result1

        # Second call: empty session → nothing to summarize, no LLM call
        tool._consolidator.consolidate_messages.reset_mock()
        result2 = await tool.execute(n_keep=0)

        assert "Nothing to summarize" in result2
        tool._consolidator.consolidate_messages.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_compact_then_add_messages_then_compact(self, tmp_path: Path) -> None:
        """After compacting, adding new messages should be compactable."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=5)  # 10 messages, SB=0
        sm.save(session)

        tool = _make_tool(tmp_path, sm)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        # First compact: keep last 2 → [0:8] dropped, 2 kept
        await tool.execute(n_keep=2)
        assert [m["content"] for m in sm.get_or_create("test:compact").messages] == [
            "user message 4",
            "assistant reply 4",
        ]

        # Add new messages
        reloaded = sm.get_or_create("test:compact")
        reloaded.add_message("user", "new question")
        reloaded.add_message("assistant", "new answer")
        sm.save(reloaded)

        # Second compact: end_idx = 4 - 2 = 2 > SB=0 → commit + drop fresh [0:2]
        tool._consolidator.consolidate_messages.reset_mock()
        await tool.execute(n_keep=2)

        chunk = tool._consolidator.consolidate_messages.call_args[0][0]
        assert [m["content"] for m in chunk] == ["user message 4", "assistant reply 4"]
        # The new messages survive
        assert [m["content"] for m in sm.get_or_create("test:compact").messages] == [
            "new question",
            "new answer",
        ]

    @pytest.mark.asyncio
    async def test_incremental_compaction_with_staged_region(self, tmp_path: Path) -> None:
        """Commit + drop with an existing staged boundary distills only [SB:end_idx]."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=10)  # 20 messages
        session.staged_boundary = 8  # [0:8] already staged
        sm.save(session)

        tool = _make_tool(tmp_path, sm)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        await tool.execute(n_keep=4)  # end_idx = 16 > SB=8

        chunk = tool._consolidator.consolidate_messages.call_args[0][0]
        # fresh = [SB=8 : end_idx=16] = 8 messages
        assert len(chunk) == 8
        assert chunk[0]["content"] == "user message 4"
        assert chunk[-1]["content"] == "assistant reply 7"

        # [0:16] dropped (staged losslessly), 4 kept, SB reset to 0
        reloaded = sm.get_or_create("test:compact")
        assert [m["content"] for m in reloaded.messages] == [
            "user message 8",
            "assistant reply 8",
            "user message 9",
            "assistant reply 9",
        ]
        assert reloaded.staged_boundary == 0


class TestCompactHistoryToolInterface:
    """Test tool schema and interface compliance."""

    def test_tool_name(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path)
        tool = _make_tool(tmp_path, sm)
        assert tool.name == "compact_history"

    def test_tool_description(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path)
        tool = _make_tool(tmp_path, sm)
        assert "consolidat" in tool.description.lower()

    def test_tool_parameters_schema(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path)
        tool = _make_tool(tmp_path, sm)
        params = tool.parameters
        assert params["type"] == "object"
        assert "n_keep" in params["properties"]
        assert params["properties"]["n_keep"]["type"] == "integer"
        assert params["properties"]["n_keep"]["minimum"] == 0
        assert params["required"] == []

    def test_set_context_updates_session_key(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path)
        tool = _make_tool(tmp_path, sm)
        assert tool._session_key == "cli:direct"  # default
        tool.set_context(Address.from_uri("slack:general"), "slack:general")
        assert tool._session_key == "slack:general"

    def test_to_schema(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path)
        tool = _make_tool(tmp_path, sm)
        schema = tool.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "compact_history"
