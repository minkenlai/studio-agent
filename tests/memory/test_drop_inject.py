"""Tests for session context injection after consolidation (drop-and-inject logic).

Rewritten for the unified dispatch: injection happens ONLY on commit + drop,
produces a NEW dual-tagged user message at ctx_end, and bumps SB by 1.
Free drops inject nothing.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.memory import MemoryConsolidator
from nanobot.providers.base import LLMClient
from nanobot.session.manager import Session, SessionManager


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "memory").mkdir()
    (ws / "sessions").mkdir()
    return ws


@pytest.fixture
def mock_provider() -> MagicMock:
    provider = MagicMock(spec=LLMClient)
    provider.generation = MagicMock()
    provider.generation.context_max = None
    return provider


@pytest.fixture
def sessions(workspace: Path) -> SessionManager:
    return SessionManager(workspace)


@pytest.fixture
def make_consolidator(mock_provider: MagicMock, sessions: SessionManager):
    """Factory for creating a MemoryConsolidator."""

    def _factory(context_window_tokens: int = 64000) -> MemoryConsolidator:
        return MemoryConsolidator(
            workspace=sessions.workspace,
            provider=mock_provider,
            model="test-model",
            sessions=sessions,
            context_window_tokens=context_window_tokens,
            build_messages=lambda *args, **kwargs: [],
            get_tool_definitions=lambda: [],
        )

    return _factory


@pytest.fixture
def session() -> Session:
    s = Session(key="test:123")
    s.messages = [
        {"role": "user", "content": "msg0"},
        {"role": "assistant", "content": "reply0"},
        {"role": "user", "content": "msg1"},
        {"role": "assistant", "content": "reply1"},
        {"role": "user", "content": "msg2"},
        {"role": "assistant", "content": "reply2"},
        {"role": "user", "content": "msg3"},
        {"role": "assistant", "content": "reply3"},
    ]
    s.staged_boundary = 0
    return s


class TestDropAndInject:
    """Session context injection after consolidation drops old messages."""

    @pytest.mark.asyncio
    async def test_inject_session_context_after_consolidation(
        self, session: Session, make_consolidator, sessions: SessionManager
    ):
        """Commit + drop whose consolidation returns a session_context injects it.

        The injection is a NEW message (insert at ctx_end), dual-tagged, with
        content prefix '[Previous session context]: '. SB is bumped by 1.
        """
        consolidator = make_consolidator(context_window_tokens=6400)

        estimate_calls = [0]

        def mock_estimate(sess):
            estimate_calls[0] += 1
            return (10000, "mock") if estimate_calls[0] == 1 else (100, "mock")

        with (
            patch.object(consolidator, "estimate_session_prompt_tokens", side_effect=mock_estimate),
            patch.object(
                consolidator,
                "consolidate_messages",
                new=AsyncMock(return_value=("summary", "User is working on memory consolidation")),
            ),
            patch.object(sessions, "save", new=MagicMock()),
        ):
            await consolidator.maybe_consolidate_by_tokens(session, 6400)

        summary_msgs = [m for m in session.messages if m.get("_compact_summary")]
        assert len(summary_msgs) == 1
        injected = summary_msgs[0]
        # Positioned at ctx_end (index 0 here — no pre-existing pinned prefix).
        assert session.messages.index(injected) == 0
        assert injected["role"] == "user"
        assert injected["content"] == (
            "[Previous session context]: User is working on memory consolidation"
        )
        assert injected["_session_context"] is True
        assert injected["_compact_summary"] is True
        # Dropped 6 staged+fresh, kept last 2, injected 1 -> 3 messages; SB = 0 + 1.
        assert len(session.messages) == 3
        assert session.staged_boundary == 1

    @pytest.mark.asyncio
    async def test_no_inject_when_context_is_none(
        self, session: Session, make_consolidator, sessions: SessionManager
    ):
        """No session_context from the consolidation -> no injection, but the drop still happens."""
        consolidator = make_consolidator(context_window_tokens=6400)

        estimate_calls = [0]

        def mock_estimate(sess):
            estimate_calls[0] += 1
            return (10000, "mock") if estimate_calls[0] == 1 else (100, "mock")

        with (
            patch.object(consolidator, "estimate_session_prompt_tokens", side_effect=mock_estimate),
            patch.object(
                consolidator, "consolidate_messages", new=AsyncMock(return_value=("summary", None))
            ),
            patch.object(sessions, "save", new=MagicMock()),
        ):
            await consolidator.maybe_consolidate_by_tokens(session, 6400)

        summary_msgs = [m for m in session.messages if m.get("_compact_summary")]
        assert summary_msgs == []
        # Commit + drop still physically dropped [0:6]; SB stays at ctx_end (0).
        assert [m["content"] for m in session.messages] == ["msg3", "reply3"]
        assert session.staged_boundary == 0

    @pytest.mark.asyncio
    async def test_inject_creates_new_message_not_mutation(
        self, session: Session, make_consolidator, sessions: SessionManager
    ):
        """The injection inserts a new message; existing messages are only shifted."""
        consolidator = make_consolidator(context_window_tokens=6400)

        original_count = len(session.messages)
        original_ids = {id(m) for m in session.messages}

        estimate_calls = [0]

        def mock_estimate(sess):
            estimate_calls[0] += 1
            return (10000, "mock") if estimate_calls[0] == 1 else (100, "mock")

        with (
            patch.object(consolidator, "estimate_session_prompt_tokens", side_effect=mock_estimate),
            patch.object(
                consolidator,
                "consolidate_messages",
                new=AsyncMock(return_value=("summary", "Context data")),
            ),
            patch.object(sessions, "save", new=MagicMock()),
        ):
            await consolidator.maybe_consolidate_by_tokens(session, 6400)

        # 6 dropped, 1 injected -> count is NOT simply original + 1, but every
        # surviving original message is the SAME dict object (no mutation/replace).
        assert len(session.messages) == original_count - 6 + 1
        surviving = [m for m in session.messages if not m.get("_compact_summary")]
        assert all(id(m) in original_ids for m in surviving)
        assert [m["content"] for m in surviving] == ["msg3", "reply3"]
        assert len([m for m in session.messages if m.get("_compact_summary")]) == 1

    @pytest.mark.asyncio
    async def test_free_drop_injects_nothing(
        self, session: Session, make_consolidator, sessions: SessionManager
    ):
        """FREE DROP (boundary within staged region) must not call the LLM and injects nothing."""
        consolidator = make_consolidator(context_window_tokens=6400)
        session.staged_boundary = 6  # [0:6] already staged

        estimate_calls = [0]

        def mock_estimate(sess):
            estimate_calls[0] += 1
            # est == budget exactly: passes the trigger gate; picker target
            # (640) exceeds the real removed tokens, so the picker returns the
            # LAST user boundary (idx 6), which lies in the staged region.
            if estimate_calls[0] == 1:
                return (1280, "mock")
            return (100, "mock")

        mock_consolidate = AsyncMock(return_value=("summary", None))
        with (
            patch.object(consolidator, "estimate_session_prompt_tokens", side_effect=mock_estimate),
            patch.object(consolidator, "consolidate_messages", new=mock_consolidate),
            patch.object(sessions, "save", new=MagicMock()),
        ):
            await consolidator.maybe_consolidate_by_tokens(session, 6400)

        # Picker returns the last user boundary (idx 6, msg3) — inside staged [0:6]
        # -> free drop of [0:6].
        mock_consolidate.assert_not_awaited()
        assert [m for m in session.messages if m.get("_compact_summary")] == []
        assert [m["content"] for m in session.messages] == ["msg3", "reply3"]
        # SB shifted by the drop amount: 0 + (6 - 6) = 0.
        assert session.staged_boundary == 0

    @pytest.mark.asyncio
    async def test_multiple_consolidation_rounds_preserve_context(
        self, session: Session, make_consolidator, sessions: SessionManager
    ):
        """Round 1 (commit + drop) injects ONE context msg; a later round over the
        staged region is a free drop — exactly one context message, no stacking.
        """
        consolidator = make_consolidator(context_window_tokens=6400)

        # --- Round 1: commit + drop with injection ---
        estimate_calls = [0]

        def estimate_round1(sess):
            estimate_calls[0] += 1
            return (10000, "mock") if estimate_calls[0] == 1 else (100, "mock")

        with (
            patch.object(
                consolidator, "estimate_session_prompt_tokens", side_effect=estimate_round1
            ),
            patch.object(
                consolidator,
                "consolidate_messages",
                new=AsyncMock(return_value=("summary", "Round 1 context")),
            ),
            patch.object(sessions, "save", new=MagicMock()),
        ):
            await consolidator.maybe_consolidate_by_tokens(session, 6400)

        assert len([m for m in session.messages if m.get("_compact_summary")]) == 1
        assert "Round 1 context" in session.messages[0]["content"]
        assert Session.context_prefix_len(session.messages) == 1  # new ctx_end

        # --- New work arrives and is fully staged (SB advanced past ctx_end) ---
        session.add_message("user", "msg4")
        session.add_message("assistant", "reply4")
        session.add_message("user", "msg5")
        session.add_message("assistant", "reply5")
        # state: [ctx(0), msg3(1), reply3(2), msg4(3), reply4(4), msg5(5), reply5(6)]
        session.staged_boundary = 7  # everything after ctx is staged

        # --- Round 2: picker boundary falls in staged region -> FREE DROP ---
        mock_consolidate2 = AsyncMock(return_value=("summary", None))
        round2_calls = [0]

        def estimate_round2(sess):
            round2_calls[0] += 1
            # est == budget exactly: passes the trigger gate, and the picker
            # target (640) is far above the real removed tokens, so it returns
            # the LAST user boundary (idx 5), which lies in the staged region.
            if round2_calls[0] == 1:
                return (1280, "mock")
            return (100, "mock")

        with (
            patch.object(
                consolidator, "estimate_session_prompt_tokens", side_effect=estimate_round2
            ),
            patch.object(consolidator, "consolidate_messages", new=mock_consolidate2),
            patch.object(sessions, "save", new=MagicMock()),
        ):
            await consolidator.maybe_consolidate_by_tokens(session, 6400)

        # Free drop: no LLM work, no second injection.
        mock_consolidate2.assert_not_awaited()
        summary_msgs = [m for m in session.messages if m.get("_compact_summary")]
        assert len(summary_msgs) == 1
        assert "Round 1 context" in summary_msgs[0]["content"]
        assert Session.context_prefix_len(session.messages) == 1
        # [1:5] (msg3, reply3, msg4, reply4) freed; SB' = ctx_end + (SB - B) = 1 + 2.
        assert session.messages[0].get("_compact_summary") is True  # ctx kept
        assert [m["content"] for m in session.messages[1:]] == ["msg5", "reply5"]
        assert session.staged_boundary == 3

    @pytest.mark.asyncio
    async def test_injected_message_has_compact_summary_tag(
        self, session: Session, make_consolidator, sessions: SessionManager
    ):
        """Verify the injected message has _compact_summary: True."""
        consolidator = make_consolidator(context_window_tokens=6400)

        estimate_calls = [0]

        def mock_estimate(sess):
            estimate_calls[0] += 1
            return (10000, "mock") if estimate_calls[0] == 1 else (100, "mock")

        with (
            patch.object(consolidator, "estimate_session_prompt_tokens", side_effect=mock_estimate),
            patch.object(
                consolidator,
                "consolidate_messages",
                new=AsyncMock(return_value=("summary", "Context data")),
            ),
            patch.object(sessions, "save", new=MagicMock()),
        ):
            await consolidator.maybe_consolidate_by_tokens(session, 6400)

        summary_msgs = [m for m in session.messages if m.get("_compact_summary")]
        assert len(summary_msgs) == 1
        assert summary_msgs[0].get("_compact_summary") is True

    @pytest.mark.asyncio
    async def test_no_cross_session_context_leak(
        self, session: Session, make_consolidator, sessions: SessionManager
    ):
        """A-1: session context is per-distillation, not shared instance state.

        A background archive from ANOTHER session that distilled a context must
        not leak into this session's commit+drop injection.
        """
        consolidator = make_consolidator(context_window_tokens=6400)

        # Another session's background archive distilled a context.
        with patch.object(
            consolidator.store, "consolidate", new=AsyncMock(return_value=("entry A", "CTX-A"))
        ):
            await consolidator.consolidate_messages(
                [{"role": "user", "content": "other session msg"}]
            )

        # Session B now dispatches a commit+drop with NO context of its own.
        estimate_calls = [0]

        def mock_estimate(sess):
            estimate_calls[0] += 1
            return (10000, "mock") if estimate_calls[0] == 1 else (100, "mock")

        with (
            patch.object(consolidator, "estimate_session_prompt_tokens", side_effect=mock_estimate),
            patch.object(
                consolidator, "consolidate_messages", new=AsyncMock(return_value=("entry B", None))
            ),
            patch.object(sessions, "save", new=MagicMock()),
        ):
            await consolidator.maybe_consolidate_by_tokens(session, 6400)

        # No injection: B distilled no context, and A's context did not leak in.
        assert [m for m in session.messages if m.get("_compact_summary")] == []
        assert [m["content"] for m in session.messages] == ["msg3", "reply3"]

    def test_get_history_preserves_compact_summary(self, session: Session):
        """Verify get_history() preserves messages tagged with _compact_summary
        (doesn't drop them as leading non-user messages). The _compact_summary key
        itself is stripped from the output, but the message content is preserved."""
        # Insert a compact summary at the boundary
        summary_msg = {
            "role": "user",
            "content": "[Previous session context]: Some context",
            "_compact_summary": True,
        }
        session.messages.insert(0, summary_msg)

        history = session.get_history()

        # The summary message should appear in history (not dropped as leading non-user)
        context_in_history = any(
            "[Previous session context]" in (m.get("content") or "") for m in history
        )
        assert context_in_history, (
            "Compact summary message should be preserved in get_history() output"
        )
        assert history[0]["content"] == "[Previous session context]: Some context"
