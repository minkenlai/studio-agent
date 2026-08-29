"""End-to-end integration tests for the full consolidation lifecycle (Phase 2).

Verifies that maybe_consolidate_by_tokens correctly:
- Triggers consolidation when over budget
- Injects session_context as a compact summary
- Advances last_consolidated
- Preserves summaries in get_history() output
- Dedupes summaries across multiple consolidation cycles
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
    provider.generation.context_max = 64000
    return provider


@pytest.fixture
def sessions(workspace: Path) -> SessionManager:
    return SessionManager(workspace)


@pytest.fixture
def make_consolidator(mock_provider: MagicMock, sessions: SessionManager):
    """Factory for creating a MemoryConsolidator with full mocking."""

    def _factory(context_window_tokens: int = 64000) -> MemoryConsolidator:
        # max_completion_tokens=128 keeps budgets positive even for small
        # context windows (budget = window - 128 - 1024).
        return MemoryConsolidator(
            workspace=sessions.workspace,
            provider=mock_provider,
            model="test-model",
            sessions=sessions,
            context_window_tokens=context_window_tokens,
            build_messages=lambda *args, **kwargs: [],
            get_tool_definitions=lambda: [],
            max_completion_tokens=128,
        )

    return _factory


def _build_overbudget_session(num_pairs: int = 10) -> Session:
    """Create a session with enough message pairs to be over budget."""
    s = Session(key="test:fullcycle")
    for i in range(num_pairs):
        s.add_message("user", f"This is user message number {i} with some content to fill tokens.")
        s.add_message(
            "assistant", f"Assistant reply to message {i} with detailed response content."
        )
    s.staged_boundary = 0
    return s


class TestFullConsolidationCycle:
    """End-to-end integration tests for consolidation with context injection."""

    @pytest.mark.asyncio
    async def test_full_consolidation_cycle_with_context_injection(
        self, make_consolidator, sessions: SessionManager
    ):
        """Full lifecycle: over-budget → commit + drop → inject context → under-budget.

        Verifies a single dual-tagged context message is injected at the new
        pinned prefix (ctx_end), SB advances to ctx_end + 1, and get_history()
        starts with the summary.
        """
        consolidator = make_consolidator(context_window_tokens=2500)
        session = _build_overbudget_session(num_pairs=10)

        # No pre-existing context; the consolidation below produces one.
        assert consolidator._last_session_context is None

        # estimate: first call over budget (trigger), second under target (stop).
        estimate_calls = [0]

        def mock_estimate(sess: Session):
            estimate_calls[0] += 1
            if estimate_calls[0] == 1:
                return (10000, "mock")  # over budget → trigger consolidation
            return (100, "mock")  # under target → stop

        # consolidate_messages: return a summary entry and set session_context.
        async def mock_consolidate(chunk):
            consolidator._last_session_context = "Phase 2 consolidation cycle complete."
            return "2026-08-06: Consolidated session messages."

        mock_consolidate_msg = AsyncMock(side_effect=mock_consolidate)
        with (
            patch.object(
                consolidator,
                "estimate_session_prompt_tokens",
                side_effect=mock_estimate,
            ),
            patch.object(consolidator, "consolidate_messages", new=mock_consolidate_msg),
            patch.object(sessions, "save", new=MagicMock()),
        ):
            await consolidator.maybe_consolidate_by_tokens(session, 2500)

            # Commit + drop happened exactly once.
            mock_consolidate_msg.assert_awaited_once()
            # The distilled chunk is the fresh part [SB:end_idx] = [0:18] (picker
            # returns the last user boundary, idx 18).
            archived = mock_consolidate_msg.call_args[0][0]
            assert len(archived) == 18

        # A compact summary should have been injected (single dual-tagged msg).
        summary_msgs = [m for m in session.messages if m.get("_compact_summary")]
        assert len(summary_msgs) == 1
        injected = session.messages[0]
        assert injected is summary_msgs[0]
        assert injected["role"] == "user"
        assert "[Previous session context]" in injected["content"]
        assert "Phase 2 consolidation cycle complete." in injected["content"]
        assert injected["_session_context"] is True
        assert injected["_compact_summary"] is True

        # Only the last live pair (u9, a9) remains after the drop + 1 injected.
        assert [m["content"] for m in session.messages[1:]] == [
            "This is user message number 9 with some content to fill tokens.",
            "Assistant reply to message 9 with detailed response content.",
        ]
        assert session.staged_boundary == 1  # ctx_end(0) + 1 for the injected msg

        # get_history() should include the summary as the first message.
        # (get_history strips the _compact_summary tag key but keeps the content.)
        history = session.get_history()
        assert history[0]["role"] == "user"
        assert "[Previous session context]" in history[0].get("content", "")

    @pytest.mark.asyncio
    async def test_get_history_shows_compact_summary_first(
        self, make_consolidator, sessions: SessionManager
    ):
        """After consolidation + injection, get_history() returns summary as first message.

        Verifies subsequent messages follow naturally after the summary.
        """
        consolidator = make_consolidator(context_window_tokens=2500)
        session = _build_overbudget_session(num_pairs=10)

        estimate_calls = [0]

        def mock_estimate(sess: Session):
            estimate_calls[0] += 1
            if estimate_calls[0] == 1:
                return (10000, "mock")
            return (100, "mock")

        async def mock_consolidate(chunk):
            consolidator._last_session_context = "Pipeline build phase complete."
            return "Consolidated pipeline work."

        with (
            patch.object(
                consolidator,
                "estimate_session_prompt_tokens",
                side_effect=mock_estimate,
            ),
            patch.object(
                consolidator,
                "consolidate_messages",
                new=AsyncMock(side_effect=mock_consolidate),
            ),
            patch.object(sessions, "save", new=MagicMock()),
        ):
            await consolidator.maybe_consolidate_by_tokens(session, 2500)

        history = session.get_history()

        # The first message in history should be the compact summary
        assert len(history) > 0, "History should not be empty after consolidation"
        first_msg = history[0]
        assert first_msg.get("role") == "user"
        assert "[Previous session context]" in (first_msg.get("content") or "")
        assert "Pipeline build phase complete." in (first_msg.get("content") or "")

        # Subsequent messages should follow naturally (not be compact summaries)
        subsequent = history[1:]
        assert len(subsequent) > 0, "Should have messages after the summary"
        assert all(not m.get("_compact_summary") for m in subsequent)

    @pytest.mark.asyncio
    async def test_multiple_cycles_dont_stack_summaries(
        self, make_consolidator, sessions: SessionManager
    ):
        """Round 1 (commit + drop) injects ONE context msg. Round 2 lands on the
        staged region → free drop: no second LLM call, no second injection, and
        get_history() shows exactly one summary (no stacking).
        """
        consolidator = make_consolidator(context_window_tokens=2500)
        session = _build_overbudget_session(num_pairs=20)

        # estimate spans BOTH consolidation invocations:
        #   1) round 1 trigger (over), 2) round 1 stop,
        #   3) round 2 trigger (= budget exactly), 4) round 2 stop.
        estimates = [(10000, "mock"), (100, "mock"), (1348, "mock"), (100, "mock")]
        estimate_calls = [0]

        def mock_estimate(sess: Session):
            idx = min(estimate_calls[0], len(estimates) - 1)
            estimate_calls[0] += 1
            return estimates[idx]

        consolidate_calls = [0]

        async def mock_consolidate(chunk):
            consolidate_calls[0] += 1
            consolidator._last_session_context = f"Cycle {consolidate_calls[0]} context."
            return f"Consolidated batch {consolidate_calls[0]}."

        mock_consolidate_msg = AsyncMock(side_effect=mock_consolidate)
        with (
            patch.object(
                consolidator,
                "estimate_session_prompt_tokens",
                side_effect=mock_estimate,
            ),
            patch.object(consolidator, "consolidate_messages", new=mock_consolidate_msg),
            patch.object(sessions, "save", new=MagicMock()),
        ):
            # --- Round 1: over-budget → commit + drop → inject context ---
            await consolidator.maybe_consolidate_by_tokens(session, 2500)
            assert consolidate_calls[0] == 1
            assert len([m for m in session.messages if m.get("_compact_summary")]) == 1
            assert "Cycle 1 context." in session.messages[0]["content"]
            # state: [ctx(0), u19(1), a19(2)], SB = 1

            # --- New work arrives and is fully staged ---
            session.add_message("user", "u20")
            session.add_message("assistant", "a20")
            session.add_message("user", "u21")
            session.add_message("assistant", "a21")
            # state: [ctx(0), u19(1), a19(2), u20(3), a20(4), u21(5), a21(6)]
            session.staged_boundary = 7

            # --- Round 2: picker boundary in staged region → FREE DROP ---
            await consolidator.maybe_consolidate_by_tokens(session, 2500)

            # Only ONE consolidation ever ran (round 2 was a free drop).
            assert consolidate_calls[0] == 1
            assert mock_consolidate_msg.await_count == 1

            # Exactly one compact summary exists, with the round-1 context.
            summary_msgs = [m for m in session.messages if m.get("_compact_summary")]
            assert len(summary_msgs) == 1
            assert "Cycle 1 context." in summary_msgs[0]["content"]
            assert Session.context_prefix_len(session.messages) == 1

            # Free drop removed [1:5] (u19, a19, u20, a20); SB' = 1 + (7 - 5) = 3.
            assert [m["content"] for m in session.messages[1:]] == ["u21", "a21"]
            assert session.staged_boundary == 3

        # get_history() shows only one summary, as the first message.
        # (get_history strips the _compact_summary tag key but keeps the content.)
        history = session.get_history()
        history_summaries = [
            m for m in history if "[Previous session context]" in (m.get("content") or "")
        ]
        assert len(history_summaries) == 1
        assert history[0]["role"] == "user"
