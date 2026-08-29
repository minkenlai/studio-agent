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


def _build_overbudget_session(num_pairs: int = 10) -> Session:
    """Create a session with enough message pairs to be over budget."""
    s = Session(key="test:fullcycle")
    for i in range(num_pairs):
        s.add_message("user", f"This is user message number {i} with some content to fill tokens.")
        s.add_message(
            "assistant", f"Assistant reply to message {i} with detailed response content."
        )
    s.last_consolidated = 0
    return s


class TestFullConsolidationCycle:
    """End-to-end integration tests for consolidation with context injection."""

    @pytest.mark.asyncio
    async def test_full_consolidation_cycle_with_context_injection(
        self, make_consolidator, sessions: SessionManager
    ):
        """Full lifecycle: over-budget → consolidate → inject context → under-budget.

        Verifies last_consolidated advances, compact summary injected, and
        get_history() includes the summary.
        """
        consolidator = make_consolidator(context_window_tokens=500)
        session = _build_overbudget_session(num_pairs=10)

        # Pre-set a session_context from a prior consolidation so injection happens
        consolidator._last_session_context = "Working on Phase 2 unified consolidation tasks."

        # estimate_session_prompt_tokens: first call says over-budget, second says under
        estimate_calls = [0]

        def mock_estimate(sess: Session):
            estimate_calls[0] += 1
            if estimate_calls[0] == 1:
                return (10000, "mock")  # over budget → trigger consolidation
            return (100, "mock")  # under budget → stop

        # consolidate_messages: return a summary entry and set session_context
        async def mock_consolidate(chunk):
            consolidator._last_session_context = "Phase 2 consolidation cycle complete."
            return "2026-08-06: Consolidated session messages."

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
            await consolidator.maybe_consolidate_by_tokens(session, 500)

        # last_consolidated should have advanced past 0
        assert session.last_consolidated > 0, (
            f"last_consolidated should advance after consolidation, got {session.last_consolidated}"
        )

        # A compact summary should have been injected
        summary_msgs = [m for m in session.messages if m.get("_compact_summary")]
        assert len(summary_msgs) == 1, (
            f"Expected exactly 1 summary message, got {len(summary_msgs)}"
        )
        assert summary_msgs[0]["role"] == "user"
        assert "[Previous session context]" in summary_msgs[0]["content"]
        assert "Phase 2 consolidation cycle complete." in summary_msgs[0]["content"]

        # get_history() should include the summary
        history = session.get_history()
        context_in_history = any(
            "[Previous session context]" in (m.get("content") or "") for m in history
        )
        assert context_in_history, "Compact summary should be present in get_history()"

    @pytest.mark.asyncio
    async def test_get_history_shows_compact_summary_first(
        self, make_consolidator, sessions: SessionManager
    ):
        """After consolidation + injection, get_history() returns summary as first message.

        Verifies subsequent messages follow naturally after the summary.
        """
        consolidator = make_consolidator(context_window_tokens=500)
        session = _build_overbudget_session(num_pairs=10)
        consolidator._last_session_context = "Building AI consolidation pipeline."

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
            await consolidator.maybe_consolidate_by_tokens(session, 500)

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
        """Simulate 2+ consolidation cycles; verify only 1 compact summary exists.

        Old summaries are removed before each new injection.
        """
        consolidator = make_consolidator(context_window_tokens=500)
        session = _build_overbudget_session(num_pairs=20)
        consolidator._last_session_context = "Cycle 0 context."

        estimate_calls = [0]

        # 3 calls: initial check (over), post-round1 (over), post-round2 (under)
        def mock_estimate(sess: Session):
            estimate_calls[0] += 1
            if estimate_calls[0] <= 2:
                return (10000, "mock")
            return (100, "mock")

        consolidate_calls = [0]
        contexts = ["Cycle 1 context.", "Cycle 2 context."]

        async def mock_consolidate(chunk):
            idx = consolidate_calls[0]
            consolidate_calls[0] += 1
            ctx = contexts[min(idx, len(contexts) - 1)]
            consolidator._last_session_context = ctx
            return f"Consolidated batch {idx + 1}."

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
            await consolidator.maybe_consolidate_by_tokens(session, 500)

        # Only one compact summary should exist at the end
        summary_msgs = [m for m in session.messages if m.get("_compact_summary")]
        assert len(summary_msgs) == 1, (
            f"Expected exactly 1 summary after multiple cycles, got {len(summary_msgs)}. "
            "Old summaries should be removed before injecting new ones."
        )

        # It should contain the latest context
        assert "Cycle 2 context." in summary_msgs[0]["content"]

        # get_history() should also show only one summary
        history = session.get_history()
        history_summaries = [
            m for m in history if "[Previous session context]" in (m.get("content") or "")
        ]
        assert len(history_summaries) == 1, (
            f"get_history() should contain exactly 1 summary, got {len(history_summaries)}"
        )
