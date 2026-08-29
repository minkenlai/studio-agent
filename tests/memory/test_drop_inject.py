"""Tests for session context injection after consolidation (drop-and-inject logic)."""

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
    s.last_consolidated = 0
    return s


class TestDropAndInject:
    """Session context injection after consolidation drops old messages."""

    @pytest.mark.asyncio
    async def test_inject_session_context_after_consolidation(
        self, session: Session, make_consolidator, sessions: SessionManager
    ):
        """When _last_session_context is set and consolidation advances last_consolidated,
        a summary message should be injected at the boundary."""
        consolidator = make_consolidator(context_window_tokens=500)
        consolidator._last_session_context = "User is working on memory consolidation"

        original_msg_count = len(session.messages)

        with (
            patch.object(
                consolidator,
                "estimate_session_prompt_tokens",
                return_value=(10000, "mock"),
            ),
            patch.object(
                consolidator,
                "consolidate_messages",
                new=AsyncMock(return_value="summary"),
            ),
            patch.object(sessions, "save", new=MagicMock()),
        ):
            await consolidator.maybe_consolidate_by_tokens(session, 500)

        # A summary message should have been injected
        summary_msgs = [m for m in session.messages if m.get("_compact_summary")]
        assert len(summary_msgs) >= 1, "Expected at least one summary message after consolidation"
        summary = summary_msgs[-1]
        assert summary["role"] == "user"
        assert "[Previous session context]" in summary["content"]
        assert "User is working on memory consolidation" in summary["content"]
        assert len(session.messages) >= original_msg_count + 1

    @pytest.mark.asyncio
    async def test_no_inject_when_context_is_none(
        self, session: Session, make_consolidator, sessions: SessionManager
    ):
        """When _last_session_context is None, no summary message should be injected."""
        consolidator = make_consolidator(context_window_tokens=500)
        consolidator._last_session_context = None

        original_msg_count = len(session.messages)

        with (
            patch.object(
                consolidator,
                "estimate_session_prompt_tokens",
                return_value=(10000, "mock"),
            ),
            patch.object(
                consolidator,
                "consolidate_messages",
                new=AsyncMock(return_value="summary"),
            ),
            patch.object(sessions, "save", new=MagicMock()),
        ):
            await consolidator.maybe_consolidate_by_tokens(session, 500)

        summary_msgs = [m for m in session.messages if m.get("_compact_summary")]
        assert len(summary_msgs) == 0, "No summary should be injected when context is None"
        assert len(session.messages) == original_msg_count

    @pytest.mark.asyncio
    async def test_injected_message_has_compact_summary_tag(
        self, session: Session, make_consolidator, sessions: SessionManager
    ):
        """Verify the injected message has _compact_summary: True."""
        consolidator = make_consolidator(context_window_tokens=500)
        consolidator._last_session_context = "Context data"

        with (
            patch.object(
                consolidator,
                "estimate_session_prompt_tokens",
                return_value=(10000, "mock"),
            ),
            patch.object(
                consolidator,
                "consolidate_messages",
                new=AsyncMock(return_value="summary"),
            ),
            patch.object(sessions, "save", new=MagicMock()),
        ):
            await consolidator.maybe_consolidate_by_tokens(session, 500)

        summary_msgs = [m for m in session.messages if m.get("_compact_summary")]
        assert len(summary_msgs) == 1
        assert summary_msgs[0].get("_compact_summary") is True

    def test_get_history_preserves_compact_summary(self, session: Session):
        """Verify get_history() preserves messages tagged with _compact_summary
        (doesn't drop them as leading non-user messages). The _compact_summary key
        itself is stripped from the output, but the message content is preserved."""
        session.last_consolidated = 2

        # Insert a compact summary at the boundary
        summary_msg = {
            "role": "user",
            "content": "[Previous session context]: Some context",
            "_compact_summary": True,
        }
        session.messages.insert(2, summary_msg)

        history = session.get_history()

        # The summary message should appear in history (not dropped as leading non-user)
        context_in_history = any(
            "[Previous session context]" in (m.get("content") or "") for m in history
        )
        assert context_in_history, (
            "Compact summary message should be preserved in get_history() output"
        )

    @pytest.mark.asyncio
    async def test_inject_creates_new_message_not_mutation(
        self, session: Session, make_consolidator, sessions: SessionManager
    ):
        """The injection appends a new message to session.messages, doesn't modify existing msgs."""
        consolidator = make_consolidator(context_window_tokens=500)
        consolidator._last_session_context = "Context data"

        original_count = len(session.messages)

        # 3 calls: initial estimate, post-consolidation, then loop check → stops
        round_values = [10000, 100, 100]
        round_index = [0]

        def mock_estimate(sess):
            val = round_values[round_index[0]]
            round_index[0] += 1
            return (val, "mock")

        with (
            patch.object(
                consolidator,
                "estimate_session_prompt_tokens",
                side_effect=mock_estimate,
            ),
            patch.object(
                consolidator,
                "consolidate_messages",
                new=AsyncMock(return_value="summary"),
            ),
            patch.object(sessions, "save", new=MagicMock()),
        ):
            await consolidator.maybe_consolidate_by_tokens(session, 500)

        # A new message was added
        assert len(session.messages) == original_count + 1

        # Original messages are unchanged (just shifted by the insertion)
        # The compact summary was inserted, so original messages start at index 1+
        summary_msgs = [m for m in session.messages if m.get("_compact_summary")]
        assert len(summary_msgs) == 1

    @pytest.mark.asyncio
    async def test_multiple_consolidation_rounds_preserve_context(
        self, session: Session, make_consolidator, sessions: SessionManager
    ):
        """Each consolidation round should overwrite previous compact summary, not stack them."""
        consolidator = make_consolidator(context_window_tokens=500)
        consolidator._last_session_context = "Round 1 context"

        round_values = [10000, 10000, 100]  # 2 consolidation rounds, then done
        round_index = [0]

        def mock_estimate(sess):
            val = round_values[round_index[0]]
            round_index[0] += 1
            return (val, "mock")

        context_values = ["Round 1 context", "Round 2 context"]
        context_index = [0]

        async def mock_consolidate(chunk):
            ctx = context_values[context_index[0]]
            context_index[0] += 1
            consolidator._last_session_context = ctx
            return "summary"

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

        # Only one compact summary should exist (latest one)
        summary_msgs = [m for m in session.messages if m.get("_compact_summary")]
        assert len(summary_msgs) == 1, (
            f"Expected exactly 1 summary message, got {len(summary_msgs)}. "
            "Previous summaries should be removed before injecting new one."
        )
        # It should be the latest context
        assert "Round 2 context" in summary_msgs[0]["content"]
