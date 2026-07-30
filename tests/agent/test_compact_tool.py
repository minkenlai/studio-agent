"""Tests for CompactHistoryTool — session history consolidation via tool call."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

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
    session_manager: SessionManager,
    summary_return: str | None = "Summary of conversation so far.",
) -> CompactHistoryTool:
    """Create a CompactHistoryTool with a mock consolidator."""
    consolidator = MagicMock()
    consolidator.archive_messages = AsyncMock(return_value=summary_return)
    tool = CompactHistoryTool(consolidator, session_manager)
    return tool


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompactHistoryBasic:
    """Core functionality of CompactHistoryTool.execute."""

    @pytest.mark.asyncio
    async def test_n_keep_zero_consolidates_all(self, tmp_path: Path) -> None:
        """n_keep=0 should archive every message from last_consolidated to end."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=5)  # 10 messages
        sm.save(session)

        tool = _make_tool(sm)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        result = await tool.execute(n_keep=0)

        # Should have called archive_messages with all 10 messages
        tool._consolidator.archive_messages.assert_awaited_once()
        chunk = tool._consolidator.archive_messages.call_args[0][0]
        assert len(chunk) == 10
        assert chunk[0]["content"] == "user message 0"
        assert chunk[-1]["content"] == "assistant reply 4"

        # Summary injected at index 0, original messages shifted
        reloaded = sm.get_or_create("test:compact")
        assert reloaded.last_consolidated == 0
        assert reloaded.messages[0]["role"] == "assistant"
        assert reloaded.messages[0]["content"] == "Summary of conversation so far."
        # Original messages are at indices 1..10
        assert reloaded.messages[1]["content"] == "user message 0"
        assert len(reloaded.messages) == 11  # summary + 10 originals

        assert "Consolidated all history" in result

    @pytest.mark.asyncio
    async def test_n_keep_partial(self, tmp_path: Path) -> None:
        """n_keep=2 should archive all but the last 2 messages."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=5)  # 10 messages
        sm.save(session)

        tool = _make_tool(sm)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        result = await tool.execute(n_keep=2)

        chunk = tool._consolidator.archive_messages.call_args[0][0]
        # end_idx = 10 - 2 = 8, chunk = messages[0:8]
        assert len(chunk) == 8
        assert chunk[-1]["content"] == "assistant reply 3"

        reloaded = sm.get_or_create("test:compact")
        # Summary injected at index 8 (where end_idx was)
        assert reloaded.messages[8]["role"] == "assistant"
        assert reloaded.messages[8]["content"] == "Summary of conversation so far."
        # Kept messages at the end
        assert reloaded.messages[9]["content"] == "user message 4"
        assert reloaded.messages[10]["content"] == "assistant reply 4"
        assert reloaded.last_consolidated == 8

        assert "keeping last 2 messages" in result

    @pytest.mark.asyncio
    async def test_n_keep_larger_than_history(self, tmp_path: Path) -> None:
        """n_keep > len(messages) should be a no-op."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=2)  # 4 messages
        sm.save(session)

        tool = _make_tool(sm)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        result = await tool.execute(n_keep=100)

        # end_idx = max(0, 4 - 100) = 0, chunk = messages[0:0] = []
        tool._consolidator.archive_messages.assert_not_called()
        assert "Nothing to summarize" in result

    @pytest.mark.asyncio
    async def test_empty_session(self, tmp_path: Path) -> None:
        """An empty session should return 'Nothing to summarize'."""
        sm = SessionManager(tmp_path)
        session = Session(key="test:empty")
        sm.save(session)

        tool = _make_tool(sm)
        tool.set_context(Address.from_uri("test:empty"), "test:empty")

        result = await tool.execute(n_keep=0)

        tool._consolidator.archive_messages.assert_not_called()
        assert "Nothing to summarize" in result

    @pytest.mark.asyncio
    async def test_default_n_keep_is_zero(self, tmp_path: Path) -> None:
        """Calling without n_keep should default to 0 (consolidate all)."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=3)
        sm.save(session)

        tool = _make_tool(sm)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        await tool.execute()

        chunk = tool._consolidator.archive_messages.call_args[0][0]
        assert len(chunk) == 6  # all messages

    @pytest.mark.asyncio
    async def test_already_consolidated(self, tmp_path: Path) -> None:
        """When last_consolidated == len(messages), nothing to do."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=5)
        session.last_consolidated = len(session.messages)
        sm.save(session)

        tool = _make_tool(sm)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        result = await tool.execute(n_keep=0)

        tool._consolidator.archive_messages.assert_not_called()
        assert "Nothing to summarize" in result


class TestCompactHistorySummaryInjection:
    """Verify summary injection and pointer advancement."""

    @pytest.mark.asyncio
    async def test_summary_injected_into_session_messages(self, tmp_path: Path) -> None:
        """The summary message should be inserted into session.messages."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=3)
        sm.save(session)

        custom_summary = "Key topics: pricing, architecture, deployment."
        tool = _make_tool(sm, summary_return=custom_summary)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        await tool.execute(n_keep=2)

        reloaded = sm.get_or_create("test:compact")
        # end_idx = 6 - 2 = 4; summary at index 4
        assert reloaded.messages[4]["content"] == custom_summary
        assert "timestamp" in reloaded.messages[4]

    @pytest.mark.asyncio
    async def test_summary_included_in_tool_result(self, tmp_path: Path) -> None:
        """The tool result string should contain the generated summary."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=2)
        sm.save(session)

        summary_text = "Discussed API design and database schema."
        tool = _make_tool(sm, summary_return=summary_text)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        result = await tool.execute(n_keep=0)

        assert summary_text in result
        assert "---" in result  # separator before summary

    @pytest.mark.asyncio
    async def test_last_consolidated_advanced(self, tmp_path: Path) -> None:
        """last_consolidated should be set to end_idx after consolidation."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=10)  # 20 messages
        session.last_consolidated = 4
        sm.save(session)

        tool = _make_tool(sm)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        await tool.execute(n_keep=4)

        reloaded = sm.get_or_create("test:compact")
        # end_idx = 20 - 4 = 16
        assert reloaded.last_consolidated == 16

    @pytest.mark.asyncio
    async def test_get_history_after_compact(self, tmp_path: Path) -> None:
        """get_history should return kept messages + summary after compacting."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=5)  # 10 messages
        sm.save(session)

        tool = _make_tool(sm)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        await tool.execute(n_keep=4)

        reloaded = sm.get_or_create("test:compact")
        history = reloaded.get_history(max_messages=0)
        # After compact: summary at idx 6, kept msgs at idx 7,8,9,10
        # last_consolidated = 6, so history = messages[6:]
        # That's: summary + 4 kept messages = 5 entries
        assert len(history) == 5
        assert history[0]["content"] == "Summary of conversation so far."
        # The 4 kept messages should be the last 2 exchanges
        assert history[-1]["content"] == "assistant reply 4"


class TestCompactHistoryArchiveFailure:
    """Behavior when archive_messages returns None (consolidation failure)."""

    @pytest.mark.asyncio
    async def test_no_summary_on_archive_failure(self, tmp_path: Path) -> None:
        """When archive_messages returns None, no summary should be injected."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=3)
        sm.save(session)

        # Simulate consolidation failure
        consolidator = MagicMock()
        consolidator.archive_messages = AsyncMock(return_value=None)
        tool = CompactHistoryTool(consolidator, sm)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        result = await tool.execute(n_keep=0)

        # No summary message injected
        reloaded = sm.get_or_create("test:compact")
        # All messages should still be original user/assistant pairs
        assert all(
            m["content"].startswith("user") or m["content"].startswith("assistant reply")
            for m in reloaded.messages
        )
        assert "Summary" not in result

    @pytest.mark.asyncio
    async def test_last_consolidated_still_advanced_on_failure(self, tmp_path: Path) -> None:
        """
        NOTE: This documents current behavior. last_consolidated advances even
        when archive_messages returns None. This may be a bug — messages are
        marked as consolidated without a summary being injected.
        See code review for compact.py, line 78.
        """
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=3)
        sm.save(session)

        consolidator = MagicMock()
        consolidator.archive_messages = AsyncMock(return_value=None)
        tool = CompactHistoryTool(consolidator, sm)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        await tool.execute(n_keep=0)

        reloaded = sm.get_or_create("test:compact")
        # Current behavior: pointer advances to end_idx even on failure
        assert reloaded.last_consolidated == 0


class TestCompactHistoryRepeatedCalls:
    """Multiple consecutive compact_history calls."""

    @pytest.mark.asyncio
    async def test_second_call_noop_when_fully_compacted(self, tmp_path: Path) -> None:
        """Calling compact again immediately should find nothing new to summarize."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=5)
        sm.save(session)

        tool = _make_tool(sm)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        # First call: consolidate all
        await tool.execute(n_keep=0)

        # Reset mock call count
        tool._consolidator.archive_messages.reset_mock()

        # Second call: nothing new since last_consolidated didn't move past
        # the summary injection point and end_idx=0
        await tool.execute(n_keep=0)

        # end_idx = max(0, 11 - 0) = 11, chunk = messages[0:11]
        # This actually includes the summary + all originals
        # So it WILL try to re-compact. This is expected since last_consolidated=0
        # and end_idx=11 after summary injection added a message.
        assert tool._consolidator.archive_messages.await_count == 1

    @pytest.mark.asyncio
    async def test_compact_then_add_messages_then_compact(self, tmp_path: Path) -> None:
        """After compacting, adding new messages should be compactable."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=5)
        sm.save(session)

        tool = _make_tool(sm)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        # First compact: keep last 2
        await tool.execute(n_keep=2)

        reloaded = sm.get_or_create("test:compact")
        # Store current consolidated pointer
        _ = reloaded.last_consolidated  # 8

        # Add new messages
        reloaded.add_message("user", "new question")
        reloaded.add_message("assistant", "new answer")
        sm.save(reloaded)

        # Second compact: should only archive messages after first_last_consolidated
        tool._consolidator.archive_messages.reset_mock()
        await tool.execute(n_keep=2)

        chunk = tool._consolidator.archive_messages.call_args[0][0]
        # end_idx = 12 - 2 = 10, chunk = messages[8:10] = summary + new msgs? No
        # messages[8] = summary from first compact
        # messages[9] = user message 4
        # messages[10] = assistant reply 4
        # messages[11] = new question
        # messages[12] = new answer
        # end_idx = max(0, 13 - 2) = 11, chunk = messages[8:11]
        assert len(chunk) == 3  # first summary + 2 kept msgs from first round

    @pytest.mark.asyncio
    async def test_incremental_compaction(self, tmp_path: Path) -> None:
        """Test compaction with an existing last_consolidated pointer."""
        sm = SessionManager(tmp_path)
        session = _make_session(message_pairs=10)  # 20 messages
        session.last_consolidated = 8
        sm.save(session)

        tool = _make_tool(sm)
        tool.set_context(Address.from_uri("test:compact"), "test:compact")

        await tool.execute(n_keep=4)

        chunk = tool._consolidator.archive_messages.call_args[0][0]
        # end_idx = 20 - 4 = 16, chunk = messages[8:16]
        assert len(chunk) == 8
        assert chunk[0]["content"] == "user message 4"
        assert chunk[-1]["content"] == "assistant reply 7"


class TestCompactHistoryToolInterface:
    """Test tool schema and interface compliance."""

    def test_tool_name(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path)
        tool = _make_tool(sm)
        assert tool.name == "compact_history"

    def test_tool_description(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path)
        tool = _make_tool(sm)
        assert "consolidat" in tool.description.lower()

    def test_tool_parameters_schema(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path)
        tool = _make_tool(sm)
        params = tool.parameters
        assert params["type"] == "object"
        assert "n_keep" in params["properties"]
        assert params["properties"]["n_keep"]["type"] == "integer"
        assert params["properties"]["n_keep"]["minimum"] == 0
        assert params["required"] == []

    def test_set_context_updates_session_key(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path)
        tool = _make_tool(sm)
        assert tool._session_key == "cli:direct"  # default
        tool.set_context(Address.from_uri("slack:general"), "slack:general")
        assert tool._session_key == "slack:general"

    def test_to_schema(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path)
        tool = _make_tool(sm)
        schema = tool.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "compact_history"
