"""Tests for MemoryConsolidator.consolidate_messages."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.memory import MemoryConsolidator
from nanobot.providers.base import LLMClient


class MockResponse:
    def __init__(self, content=None, tool_calls=None, finish_reason=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason
        self.has_tool_calls = len(self.tool_calls) > 0


class MockToolCall:
    def __init__(self, arguments):
        self.arguments = arguments


@pytest.mark.asyncio
async def test_consolidate_messages_returns_summary_string(tmp_path):
    """Verify consolidate_messages returns the summary entry (as a tuple), not a boolean."""
    # Setup
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "memory").mkdir()

    mock_provider = MagicMock(spec=LLMClient)
    mock_provider.generation = MagicMock()
    mock_provider.generation.context_max = 64000

    # Mock successful tool call response
    summary_text = "This is a test summary of the conversation."
    mock_response = MockResponse(
        tool_calls=[
            MockToolCall(
                arguments=f'{{"history_entry": "{summary_text}", "memory_update": "Fact: test update"}}'
            )
        ]
    )
    mock_provider.chat_with_retry = AsyncMock(return_value=mock_response)

    consolidator = MemoryConsolidator(
        workspace=workspace,
        provider=mock_provider,
        model="test-model",
        sessions=MagicMock(),
        context_window_tokens=64000,
        build_messages=lambda *args, **kwargs: [],
        get_tool_definitions=lambda: [],
    )

    messages = [{"role": "user", "content": "hello"}]

    # Execute
    result = await consolidator.consolidate_messages(messages)

    # Assert: (history_entry, session_context) tuple; entry is the summary str.
    assert isinstance(result, tuple)
    entry, ctx = result
    assert entry == summary_text
    assert isinstance(entry, str)
    assert entry is not True  # Specifically checking against the bug
    assert ctx is None  # no session_context in the tool response


@pytest.mark.asyncio
async def test_consolidate_messages_returns_none_on_failure(tmp_path):
    """Verify consolidate_messages returns None when consolidation fails."""
    # Setup
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "memory").mkdir()

    mock_provider = MagicMock(spec=LLMClient)
    mock_provider.generation = MagicMock()
    mock_provider.generation.context_max = 64000

    # Mock failure (no tool calls, no content)
    mock_response = MockResponse(content=None, tool_calls=[])
    mock_provider.chat_with_retry = AsyncMock(return_value=mock_response)

    consolidator = MemoryConsolidator(
        workspace=workspace,
        provider=mock_provider,
        model="test-model",
        sessions=MagicMock(),
        context_window_tokens=64000,
        build_messages=lambda *args, **kwargs: [],
        get_tool_definitions=lambda: [],
    )

    messages = [{"role": "user", "content": "hello"}]

    # Execute
    result = await consolidator.consolidate_messages(messages)

    # Assert: failure propagates as (None, None)
    assert result == (None, None)
