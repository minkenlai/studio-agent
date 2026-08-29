"""Tests for session_context field in save_memory tool and consolidation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.memory import _SAVE_MEMORY_TOOL, MemoryStore
from nanobot.providers.base import LLMClient


class MockToolCall:
    def __init__(self, arguments):
        self.arguments = arguments


class MockResponse:
    def __init__(self, content=None, tool_calls=None, finish_reason="stop"):
        self.content = content
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason
        self.has_tool_calls = len(self.tool_calls) > 0


class TestSaveMemoryToolSchema:
    """Verify the save_memory tool schema has session_context as optional."""

    def test_schema_has_session_context_property(self):
        """session_context must be an optional property in the schema."""
        tool = _SAVE_MEMORY_TOOL[0]
        props = tool["function"]["parameters"]["properties"]
        assert "session_context" in props, "session_context property missing from schema"

    def test_session_context_is_not_required(self):
        """session_context must NOT be in the required list (backward compat)."""
        tool = _SAVE_MEMORY_TOOL[0]
        required = tool["function"]["parameters"]["required"]
        assert "session_context" not in required, "session_context should be optional"

    def test_session_context_type_is_string(self):
        """session_context must be a string type."""
        tool = _SAVE_MEMORY_TOOL[0]
        props = tool["function"]["parameters"]["properties"]
        assert props["session_context"]["type"] == "string"

    def test_required_fields_still_present(self):
        """history_entry and memory_update must still be required."""
        tool = _SAVE_MEMORY_TOOL[0]
        required = tool["function"]["parameters"]["required"]
        assert "history_entry" in required
        assert "memory_update" in required


@pytest.mark.asyncio
class TestConsolidateSessionContext:
    """Test session_context extraction and return from consolidate()."""

    @pytest.fixture
    def workspace(self, tmp_path: Path) -> Path:
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "memory").mkdir()
        return ws

    @pytest.fixture
    def mock_provider(self) -> MagicMock:
        provider = MagicMock(spec=LLMClient)
        provider.generation = MagicMock()
        provider.generation.context_max = 64000
        provider.generation.max_tokens = 4096
        return provider

    @pytest.fixture
    def store(self, workspace: Path) -> MemoryStore:
        return MemoryStore(workspace)

    async def test_consolidate_returns_session_context_when_provided(
        self, store: MemoryStore, mock_provider: MagicMock
    ):
        """When the LLM returns session_context, consolidate() should return it."""
        mock_response = MockResponse(
            tool_calls=[
                MockToolCall(
                    arguments=json.dumps(
                        {
                            "history_entry": "[2026-08-05 20:00] Test conversation summary",
                            "memory_update": "Fact: test",
                            "session_context": "User is working on memory consolidation project",
                        }
                    )
                )
            ]
        )
        mock_provider.chat_with_retry = AsyncMock(return_value=mock_response)

        result = await store.consolidate(
            messages=[{"role": "user", "content": "hello"}],
            provider=mock_provider,
            model="test-model",
        )

        # Result should include session_context
        assert result is not None
        # After implementation: result will be a tuple (history_entry, session_context)
        # or a dict/dataclass with both fields
        if isinstance(result, tuple):
            assert len(result) == 2
            assert result[0] == "[2026-08-05 20:00] Test conversation summary"
            assert result[1] == "User is working on memory consolidation project"
        elif isinstance(result, dict):
            assert result.get("history_entry") == "[2026-08-05 20:00] Test conversation summary"
            assert (
                result.get("session_context") == "User is working on memory consolidation project"
            )

    async def test_consolidate_works_without_session_context(
        self, store: MemoryStore, mock_provider: MagicMock
    ):
        """Backward compat: consolidate must work if session_context is omitted."""
        mock_response = MockResponse(
            tool_calls=[
                MockToolCall(
                    arguments=json.dumps(
                        {
                            "history_entry": "[2026-08-05 20:00] Test summary",
                            "memory_update": "Fact: test",
                        }
                    )
                )
            ]
        )
        mock_provider.chat_with_retry = AsyncMock(return_value=mock_response)

        result = await store.consolidate(
            messages=[{"role": "user", "content": "hello"}],
            provider=mock_provider,
            model="test-model",
        )

        assert result is not None
        # session_context should be None/empty when not provided
        if isinstance(result, tuple):
            assert result[1] is None or result[1] == ""
        elif isinstance(result, dict):
            assert result.get("session_context") is None or result.get("session_context") == ""

    async def test_consolidate_uses_8k_max_tokens(
        self, store: MemoryStore, mock_provider: MagicMock
    ):
        """The consolidation LLM call should use max_tokens=8192."""
        mock_response = MockResponse(
            tool_calls=[
                MockToolCall(
                    arguments=json.dumps(
                        {
                            "history_entry": "[2026-08-05 20:00] Test",
                            "memory_update": "test",
                        }
                    )
                )
            ]
        )
        mock_provider.chat_with_retry = AsyncMock(return_value=mock_response)

        await store.consolidate(
            messages=[{"role": "user", "content": "hello"}],
            provider=mock_provider,
            model="test-model",
        )

        # Verify chat_with_retry was called with max_tokens=8192
        assert mock_provider.chat_with_retry.called
        call_kwargs = mock_provider.chat_with_retry.call_args[1]
        assert call_kwargs.get("max_tokens") == 8192, (
            f"Expected max_tokens=8192, got {call_kwargs.get('max_tokens')}"
        )

    async def test_prompt_includes_staging_content(
        self, store: MemoryStore, mock_provider: MagicMock
    ):
        """The consolidation prompt should include STAGING.md content."""
        # Write something to STAGING.md
        store.staging_file.write_text(
            "## Notes\n- Important fact about the project", encoding="utf-8"
        )

        mock_response = MockResponse(
            tool_calls=[
                MockToolCall(
                    arguments=json.dumps(
                        {
                            "history_entry": "[2026-08-05 20:00] Test",
                            "memory_update": "test",
                        }
                    )
                )
            ]
        )
        mock_provider.chat_with_retry = AsyncMock(return_value=mock_response)

        await store.consolidate(
            messages=[{"role": "user", "content": "hello"}],
            provider=mock_provider,
            model="test-model",
        )

        # Check the prompt passed to chat_with_retry contains staging content
        call_kwargs = mock_provider.chat_with_retry.call_args[1]
        user_msg = call_kwargs["messages"][1]["content"]
        assert "## Pending Staging Notes" in user_msg
        assert "Important fact about the project" in user_msg

    async def test_prompt_shows_none_when_staging_empty(
        self, store: MemoryStore, mock_provider: MagicMock
    ):
        """The consolidation prompt should show '(none)' when STAGING.md doesn't exist."""
        # Ensure STAGING.md doesn't exist
        if store.staging_file.exists():
            store.staging_file.unlink()

        mock_response = MockResponse(
            tool_calls=[
                MockToolCall(
                    arguments=json.dumps(
                        {
                            "history_entry": "[2026-08-05 20:00] Test",
                            "memory_update": "test",
                        }
                    )
                )
            ]
        )
        mock_provider.chat_with_retry = AsyncMock(return_value=mock_response)

        await store.consolidate(
            messages=[{"role": "user", "content": "hello"}],
            provider=mock_provider,
            model="test-model",
        )

        call_kwargs = mock_provider.chat_with_retry.call_args[1]
        user_msg = call_kwargs["messages"][1]["content"]
        assert "## Pending Staging Notes" in user_msg
        assert "(none)" in user_msg
