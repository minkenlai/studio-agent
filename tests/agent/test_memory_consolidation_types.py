"""Test MemoryStore.consolidate() handles non-string tool call arguments.

Regression test for https://github.com/HKUDS/nanobot/issues/1042
When memory consolidation receives dict values instead of strings from the LLM
tool call response, it should serialize them to JSON instead of raising TypeError.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.memory import MemoryStore
from nanobot.providers.base import LLMClient, LLMResponse, ToolCallRequest


def _make_messages(message_count: int = 30):
    """Create a list of mock messages."""
    return [
        {"role": "user", "content": f"msg{i}", "timestamp": "2026-01-01 00:00"}
        for i in range(message_count)
    ]


def _make_tool_response(history_entry, memory_update):
    """Create an LLMResponse with a save_memory tool call."""
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCallRequest(
                id="call_1",
                name="save_memory",
                arguments={
                    "history_entry": history_entry,
                    "memory_update": memory_update,
                },
            )
        ],
    )


class ScriptedProvider(LLMClient):
    def __init__(self, responses: list[LLMResponse]):
        super().__init__()
        self._responses = list(responses)
        self.calls = 0

    async def chat(self, *args, **kwargs) -> LLMResponse:
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(content="", tool_calls=[])

    def get_default_model(self) -> str:
        return "test-model"


class TestMemoryConsolidationTypeHandling:
    """Test that consolidation handles various argument types correctly."""

    @pytest.mark.asyncio
    async def test_string_arguments_work(self, tmp_path: Path) -> None:
        """Normal case: LLM returns string arguments."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat = AsyncMock(
            return_value=_make_tool_response(
                history_entry="[2026-01-01] User discussed testing.",
                memory_update="# Memory\nUser likes testing.",
            )
        )
        provider.chat_with_retry = provider.chat
        messages = _make_messages(message_count=60)

        result = await store.consolidate(messages, provider, "test-model")

        assert result is not None
        assert store.history_file.exists()
        assert "[2026-01-01] User discussed testing." in store.history_file.read_text()
        assert "User likes testing." in store.staging_file.read_text()

    @pytest.mark.asyncio
    async def test_dict_arguments_serialized_to_json(self, tmp_path: Path) -> None:
        """Issue #1042: LLM returns dict instead of string — must not raise TypeError."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat = AsyncMock(
            return_value=_make_tool_response(
                history_entry={"timestamp": "2026-01-01", "summary": "User discussed testing."},
                memory_update={"facts": ["User likes testing"], "topics": ["testing"]},
            )
        )
        provider.chat_with_retry = provider.chat
        messages = _make_messages(message_count=60)

        result = await store.consolidate(messages, provider, "test-model")

        assert result is not None
        assert store.history_file.exists()
        history_content = store.history_file.read_text()
        parsed = json.loads(history_content.strip())
        assert parsed["summary"] == "User discussed testing."

        staging_content = store.staging_file.read_text()
        assert "User likes testing" in staging_content

    @pytest.mark.asyncio
    async def test_string_arguments_as_raw_json(self, tmp_path: Path) -> None:
        """Some providers return arguments as a JSON string instead of parsed dict."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()

        response = LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="save_memory",
                    arguments=json.dumps(
                        {
                            "history_entry": "[2026-01-01] User discussed testing.",
                            "memory_update": "# Memory\nUser likes testing.",
                        }
                    ),
                )
            ],
        )
        provider.chat = AsyncMock(return_value=response)
        provider.chat_with_retry = provider.chat
        messages = _make_messages(message_count=60)

        result = await store.consolidate(messages, provider, "test-model")

        assert result is not None
        assert "User discussed testing." in store.history_file.read_text()

    @pytest.mark.asyncio
    async def test_no_tool_call_returns_false(self, tmp_path: Path) -> None:
        """When LLM doesn't use the save_memory tool, return False."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat = AsyncMock(
            return_value=LLMResponse(content="I summarized the conversation.", tool_calls=[])
        )
        provider.chat_with_retry = provider.chat
        messages = _make_messages(message_count=60)

        result = await store.consolidate(messages, provider, "test-model")

        assert result is None
        assert not store.history_file.exists()

    @pytest.mark.asyncio
    async def test_skips_when_message_chunk_is_empty(self, tmp_path: Path) -> None:
        """Consolidation should be a no-op when the selected chunk is empty."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat_with_retry = provider.chat
        messages: list[dict] = []

        result = await store.consolidate(messages, provider, "test-model")

        assert result is None  # empty chunk → no summary
        provider.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_list_arguments_extracts_first_dict(self, tmp_path: Path) -> None:
        """Some providers return arguments as a list - extract first element if it's a dict."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()

        response = LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="save_memory",
                    arguments=[
                        {
                            "history_entry": "[2026-01-01] User discussed testing.",
                            "memory_update": "# Memory\nUser likes testing.",
                        }
                    ],
                )
            ],
        )
        provider.chat = AsyncMock(return_value=response)
        provider.chat_with_retry = provider.chat
        messages = _make_messages(message_count=60)

        result = await store.consolidate(messages, provider, "test-model")

        assert result is not None
        assert "User discussed testing." in store.history_file.read_text()
        assert "User likes testing." in store.staging_file.read_text()

    @pytest.mark.asyncio
    async def test_list_arguments_empty_list_returns_false(self, tmp_path: Path) -> None:
        """Empty list arguments should return False."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()

        response = LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="save_memory",
                    arguments=[],
                )
            ],
        )
        provider.chat = AsyncMock(return_value=response)
        provider.chat_with_retry = provider.chat
        messages = _make_messages(message_count=60)

        result = await store.consolidate(messages, provider, "test-model")

        assert result is None

    @pytest.mark.asyncio
    async def test_list_arguments_non_dict_content_returns_false(self, tmp_path: Path) -> None:
        """List with non-dict content should return False."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()

        response = LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="save_memory",
                    arguments=["string", "content"],
                )
            ],
        )
        provider.chat = AsyncMock(return_value=response)
        provider.chat_with_retry = provider.chat
        messages = _make_messages(message_count=60)

        result = await store.consolidate(messages, provider, "test-model")

        assert result is None

    @pytest.mark.asyncio
    async def test_missing_history_entry_returns_false_without_writing(
        self, tmp_path: Path
    ) -> None:
        """Do not persist partial results when required fields are missing."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(
            return_value=LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="save_memory",
                        arguments={"memory_update": "# Memory\nOnly memory update"},
                    )
                ],
            )
        )
        messages = _make_messages(message_count=60)

        result = await store.consolidate(messages, provider, "test-model")

        assert result is None
        assert not store.history_file.exists()
        assert not store.memory_file.exists()

    @pytest.mark.asyncio
    async def test_missing_memory_update_returns_false_without_writing(
        self, tmp_path: Path
    ) -> None:
        """Do not append history if memory_update is missing."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(
            return_value=LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="save_memory",
                        arguments={"history_entry": "[2026-01-01] Partial output."},
                    )
                ],
            )
        )
        messages = _make_messages(message_count=60)

        result = await store.consolidate(messages, provider, "test-model")

        assert result is None
        assert not store.history_file.exists()
        assert not store.memory_file.exists()

    @pytest.mark.asyncio
    async def test_null_required_field_returns_false_without_writing(self, tmp_path: Path) -> None:
        """Null required fields should be rejected before persistence."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(
            return_value=_make_tool_response(
                history_entry=None,
                memory_update="# Memory\nUser likes testing.",
            )
        )
        messages = _make_messages(message_count=60)

        result = await store.consolidate(messages, provider, "test-model")

        assert result is None
        assert not store.history_file.exists()
        assert not store.memory_file.exists()

    @pytest.mark.asyncio
    async def test_empty_history_entry_returns_false_without_writing(self, tmp_path: Path) -> None:
        """Empty history entries should be rejected to avoid blank archival records."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(
            return_value=_make_tool_response(
                history_entry="   ",
                memory_update="# Memory\nUser likes testing.",
            )
        )
        messages = _make_messages(message_count=60)

        result = await store.consolidate(messages, provider, "test-model")

        assert result is None
        assert not store.history_file.exists()
        assert not store.memory_file.exists()

    @pytest.mark.asyncio
    async def test_retries_transient_error_then_succeeds(self, tmp_path: Path, monkeypatch) -> None:
        store = MemoryStore(tmp_path)
        provider = ScriptedProvider(
            [
                LLMResponse(content="503 server error", finish_reason="error"),
                _make_tool_response(
                    history_entry="[2026-01-01] User discussed testing.",
                    memory_update="# Memory\nUser likes testing.",
                ),
            ]
        )
        messages = _make_messages(message_count=60)
        delays: list[int] = []

        async def _fake_sleep(delay: int) -> None:
            delays.append(delay)

        monkeypatch.setattr("nanobot.providers.base.asyncio.sleep", _fake_sleep)

        result = await store.consolidate(messages, provider, "test-model")

        assert result is not None
        assert provider.calls == 2
        assert delays == [1]

    @pytest.mark.asyncio
    async def test_consolidation_delegates_to_provider_defaults(self, tmp_path: Path) -> None:
        """Consolidation passes max_tokens=8192 for expanded output, but delegates other params to provider."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(
            return_value=_make_tool_response(
                history_entry="[2026-01-01] User discussed testing.",
                memory_update="# Memory\nUser likes testing.",
            )
        )
        messages = _make_messages(message_count=60)

        result = await store.consolidate(messages, provider, "test-model")

        assert result is not None
        provider.chat_with_retry.assert_awaited_once()
        _, kwargs = provider.chat_with_retry.await_args
        assert kwargs["model"] == "test-model"
        assert kwargs["max_tokens"] == 8192
        assert "temperature" not in kwargs
        assert "reasoning_effort" not in kwargs

    @pytest.mark.asyncio
    async def test_tool_choice_fallback_on_unsupported_error(self, tmp_path: Path) -> None:
        """Forced tool_choice rejected by provider -> retry with auto and succeed."""
        store = MemoryStore(tmp_path)
        error_resp = LLMResponse(
            content="Error calling LLM: BadRequestError: "
            "The tool_choice parameter does not support being set to required or object",
            finish_reason="error",
            tool_calls=[],
        )
        ok_resp = _make_tool_response(
            history_entry="[2026-01-01] Fallback worked.",
            memory_update="# Memory\nFallback OK.",
        )

        call_log: list[dict] = []

        async def _tracking_chat(**kwargs):
            call_log.append(kwargs)
            return error_resp if len(call_log) == 1 else ok_resp

        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(side_effect=_tracking_chat)
        messages = _make_messages(message_count=60)

        result = await store.consolidate(messages, provider, "test-model")

        assert result is not None
        assert len(call_log) == 2
        assert isinstance(call_log[0]["tool_choice"], dict)
        assert call_log[1]["tool_choice"] == "auto"
        assert "Fallback worked." in store.history_file.read_text()

    @pytest.mark.asyncio
    async def test_tool_choice_fallback_auto_no_tool_call(self, tmp_path: Path) -> None:
        """Forced rejected, auto retry also produces no tool call -> return False."""
        store = MemoryStore(tmp_path)
        error_resp = LLMResponse(
            content="Error: tool_choice must be none or auto",
            finish_reason="error",
            tool_calls=[],
        )
        no_tool_resp = LLMResponse(
            content="Here is a summary.",
            finish_reason="stop",
            tool_calls=[],
        )

        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(side_effect=[error_resp, no_tool_resp])
        messages = _make_messages(message_count=60)

        result = await store.consolidate(messages, provider, "test-model")

        assert result is None
        assert not store.history_file.exists()

    @pytest.mark.asyncio
    async def test_raw_archive_after_consecutive_failures(self, tmp_path: Path) -> None:
        """After 3 consecutive failures, raw-archive messages to recovery sidecar and return True."""
        store = MemoryStore(tmp_path)
        no_tool = LLMResponse(content="No tool call.", finish_reason="stop", tool_calls=[])
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(return_value=no_tool)
        messages = _make_messages(message_count=10)

        assert await store.consolidate(messages, provider, "m") is None
        assert await store.consolidate(messages, provider, "m") is None
        assert await store.consolidate(messages, provider, "m") is None

        assert "⚠️ Synthesis failed" in store.history_file.read_text()
        assert "recovery/" in store.history_file.read_text()
        # Verify the recovery sidecar file was created (filter WAL files)
        recovery_files = list((tmp_path / "memory" / "recovery").glob("raw-*.json"))
        assert len(recovery_files) == 1
        recovery_content = recovery_files[0].read_text()
        assert "msg0" in recovery_content
        assert not store.memory_file.exists()

    @pytest.mark.asyncio
    async def test_raw_archive_counter_resets_on_success(self, tmp_path: Path) -> None:
        """A successful consolidation resets the failure counter."""
        store = MemoryStore(tmp_path)
        no_tool = LLMResponse(content="Nope.", finish_reason="stop", tool_calls=[])
        ok_resp = _make_tool_response(
            history_entry="[2026-01-01] OK.",
            memory_update="# Memory\nOK.",
        )
        messages = _make_messages(message_count=10)

        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(return_value=no_tool)
        assert await store.consolidate(messages, provider, "m") is None
        assert await store.consolidate(messages, provider, "m") is None
        assert store._consecutive_failures == 2

        wal_after_2_failures = list((tmp_path / "memory" / "recovery").glob("wal-*.json"))
        assert len(wal_after_2_failures) == 2  # WAL files persist after failures

        provider.chat_with_retry = AsyncMock(return_value=ok_resp)
        assert await store.consolidate(messages, provider, "m") is not None
        assert store._consecutive_failures == 0
        # The success call's own WAL is cleaned, but prior failure WALs remain (for recovery)
        wal_after_success = list((tmp_path / "memory" / "recovery").glob("wal-*.json"))
        assert len(wal_after_success) == 2  # only the 3rd (success) WAL was cleaned

        provider.chat_with_retry = AsyncMock(return_value=no_tool)
        assert await store.consolidate(messages, provider, "m") is None
        assert store._consecutive_failures == 1
        # One more WAL from the final failure
        wal_after_4th = list((tmp_path / "memory" / "recovery").glob("wal-*.json"))
        assert len(wal_after_4th) == 3
