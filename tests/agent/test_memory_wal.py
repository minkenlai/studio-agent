"""Tests for Write-Ahead Log (WAL) checkpoint in MemoryStore.consolidate().

Ensures interrupted consolidations (cancellation, exceptions) preserve
raw message data on disk for later recovery.
"""

import asyncio
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.memory import MemoryStore
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _make_messages(message_count: int = 10):
    """Create a list of mock messages."""
    return [
        {"role": "user", "content": f"msg{i}", "timestamp": "2026-01-01 00:00"}
        for i in range(message_count)
    ]


def _make_tool_response(history_entry: str, memory_update: str) -> LLMResponse:
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


class TestWALHelpers:
    """Test the _write_wal, _cleanup_wal, and _cleanup_stale_wals helpers."""

    def test_write_wal_creates_json_file(self, tmp_path: Path) -> None:
        """_write_wal creates a WAL JSON file in the recovery directory."""
        store = MemoryStore(tmp_path)
        messages = _make_messages(5)
        wal_path = store._write_wal(messages)

        assert wal_path.exists()
        assert wal_path.name.startswith("wal-")
        assert wal_path.suffix == ".json"
        assert wal_path.parent == store.recovery_dir

    def test_wal_content_round_trips(self, tmp_path: Path) -> None:
        """WAL file content matches the original messages."""
        import json

        store = MemoryStore(tmp_path)
        messages = [
            {"role": "user", "content": "hello", "timestamp": "2026-01-01 00:00"},
            {"role": "assistant", "content": "world"},
        ]
        wal_path = store._write_wal(messages)

        loaded = json.loads(wal_path.read_text())
        assert loaded == messages

    def test_cleanup_wal_removes_file(self, tmp_path: Path) -> None:
        """_cleanup_wal removes the WAL file after successful consolidation."""
        store = MemoryStore(tmp_path)
        messages = _make_messages(3)
        wal_path = store._write_wal(messages)
        assert wal_path.exists()

        store._cleanup_wal(wal_path)
        assert not wal_path.exists()

    def test_cleanup_wal_noop_if_missing(self, tmp_path: Path) -> None:
        """_cleanup_wal does not raise if the file is already missing."""
        store = MemoryStore(tmp_path)
        nonexistent = store.recovery_dir / "wal-gone.json"
        assert not nonexistent.exists()

        # Should not raise
        store._cleanup_wal(nonexistent)

    def test_cleanup_stale_wals_removes_old_files(self, tmp_path: Path) -> None:
        """_cleanup_stale_wals removes WAL files older than max_age_hours."""
        import json

        store = MemoryStore(tmp_path)

        # Create two WAL files
        old_wal = store.recovery_dir / "wal-20260101_000000_000000.json"
        new_wal = store.recovery_dir / "wal-20260730_000000_000000.json"
        old_wal.write_text(json.dumps([{"role": "user", "content": "old"}]))
        new_wal.write_text(json.dumps([{"role": "user", "content": "new"}]))

        # Make old_wal have an old mtime (48 hours ago)
        old_time = time.time() - 48 * 3600
        os.utime(old_wal, (old_time, old_time))

        cleaned = store._cleanup_stale_wals(store.recovery_dir, max_age_hours=24)
        assert cleaned == 1
        assert not old_wal.exists()
        assert new_wal.exists()

    def test_cleanup_stale_wals_keeps_recent_files(self, tmp_path: Path) -> None:
        """_cleanup_stale_wals keeps WAL files created within the age window."""
        import json

        store = MemoryStore(tmp_path)

        recent_wal = store.recovery_dir / "wal-20260730_000000_000000.json"
        recent_wal.write_text(json.dumps([{"role": "user", "content": "recent"}]))

        cleaned = store._cleanup_stale_wals(store.recovery_dir, max_age_hours=24)
        assert cleaned == 0
        assert recent_wal.exists()

    def test_cleanup_stale_wals_does_not_affect_raw_archive(self, tmp_path: Path) -> None:
        """_cleanup_stale_wals only removes wal-*.json, not raw-*.json files."""
        import json

        store = MemoryStore(tmp_path)

        raw_file = store.recovery_dir / "raw-20260101_000000.json"
        raw_file.write_text(json.dumps([{"role": "user", "content": "archived"}]))
        # Make raw_file very old
        old_time = time.time() - 48 * 3600
        os.utime(raw_file, (old_time, old_time))

        cleaned = store._cleanup_stale_wals(store.recovery_dir, max_age_hours=24)
        assert cleaned == 0
        assert raw_file.exists()


class TestWALIntegration:
    """Integration tests: WAL behavior during consolidation flows."""

    @pytest.mark.asyncio
    async def test_wal_cleaned_on_success(self, tmp_path: Path) -> None:
        """Successful consolidation removes the WAL file."""
        store = MemoryStore(tmp_path)
        ok_resp = _make_tool_response(
            history_entry="[2026-01-01 00:00] Test entry.",
            memory_update="# Memory\nTest update.",
        )
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(return_value=ok_resp)
        provider.generation.max_tokens = 4096
        messages = _make_messages(10)

        result = await store.consolidate(messages, provider, "test-model")
        assert result is not None

        wal_files = list(store.recovery_dir.glob("wal-*.json"))
        assert len(wal_files) == 0

    @pytest.mark.asyncio
    async def test_wal_preserved_on_exception(self, tmp_path: Path) -> None:
        """Failed consolidation preserves the WAL file on disk."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(side_effect=RuntimeError("boom"))
        provider.generation.max_tokens = 4096
        messages = _make_messages(10)

        result = await store.consolidate(messages, provider, "test-model")
        assert result is None

        wal_files = list(store.recovery_dir.glob("wal-*.json"))
        assert len(wal_files) == 1

    @pytest.mark.asyncio
    async def test_wal_preserved_on_cancellation(self, tmp_path: Path) -> None:
        """Cancelled consolidation preserves WAL and re-raises CancelledError."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(side_effect=asyncio.CancelledError())
        provider.generation.max_tokens = 4096
        messages = _make_messages(10)

        with pytest.raises(asyncio.CancelledError):
            await store.consolidate(messages, provider, "test-model")

        wal_files = list(store.recovery_dir.glob("wal-*.json"))
        assert len(wal_files) == 1

    @pytest.mark.asyncio
    async def test_wal_not_created_for_empty_messages(self, tmp_path: Path) -> None:
        """Consolidation with empty messages skips WAL creation."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="", tool_calls=[]))

        result = await store.consolidate([], provider, "test-model")
        assert result is None

        wal_files = list(store.recovery_dir.glob("wal-*.json"))
        assert len(wal_files) == 0

    @pytest.mark.asyncio
    async def test_stale_wals_cleaned_at_start_of_consolidate(self, tmp_path: Path) -> None:
        """Stale WAL files are cleaned at the start of each consolidation."""
        import json

        store = MemoryStore(tmp_path)

        # Create an old stale WAL
        old_wal = store.recovery_dir / "wal-20260101_000000_000000.json"
        old_wal.write_text(json.dumps([{"role": "user", "content": "stale"}]))
        old_time = time.time() - 48 * 3600
        os.utime(old_wal, (old_time, old_time))

        # Trigger a successful consolidation
        ok_resp = _make_tool_response(
            history_entry="[2026-01-01 00:00] Entry.",
            memory_update="# Memory\nUpdate.",
        )
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(return_value=ok_resp)
        provider.generation.max_tokens = 4096

        await store.consolidate(_make_messages(5), provider, "test-model")

        # Old stale WAL should be gone (cleaned at start)
        assert not old_wal.exists()
        # New WAL from this consolidation should also be gone (cleaned on success)
        wal_files = list(store.recovery_dir.glob("wal-*.json"))
        assert len(wal_files) == 0

    @pytest.mark.asyncio
    async def test_wal_and_raw_archive_both_present_after_3_failures(self, tmp_path: Path) -> None:
        """After 3 failures, both WAL file and raw-archive file exist."""
        store = MemoryStore(tmp_path)
        no_tool = LLMResponse(content="No tool call.", finish_reason="stop", tool_calls=[])
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(return_value=no_tool)
        provider.generation.max_tokens = 4096
        messages = _make_messages(10)

        # Run 3 consecutive failures
        assert await store.consolidate(messages, provider, "m") is None
        assert await store.consolidate(messages, provider, "m") is None
        assert await store.consolidate(messages, provider, "m") is None

        wal_files = list(store.recovery_dir.glob("wal-*.json"))
        raw_files = list(store.recovery_dir.glob("raw-*.json"))
        assert len(raw_files) == 1
        # All 3 WAL files persist (failures don't clean WAL, stale cleanup only targets >24h)
        assert len(wal_files) == 3
