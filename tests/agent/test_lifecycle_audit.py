from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from nanobot.agent.loop import AgentLoop


@pytest.mark.asyncio
async def test_audit_lifecycle_no_pending_notification(tmp_path: Path):
    bus = AsyncMock()
    provider = AsyncMock()

    loop = AgentLoop(bus=bus, provider=provider, workspace=tmp_path, model="test-model")

    with patch("nanobot.agent.loop.subprocess.check_output") as mock_subprocess:
        mock_subprocess.side_effect = [b"main\n", b"abcdef1\n"]

        await loop._audit_lifecycle()

    log_file = tmp_path / "logs" / "lifecycle.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")

    assert "START: branch=main sha=abcdef1 pid=" in content

    # Should not send any notification
    bus.publish_outbound.assert_not_called()


@pytest.mark.asyncio
async def test_audit_lifecycle_with_pending_notification(tmp_path: Path):
    bus = AsyncMock()
    provider = AsyncMock()

    loop = AgentLoop(bus=bus, provider=provider, workspace=tmp_path, model="test-model")

    log_file = tmp_path / "logs" / "lifecycle.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(
        "[2026-03-31 12:00:00] SHUTDOWN: RESTART_REQ cli://test_chat\n", encoding="utf-8"
    )

    with patch("nanobot.agent.loop.subprocess.check_output") as mock_subprocess:
        mock_subprocess.side_effect = [b"dev\n", b"1234567\n"]

        await loop._audit_lifecycle()

    content = log_file.read_text(encoding="utf-8")
    assert "START: branch=dev sha=1234567 pid=" in content

    # Should send a notification
    bus.publish_outbound.assert_called_once()
    outbound = bus.publish_outbound.call_args[0][0]
    assert outbound.channel == "cli"
    assert outbound.chat_id == "test_chat"
    assert outbound.message_thread_id is None
    assert "System bootstrapped successfully" in outbound.content
    assert "Branch:" in outbound.content
    assert "dev" in outbound.content
    assert "Commit:" in outbound.content
    assert "1234567" in outbound.content


@pytest.mark.asyncio
async def test_audit_lifecycle_with_thread_id(tmp_path: Path):
    bus = AsyncMock()
    provider = AsyncMock()

    loop = AgentLoop(bus=bus, provider=provider, workspace=tmp_path, model="test-model")

    log_file = tmp_path / "logs" / "lifecycle.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(
        "[2026-03-31 12:00:00] SHUTDOWN: RESTART_REQ tg://123/456\n", encoding="utf-8"
    )

    with patch("nanobot.agent.loop.subprocess.check_output") as mock_subprocess:
        mock_subprocess.side_effect = [b"main\n", b"abcdef1\n"]

        await loop._audit_lifecycle()

    bus.publish_outbound.assert_called_once()
    outbound = bus.publish_outbound.call_args[0][0]
    assert outbound.channel == "tg"
    assert outbound.chat_id == "123"
    assert outbound.message_thread_id == 456


def _loop_with_shutdown(tmp_path: Path) -> AgentLoop:
    """Build an AgentLoop whose lifecycle log contains a SHUTDOWN RESTART_REQ."""
    bus = AsyncMock()
    loop = AgentLoop(bus=bus, provider=AsyncMock(), workspace=tmp_path, model="test-model")
    log_file = tmp_path / "logs" / "lifecycle.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(
        "[2026-03-31 12:00:00] SHUTDOWN: RESTART_REQ cli://test_chat\n", encoding="utf-8"
    )
    return loop


@pytest.mark.asyncio
async def test_audit_lifecycle_recovery_ignores_empty_assets_dir(tmp_path: Path):
    """An empty assets/ subdirectory alone must NOT trigger the recovery warning."""
    loop = _loop_with_shutdown(tmp_path)
    recovery_dir = tmp_path / "memory" / "recovery"
    (recovery_dir / "assets").mkdir(parents=True)

    with patch("nanobot.agent.loop.subprocess.check_output") as mock_subprocess:
        mock_subprocess.side_effect = [b"main\n", b"abcdef1\n"]
        await loop._audit_lifecycle()

    loop.bus.publish_outbound.assert_called_once()
    outbound = loop.bus.publish_outbound.call_args[0][0]
    assert "Pending recovery data found" not in outbound.content


@pytest.mark.asyncio
async def test_audit_lifecycle_recovery_lists_payload_files(tmp_path: Path):
    """Top-level recovery JSON payloads must be flagged and named in the warning."""
    loop = _loop_with_shutdown(tmp_path)
    recovery_dir = tmp_path / "memory" / "recovery"
    (recovery_dir / "assets").mkdir(parents=True)
    (recovery_dir / "raw-20260825_120000.json").write_text("[]", encoding="utf-8")
    (recovery_dir / "wal-20260825_120100_000000.json").write_text("[]", encoding="utf-8")

    with patch("nanobot.agent.loop.subprocess.check_output") as mock_subprocess:
        mock_subprocess.side_effect = [b"main\n", b"abcdef1\n"]
        await loop._audit_lifecycle()

    loop.bus.publish_outbound.assert_called_once()
    outbound = loop.bus.publish_outbound.call_args[0][0]
    assert "Pending recovery data found" in outbound.content
    assert "raw-20260825_120000.json" in outbound.content
    assert "wal-20260825_120100_000000.json" in outbound.content
