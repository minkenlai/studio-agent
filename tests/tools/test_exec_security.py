"""Tests for exec tool internal URL blocking."""

from __future__ import annotations

import socket
from unittest.mock import patch

import pytest

from nanobot.agent.tools.shell import ExecTool


def _fake_resolve_private(hostname, port, family=0, type_=0):
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("169.254.169.254", 0))]


def _fake_resolve_localhost(hostname, port, family=0, type_=0):
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))]


def _fake_resolve_public(hostname, port, family=0, type_=0):
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]


@pytest.mark.asyncio
async def test_exec_blocks_curl_metadata():
    tool = ExecTool()
    with patch("nanobot.security.network.socket.getaddrinfo", _fake_resolve_private):
        result = await tool.execute(
            command='curl -s -H "Metadata-Flavor: Google" http://169.254.169.254/computeMetadata/v1/'
        )
    assert "Error" in result
    assert "internal" in result.lower() or "private" in result.lower()


@pytest.mark.asyncio
async def test_exec_blocks_wget_localhost():
    tool = ExecTool()
    with patch("nanobot.security.network.socket.getaddrinfo", _fake_resolve_localhost):
        result = await tool.execute(command="wget http://localhost:8080/secret -O /tmp/out")
    assert "Error" in result


@pytest.mark.asyncio
async def test_exec_allows_normal_commands():
    tool = ExecTool(timeout=5)
    result = await tool.execute(command="echo hello")
    assert "hello" in result
    assert "Error" not in result.split("\n")[0]


@pytest.mark.asyncio
async def test_exec_allows_curl_to_public_url():
    """Commands with public URLs should not be blocked by the internal URL check."""
    tool = ExecTool()
    with patch("nanobot.security.network.socket.getaddrinfo", _fake_resolve_public):
        guard_result = tool._guard_command("curl https://example.com/api", "/tmp")
    assert guard_result is None


@pytest.mark.asyncio
async def test_exec_blocks_chained_internal_url():
    """Internal URLs buried in chained commands should still be caught."""
    tool = ExecTool()
    with patch("nanobot.security.network.socket.getaddrinfo", _fake_resolve_private):
        result = await tool.execute(
            command="echo start && curl http://169.254.169.254/latest/meta-data/ && echo done"
        )
    assert "Error" in result


# ── Slash-command whitelist tests ────────────────────────────────────────


@pytest.fixture
def restricted_tool():
    """ExecTool with workspace restriction enabled."""
    return ExecTool(restrict_to_workspace=True, working_dir="/home/nanobot/.nanobot/workspace")


def test_slash_command_not_blocked_as_path(restricted_tool):
    """A bare slash command like '/new' must NOT trigger the path-outside-workspace guard."""
    result = restricted_tool._guard_command('echo "/new"', "/home/nanobot/.nanobot/workspace")
    assert result is None, f"Slash command falsely blocked: {result}"


def test_slash_command_new_in_path_still_blocked(restricted_tool):
    """'/some/path/new' looks like a real path and SHOULD be blocked (outside workspace)."""
    result = restricted_tool._guard_command(
        "cat /some/path/new", "/home/nanobot/.nanobot/workspace"
    )
    assert result is not None, "/some/path/new should be blocked as an absolute path"
    assert "path outside working dir" in (result or "").lower()


def test_slash_command_with_trailing_segment_blocked(restricted_tool):
    """'/new/seg' is a real path (not a command) and SHOULD be blocked."""
    result = restricted_tool._guard_command("ls /new/seg", "/home/nanobot/.nanobot/workspace")
    assert result is not None, "/new/seg should be blocked as an absolute path"


def test_slash_command_restart_in_command(restricted_tool):
    """Bare '/restart' should pass through."""
    result = restricted_tool._guard_command('echo "/restart"', "/home/nanobot/.nanobot/workspace")
    assert result is None


def test_slash_command_halt_in_command(restricted_tool):
    """Bare '/halt' should pass through (case-sensitive)."""
    result = restricted_tool._guard_command('echo "/halt"', "/home/nanobot/.nanobot/workspace")
    assert result is None


def test_slash_command_not_at_token_boundary_still_blocked(restricted_tool):
    """'/some/restart' should NOT match '/restart' and should be blocked."""
    result = restricted_tool._guard_command("cat /some/restart", "/home/nanobot/.nanobot/workspace")
    assert result is not None, "/some/restart should be blocked as an absolute path"


def test_slash_command_with_args_still_blocked(restricted_tool):
    """'/restart --full' extracts as '/restart' only if '--full' is separate token.
    The regex captures '/restart' (space terminates), which is a slash command -> filtered.
    The '--full' is not an absolute path. So this should pass."""
    result = restricted_tool._guard_command(
        'echo "/restart --full"', "/home/nanobot/.nanobot/workspace"
    )
    assert result is None


def test_real_absolute_path_still_blocked(restricted_tool):
    """A real absolute path like /tmp/foo should still be blocked."""
    result = restricted_tool._guard_command("cat /tmp/foo", "/home/nanobot/.nanobot/workspace")
    assert result is not None
    assert "path outside working dir" in (result or "").lower()


def test_workspace_path_not_blocked(restricted_tool):
    """A path inside the workspace should NOT be blocked."""
    result = restricted_tool._guard_command(
        "cat /home/nanobot/.nanobot/workspace/test.py", "/home/nanobot/.nanobot/workspace"
    )
    assert result is None


# ── Benign null-device whitelist tests ───────────────────────────────────


def test_stderr_redirect_to_dev_null_not_blocked(restricted_tool):
    """'cmd 2>/dev/null' must NOT trip the path-outside-workspace guard."""
    result = restricted_tool._guard_command(
        "echo hi 2>/dev/null", "/home/nanobot/.nanobot/workspace"
    )
    assert result is None, f"2>/dev/null falsely blocked: {result}"


def test_stdout_redirect_to_dev_null_not_blocked(restricted_tool):
    """'cmd >/dev/null' must NOT trip the path-outside-workspace guard."""
    result = restricted_tool._guard_command(
        "make build >/dev/null", "/home/nanobot/.nanobot/workspace"
    )
    assert result is None, f">/dev/null falsely blocked: {result}"


@pytest.mark.parametrize(
    "device",
    ["/dev/null", "/dev/zero", "/dev/full", "/dev/tty", "/dev/stdin", "/dev/stdout", "/dev/stderr"],
)
def test_benign_null_devices_not_blocked(restricted_tool, device):
    """Redirects to any whitelisted null device must pass the guard."""
    result = restricted_tool._guard_command(
        f"echo hi 2>{device}", "/home/nanobot/.nanobot/workspace"
    )
    assert result is None, f"{device} falsely blocked: {result}"


def test_real_path_still_blocked_alongside_dev_null(restricted_tool):
    """'cat /tmp/foo 2>/dev/null' — the null device is benign, but /tmp/foo is still blocked."""
    result = restricted_tool._guard_command(
        "cat /tmp/foo 2>/dev/null", "/home/nanobot/.nanobot/workspace"
    )
    assert result is not None, "/tmp/foo should still be blocked even with 2>/dev/null"
    assert "path outside working dir" in (result or "").lower()


def test_dev_device_not_in_whitelist_still_blocked(restricted_tool):
    """'/dev/sda' is a real device path (not a null device) and SHOULD be blocked."""
    result = restricted_tool._guard_command("cat /dev/sda", "/home/nanobot/.nanobot/workspace")
    assert result is not None, "/dev/sda should be blocked as an absolute path"
    assert "path outside working dir" in (result or "").lower()
