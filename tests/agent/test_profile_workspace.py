"""Tests for profile workspace resolution (Option B) and subagent wiring.

A profile ``workspace`` sets the subagent's *working directory* (where
relative paths and ``exec`` start) — NOT its sandbox. Containment stays at
the operator root, and the clamp is framework-enforced so an agent-writable
overlay can never move the boundary.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.subagent import (
    SubagentManager,
    explicit_profile_workspace,
    resolve_profile_workspace,
    under_root,
)
from nanobot.config.schema import AgentDefaults


class FakeAgents:
    """Mimics AgentsConfig.get_agent for resolver unit tests."""

    def __init__(self, profiles: dict[str, AgentDefaults]):
        self._profiles = profiles

    def get_agent(self, name: str) -> AgentDefaults:
        if name == "defaults":
            return AgentDefaults.model_validate({"model": "fake-model"})
        if name not in self._profiles:
            raise ValueError(f"Agent '{name}' is not configured.")
        return self._profiles[name]


def _agents(**profiles) -> FakeAgents:
    return FakeAgents(
        {
            name: AgentDefaults.model_validate({"model": "fake-model", **cfg})
            for name, cfg in profiles.items()
        }
    )


class TestExplicitProfileWorkspace:
    """Override detection must use model_fields_set, not default comparison."""

    def test_no_profile_or_defaults_is_not_an_override(self):
        agents = _agents()
        assert explicit_profile_workspace(agents, None) is None
        assert explicit_profile_workspace(agents, "defaults") is None
        # Unknown profile: get_agent raises → treated as no override
        assert explicit_profile_workspace(agents, "nonexistent") is None

    def test_profile_explicitly_sets_workspace(self):
        agents = _agents(experimental={"workspace": "ws/experimental"})
        assert explicit_profile_workspace(agents, "experimental") == Path("ws/experimental")

    def test_profile_set_to_default_value_still_counts_as_override(self):
        """A profile that explicitly pins the schema default value still set
        the field (model_fields_set), so it goes through the containment
        check like any other override."""
        agents = _agents(pin={"workspace": "~/.nanobot/workspace"})
        assert (
            explicit_profile_workspace(agents, "pin") == Path("~/.nanobot/workspace").expanduser()
        )


class TestResolveProfileWorkspace:
    """Relative = root-relative; resolve() before the containment check."""

    def test_no_profile_returns_root(self, tmp_path):
        ws, oor = resolve_profile_workspace(tmp_path / "ws", _agents(), None)
        assert ws == tmp_path / "ws"
        assert oor is False

    def test_defaults_profile_returns_root(self, tmp_path):
        ws, oor = resolve_profile_workspace(tmp_path / "ws", _agents(), "defaults")
        assert ws == tmp_path / "ws"
        assert oor is False

    def test_profile_without_field_returns_root(self, tmp_path):
        agents = _agents(experimental={"description": "no workspace"})
        ws, oor = resolve_profile_workspace(tmp_path / "ws", agents, "experimental")
        assert ws == tmp_path / "ws"
        assert oor is False

    def test_relative_workspace_is_root_relative(self, tmp_path):
        root = tmp_path / "ws"
        agents = _agents(experimental={"workspace": "ws/experimental"})
        ws, oor = resolve_profile_workspace(root, agents, "experimental")
        assert ws == (root / "ws" / "experimental").resolve()
        assert oor is False

    def test_absolute_workspace_inside_root(self, tmp_path):
        root = tmp_path / "ws"
        agents = _agents(scratch={"workspace": str(root / "scratch")})
        ws, oor = resolve_profile_workspace(root, agents, "scratch")
        assert ws == (root / "scratch").resolve()
        assert oor is False

    def test_absolute_workspace_outside_root_is_rejected(self, tmp_path):
        agents = _agents(evil={"workspace": str(tmp_path / "elsewhere")})
        ws, oor = resolve_profile_workspace(tmp_path / "ws", agents, "evil")
        assert ws is None
        assert oor is True

    def test_dotdot_traversal_is_rejected(self, tmp_path):
        root = tmp_path / "ws"
        (root / "ws").mkdir(parents=True)
        agents = _agents(tricky={"workspace": "ws/../../etc"})
        ws, oor = resolve_profile_workspace(root, agents, "tricky")
        assert ws is None
        assert oor is True

    def test_dotdot_staying_inside_root_is_allowed(self, tmp_path):
        root = tmp_path / "ws"
        (root / "a" / "b").mkdir(parents=True)
        agents = _agents(bounce={"workspace": "a/b/../c"})
        ws, oor = resolve_profile_workspace(root, agents, "bounce")
        assert ws == (root / "a" / "c").resolve()
        assert oor is False

    def test_symlink_escape_is_rejected(self, tmp_path):
        root = tmp_path / "ws"
        outside = tmp_path / "outside"
        root.mkdir()
        outside.mkdir()
        (root / "link").symlink_to(outside)
        agents = _agents(sneaky={"workspace": "link"})
        ws, oor = resolve_profile_workspace(root, agents, "sneaky")
        assert ws is None
        assert oor is True

    def test_workspace_equal_to_root_is_allowed(self, tmp_path):
        root = tmp_path / "ws"
        agents = _agents(full={"workspace": str(root)})
        ws, oor = resolve_profile_workspace(root, agents, "full")
        assert ws == root.resolve()
        assert oor is False


class TestUnderRoot:
    def test_equal(self, tmp_path):
        p = tmp_path / "a"
        assert under_root(p, p)

    def test_nested(self, tmp_path):
        assert under_root(tmp_path / "a" / "b", tmp_path)

    def test_sibling(self, tmp_path):
        assert not under_root(tmp_path / "b", tmp_path / "a")

    def test_prefix_string_trap(self, tmp_path):
        """'workspace-evil' must NOT count as under 'workspace'."""
        (tmp_path / "workspace").mkdir()
        assert not under_root(tmp_path / "workspace-evil", tmp_path / "workspace")


# ---------------------------------------------------------------------------
# SubagentManager wiring
# ---------------------------------------------------------------------------


class TestSubagentWiring:
    """The manager must split working dir from containment root."""

    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        root = tmp_path / "ws"
        (root / "ws" / "experimental").mkdir(parents=True)
        (root / "logs").mkdir()

        manager = object.__new__(SubagentManager)
        from collections import deque

        manager.bus = MagicMock()
        manager.bus.publish_inbound = AsyncMock()
        manager.config = MagicMock()
        manager.config.agents = _agents(
            experimental={"workspace": "ws/experimental"},
            plain={"description": "no workspace"},
            evil={"workspace": "/etc"},
        )
        manager.workspace = root
        manager.restrict_to_workspace = True
        manager._task_registry = {}
        manager._completed_ids = deque(maxlen=50)
        manager._pending_gates = {}
        manager._queued_gates = {}
        manager._session_tasks = {}
        manager._running_tasks = {}
        manager.web_search_config = MagicMock()
        manager.web_proxy = None
        manager.exec_config = MagicMock(timeout=60)

        # Capture per-tool constructor kwargs; stub every tool class the
        # manager registers.
        mocks = {}
        for cls_name in (
            "ReadFileTool",
            "WriteFileTool",
            "EditFileTool",
            "ListDirTool",
            "SearchFilesTool",
            "ExecTool",
            "WebSearchTool",
            "WebFetchTool",
            "SpawnTool",
        ):
            mocks[cls_name] = MagicMock(name=cls_name)
            monkeypatch.setattr(f"nanobot.agent.subagent.{cls_name}", mocks[cls_name])

        # Runner completes trivially.
        result = MagicMock()
        result.stop_reason = "final"
        result.final_content = "done"
        runner = MagicMock()
        runner.run = AsyncMock(return_value=result)
        manager.registry = MagicMock()
        manager.registry.get_runner = MagicMock(return_value=runner)
        manager.runner = runner

        return manager, mocks, root, runner

    async def _spawn(self, manager, profile):
        from nanobot.bus.events import Address

        origin = Address(channel="cli", segments=("direct",))
        await manager.spawn(task="t", agent=profile, origin=origin, session_key="s1")
        rid = list(manager._running_tasks)[0]
        # The background task blocks on the iteration gate first.
        manager.release_all_gates()
        await manager._running_tasks[rid]
        return rid

    @pytest.mark.asyncio
    async def test_profile_workspace_splits_working_dir_and_root(self, env):
        manager, mocks, root, runner = env
        await self._spawn(manager, "experimental")

        exp = (root / "ws" / "experimental").resolve()
        # File tools: working dir = profile ws, boundary = root
        assert mocks["ReadFileTool"].call_args.kwargs["workspace"] == exp
        assert mocks["ReadFileTool"].call_args.kwargs["allowed_dir"] == root
        assert mocks["WriteFileTool"].call_args.kwargs["workspace"] == exp
        assert mocks["EditFileTool"].call_args.kwargs["workspace"] == exp
        assert mocks["ListDirTool"].call_args.kwargs["workspace"] == exp
        assert mocks["SearchFilesTool"].call_args.kwargs["workspace"] == exp
        # Exec: cwd in profile ws, boundary at root
        assert mocks["ExecTool"].call_args.kwargs["working_dir"] == str(exp)
        assert mocks["ExecTool"].call_args.kwargs["containment_root"] == str(root)
        runner.run.assert_awaited_once()
        rec = list(manager._task_registry.values())[0]
        assert rec.status == "done"

    @pytest.mark.asyncio
    async def test_no_profile_override_keeps_legacy_wiring(self, env):
        manager, mocks, root, _ = env
        await self._spawn(manager, "plain")
        assert mocks["ReadFileTool"].call_args.kwargs["workspace"] == root
        assert mocks["ReadFileTool"].call_args.kwargs["allowed_dir"] == root
        assert mocks["ExecTool"].call_args.kwargs["working_dir"] == str(root)
        assert mocks["ExecTool"].call_args.kwargs["containment_root"] == str(root)

    @pytest.mark.asyncio
    async def test_no_agent_at_all_keeps_root(self, env):
        manager, mocks, root, _ = env
        await self._spawn(manager, None)
        assert mocks["ExecTool"].call_args.kwargs["working_dir"] == str(root)

    @pytest.mark.asyncio
    async def test_out_of_root_profile_is_rejected(self, env):
        manager, mocks, root, runner = env
        await self._spawn(manager, "evil")
        # Runner must never run; the spawn is refused with an error announce.
        runner.run.assert_not_awaited()
        mocks["ReadFileTool"].assert_not_called()
        mocks["ExecTool"].assert_not_called()
        manager.bus.publish_inbound.assert_awaited_once()
        msg = manager.bus.publish_inbound.call_args[0][0]
        assert "outside the sandbox root" in msg.content
        rec = list(manager._task_registry.values())[0]
        assert rec.status == "error"
