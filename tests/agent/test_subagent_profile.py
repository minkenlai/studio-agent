"""Tests for agent_profile propagation through subagent lifecycle."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.bus.events import Address


class TestSpawnToolProfile:
    """Verify SpawnTool correctly stores and forwards agent_profile."""

    def test_init_with_agent_profile(self):
        mgr = MagicMock()
        tool = SpawnTool(manager=mgr, agent_profile="deep")
        assert tool._agent_profile == "deep"

    def test_init_without_agent_profile(self):
        mgr = MagicMock()
        tool = SpawnTool(manager=mgr)
        assert tool._agent_profile is None

    def test_set_agent_profile(self):
        mgr = MagicMock()
        tool = SpawnTool(manager=mgr)
        assert tool._agent_profile is None
        tool.set_agent_profile("fast")
        assert tool._agent_profile == "fast"
        tool.set_agent_profile(None)
        assert tool._agent_profile is None

    @pytest.mark.asyncio
    async def test_execute_forwards_agent_profile(self):
        mgr = MagicMock()
        mgr.spawn = AsyncMock(return_value="Subagent started")
        tool = SpawnTool(manager=mgr, agent_profile="pinned-pro")

        await tool.execute(task="do something")
        mgr.spawn.assert_awaited_once()
        call_kwargs = mgr.spawn.call_args.kwargs
        assert call_kwargs["agent_profile"] == "pinned-pro"


class TestSubagentManagerProfile:
    """Verify SubagentManager passes agent_profile through the lifecycle."""

    @pytest.fixture
    def mgr(self) -> SubagentManager:
        from collections import deque

        mock_bus = MagicMock()
        mock_bus.publish_inbound = AsyncMock()

        mgr = object.__new__(SubagentManager)
        mgr._task_registry = {}
        mgr._completed_ids = deque(maxlen=50)
        mgr.bus = mock_bus
        return mgr

    @pytest.mark.asyncio
    async def test_announce_result_includes_profile_in_metadata(self, mgr):
        origin = Address(channel="tg", segments=("123",))

        await mgr._announce_result(
            task_id="abc1",
            label="test-task",
            task="do a thing",
            result="it worked",
            origin=origin,
            status="ok",
            agent_profile="pinned-pro-agent",
        )

        mgr.bus.publish_inbound.assert_awaited_once()
        msg = mgr.bus.publish_inbound.call_args[0][0]
        assert msg.metadata.get("agent_profile") == "pinned-pro-agent"

    @pytest.mark.asyncio
    async def test_announce_result_no_profile_means_empty_metadata(self, mgr):
        origin = Address(channel="tg", segments=("123",))

        await mgr._announce_result(
            task_id="abc2",
            label="test-task-2",
            task="do another thing",
            result="also worked",
            origin=origin,
            status="ok",
            agent_profile=None,
        )

        mgr.bus.publish_inbound.assert_awaited_once()
        msg = mgr.bus.publish_inbound.call_args[0][0]
        assert "agent_profile" not in msg.metadata
