"""Spawn tool for creating background subagents."""

from typing import TYPE_CHECKING, Any, Literal

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import Address

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


class SpawnTool(Tool):
    """Tool to spawn a subagent for background task execution."""

    def __init__(
        self,
        manager: "SubagentManager",
        agent_profile: str | None = None,
    ):
        self._manager = manager
        self._origin = Address(channel="cli", segments=("direct",))
        self._session_key = "cli:direct"
        self._agent_profile = agent_profile

    def set_context(self, address: Address, session_key: str) -> None:
        """Set the origin context for subagent announcements."""
        self._origin = address
        self._session_key = session_key

    def set_agent_profile(self, profile: str | None) -> None:
        """Set the agent profile to propagate when subagents report results."""
        self._agent_profile = profile

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a subagent to handle a task in the background. "
            "Use this for complex or time-consuming tasks that can run independently. "
            "The subagent will complete the task and report back when done. "
            "Subagents automatically inherit the core persona (SOUL.md), user preferences (USER.md), "
            "and operational guardrails (SUBAGENT.md), so you don't need to repeat these in the task. "
            "For deliverables or existing projects, inspect the workspace first "
            "and use a dedicated subdirectory when helpful."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        # Build agent options if possible
        agent_desc = (
            "Optional agent profile to use (e.g., 'deep', 'fast'). Overrides the default agent."
        )
        try:
            agents = self._manager.config.agents
            options = []
            # Check defaults
            options.append(f"- default: {agents.defaults.model}")
            if agents.defaults.description:
                options[-1] += f" ({agents.defaults.description})"

            # Check extra agents
            for key in agents.model_extra or {}:
                try:
                    cfg = agents.get_agent(key)
                    opt = f"- {key}: {cfg.model}"
                    if cfg.description:
                        opt += f" ({cfg.description})"
                    options.append(opt)
                except Exception:
                    continue

            if options:
                agent_desc += "\nAvailable agents:\n" + "\n".join(options)
        except Exception:
            pass

        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task for the subagent to complete",
                },
                "label": {
                    "type": "string",
                    "description": "Optional short label for the task (for display)",
                },
                "agent": {
                    "type": "string",
                    "description": agent_desc,
                },
                "context_scope": {
                    "type": "string",
                    "enum": ["full", "technical", "minimal"],
                    "description": (
                        "How much context to give the subagent. "
                        "'full' includes SOUL.md, USER.md, and SUBAGENT.md (default). "
                        "'technical' includes only SUBAGENT.md. "
                        "'minimal' includes no persona files."
                    ),
                },
                "included_skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of skill names to include for the subagent. "
                        "If omitted or null, all available skills are included."
                    ),
                },
            },
            "required": ["task"],
        }

    async def execute(
        self,
        task: str,
        label: str | None = None,
        agent: str | None = None,
        context_scope: Literal["full", "technical", "minimal"] = "full",
        included_skills: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Spawn a subagent to execute the given task."""
        return await self._manager.spawn(
            task=task,
            label=label,
            agent=agent,
            origin=self._origin,
            session_key=self._session_key,
            context_scope=context_scope,
            included_skills=included_skills,
            agent_profile=self._agent_profile,
        )
