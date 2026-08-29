"""Tool for consolidating session history."""

from logging import getLogger
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import Address

logger = getLogger(__name__)

if TYPE_CHECKING:
    from nanobot.agent.memory import MemoryConsolidator
    from nanobot.session.manager import SessionManager


class CompactHistoryTool(Tool):
    """
    Consolidates the current session history into a summary.
    The summary is injected into the session, and older messages are archived.
    """

    def __init__(
        self,
        consolidator: "MemoryConsolidator",
        session_manager: "SessionManager",
    ):
        self._consolidator = consolidator
        self._session_manager = session_manager
        self._session_key = "cli:direct"

    def set_context(self, address: Address, session_key: str) -> None:
        """Set the session context for consolidation."""
        self._session_key = session_key

    @property
    def name(self) -> str:
        return "compact_history"

    @property
    def description(self) -> str:
        return "Consolidates session history, keeping the last N messages. Returns the generated summary."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "n_keep": {
                    "type": "integer",
                    "description": "Number of most recent messages to keep. 0 means consolidate all.",
                    "minimum": 0,
                },
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        n_keep = kwargs.get("n_keep", 0)

        session = self._session_manager.get_or_create(self._session_key)

        # Unified dispatch (D4), shared with /compact and the token-budget
        # auto-consolidation: drop [ctx_end : end_idx] where
        #   free drop     — end_idx ≤ SB (already staged, no LLM work)
        #   commit + drop — end_idx >  SB (80% guard + distill fresh [SB:end_idx])
        # ctx_end = end of the pinned tagged context prefix (tag-scan).
        end_idx = max(0, len(session.messages) - n_keep)

        result = await self._consolidator.consolidate_range(session, end_idx)

        if result.error:
            return result.error

        if result.consolidated:
            if n_keep > 0:
                reply = f"✅ Consolidated history, keeping last {n_keep} messages."
            else:
                reply = "✅ Consolidated all history."
            reply += f"\n\n---\n\n{result.summary}"
            return reply

        # Free drop: [ctx_end:end_idx] was already distilled into STAGING.md
        # at stage time — reclaimed without any LLM call.
        reply = f"✅ Dropped {result.dropped} already-staged message(s), keeping last {n_keep}."
        if session.staged_boundary > 0:
            reply += f" ({session.staged_boundary} staged message(s) retained for context.)"
        return reply
