"""Tool for consolidating session history."""

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import Address

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

        # Calculate boundary
        end_idx = max(0, len(session.messages) - n_keep)
        chunk = session.messages[session.last_consolidated : end_idx]

        if not chunk:
            return "Nothing to summarize (session is already fully compacted)."

        # Use the existing consolidation pipeline
        summary = await self._consolidator.archive_messages(chunk)

        # Inject summary and advance pointer
        if summary:
            from datetime import datetime

            summary_msg = {
                "role": "assistant",
                "content": summary,
                "timestamp": datetime.now().isoformat(),
            }
            session.messages.insert(end_idx, summary_msg)
            session.last_consolidated = end_idx

        self._session_manager.save(session)

        if n_keep > 0:
            reply = f"✅ Consolidated history, keeping last {n_keep} messages."
        else:
            reply = "✅ Consolidated all history."

        if summary:
            reply += f"\n\n---\n\n{summary}"

        return reply
