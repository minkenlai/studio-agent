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

        # Calculate boundary
        end_idx = max(0, len(session.messages) - n_keep)
        chunk = session.messages[session.last_consolidated : end_idx]

        if not chunk:
            return "Nothing to summarize (session is already fully compacted)."

        # Early guard: if the chunk is too large for the consolidator's context,
        # reject upfront rather than letting the LLM fail and losing data.
        from nanobot.utils.helpers import estimate_message_tokens

        chunk_tokens = sum(estimate_message_tokens(m) for m in chunk)
        consolidator_context = getattr(self._consolidator, "context_window_tokens", None)
        if isinstance(consolidator_context, int) and consolidator_context > 0:
            if chunk_tokens > consolidator_context * 0.8:  # 80% threshold
                logger.warning(
                    "compact_history: chunk (%d tokens) exceeds consolidator context (%d) — skipping",
                    chunk_tokens,
                    consolidator_context,
                )
                return (
                    f"⚠️ Compact failed: {len(chunk)} messages ({chunk_tokens:,} tokens) exceed "
                    f"the summarizer's context window ({consolidator_context:,}). "
                    f"Session history preserved. Consider compacting with n_keep > 0 first."
                )

        # Use the existing consolidation pipeline
        summary = await self._consolidator.archive_messages(chunk)

        # Replace the chunk with the summary and advance pointer.
        # The chunk is removed and replaced at its start position so that
        # get_history() (messages[last_consolidated:]) includes the summary.
        if summary:
            from datetime import datetime

            summary_msg = {
                "role": "assistant",
                "content": summary,
                "timestamp": datetime.now().isoformat(),
                "_compact_summary": True,  # Marker for get_history() to preserve
            }
            # Replace chunk [last_consolidated : end_idx] with summary
            start_idx = session.last_consolidated
            del session.messages[start_idx:end_idx]
            session.messages.insert(start_idx, summary_msg)
            session.last_consolidated = start_idx

            self._session_manager.save(session)

            if n_keep > 0:
                reply = f"✅ Consolidated history, keeping last {n_keep} messages."
            else:
                reply = "✅ Consolidated all history."
            reply += f"\n\n---\n\n{summary}"

            return reply
        else:
            # Summary generation failed — do NOT advance last_consolidated.
            # The messages must stay in the session to avoid data loss.
            logger.warning(
                "compact_history: summary failed for %d messages in session %s — "
                "last_consolidated NOT advanced to prevent data loss",
                len(chunk),
                self._session_key,
            )
            return (
                f"⚠️ Compact failed: summary generation returned no result for {len(chunk)} messages. "
                f"Session history preserved. This may happen if the content exceeds the model's context window."
            )
