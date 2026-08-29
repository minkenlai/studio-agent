"""Session management for conversation history."""

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.config.paths import get_legacy_sessions_dir
from nanobot.utils.helpers import ensure_dir, safe_filename


@dataclass
class Session:
    """
    A conversation session.

    Stores messages in JSONL format for easy reading and persistence.

    Pointer model (unified consolidation, see PLAN_unified_consolidation.md):
    - ``staged_boundary``: everything before this index has been distilled into
      STAGING.md (idempotency watermark).
    - Pinned context prefix: leading messages tagged ``_session_context`` (or
      legacy ``_compact_summary``); its length is derived by tag-scan
      (:meth:`context_prefix_len`), not stored.
    Messages before the staged boundary are dropped from the list once
    reclamation runs (consolidation), so the file stays bounded.
    """

    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Index of the first message NOT yet distilled into STAGING.md.
    # Messages before this index have already been staged (idempotency watermark).
    staged_boundary: int = 0

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {"role": role, "content": content, "timestamp": datetime.now().isoformat(), **kwargs}
        self.messages.append(msg)
        self.updated_at = datetime.now()

    @staticmethod
    def context_prefix_len(messages: list[dict[str, Any]]) -> int:
        """Count leading tagged context messages (injected session-context summaries).

        A message counts if it has EITHER tag — all producers set both
        (``_session_context`` and ``_compact_summary``), so in practice the
        two tests agree.

        These messages mark the pinned region produced by drops; they are
        intentional context and stay in the prompt. The count is derived from
        the tags themselves — no stored pointer is needed for this boundary.
        """
        n = 0
        for msg in messages:
            if msg.get("_session_context") or msg.get("_compact_summary"):
                n += 1
            else:
                break
        return n

    @staticmethod
    def _find_legal_start(messages: list[dict[str, Any]]) -> int:
        """Find first index where every tool result has a matching assistant tool_call."""
        declared: set[str] = set()
        start = 0
        for i, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        declared.add(str(tc["id"]))
            elif role == "tool":
                tid = msg.get("tool_call_id")
                if tid and str(tid) not in declared:
                    start = i + 1
                    declared.clear()
                    for prev in messages[start : i + 1]:
                        if prev.get("role") == "assistant":
                            for tc in prev.get("tool_calls") or []:
                                if isinstance(tc, dict) and tc.get("id"):
                                    declared.add(str(tc["id"]))
        return start

    @staticmethod
    def _find_legal_end(messages: list[dict[str, Any]]) -> int:
        """Find exclusive end index that excludes trailing assistant tool_call(s) with no tool results.

        If the last assistant message has tool_calls but no following tool results,
        it's an orphaned back-end — trim back to the last complete exchange.
        Returns len(messages) if the tail is clean.
        """
        end = len(messages)
        # Walk backwards: collect any trailing tool results, then check if the
        # assistant message before them has unresolved tool_calls.
        i = end - 1
        while i >= 0:
            msg = messages[i]
            role = msg.get("role")
            if role == "tool":
                i -= 1
                continue
            if role == "assistant":
                tool_calls = msg.get("tool_calls") or []
                if not tool_calls:
                    break  # clean tail — no dangling calls
                # Collect IDs declared by this assistant message
                declared = {
                    str(tc["id"]) for tc in tool_calls if isinstance(tc, dict) and tc.get("id")
                }
                # Collect IDs resolved by tool messages that follow
                resolved = {
                    str(m.get("tool_call_id"))
                    for m in messages[i + 1 : end]
                    if m.get("role") == "tool" and m.get("tool_call_id")
                }
                if declared - resolved:
                    # Some tool_calls have no matching result — trim from here
                    end = i
                    i -= 1
                    continue
                break  # all calls resolved — tail is clean
            else:
                break  # user or other role — stop
        return end

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """Return session messages for LLM input, aligned to a legal tool-call boundary.

        No pointer skip: everything from index 0 is either pinned context
        (tagged context prefix produced by drops) or live messages (D9).
        """
        active = self.messages
        sliced = active[-max_messages:]

        # Drop leading non-user messages to avoid starting mid-turn when possible.
        # Exception: preserve tagged context messages (_session_context, and
        # legacy _compact_summary) — the pinned prefix is intentional context.
        if sliced:
            for i, msg in enumerate(sliced):
                if msg.get("_session_context") or msg.get("_compact_summary"):
                    # Tagged context prefix found — keep it and everything after.
                    break
                elif msg.get("role") == "user":
                    # First user message found — drop leading non-user msgs.
                    sliced = sliced[i:]
                    break
            # If no user or summary found, leave sliced as-is (original behavior).

        # Some providers reject orphan tool results if the matching assistant
        # tool_calls message fell outside the fixed-size history window.
        start = self._find_legal_start(sliced)
        if start:
            sliced = sliced[start:]

        # Trim trailing assistant tool_calls with no matching tool results.
        end = self._find_legal_end(sliced)
        if end < len(sliced):
            sliced = sliced[:end]

        out: list[dict[str, Any]] = []
        for message in sliced:
            content = message.get("content")
            # Normalise empty-string content to None so providers that require
            # null (e.g. Anthropic) don't reject assistant messages that carry
            # tool_calls but have no text body.
            if content == "":
                content = None
            entry: dict[str, Any] = {"role": message["role"]}
            if content is not None:
                entry["content"] = content
            if "tool_calls" in message:
                # Strip non-standard keys (e.g. extra_content with Google
                # thought_signature) that Anthropic and other providers reject.
                tool_call_keys = {"id", "type", "function"}
                entry["tool_calls"] = [
                    {k: v for k, v in tc.items() if k in tool_call_keys}
                    for tc in message["tool_calls"]
                ]
            for key in ("tool_call_id", "name"):
                if key in message:
                    entry[key] = message[key]
            out.append(entry)
        return out

    def clear(self) -> None:
        """Clear all messages and reset session to initial state."""
        self.messages = []
        self.staged_boundary = 0
        self.updated_at = datetime.now()

    def retain_recent_legal_suffix(self, max_messages: int) -> None:
        """Keep a legal recent suffix, mirroring get_history boundary rules."""
        if max_messages <= 0:
            self.clear()
            return
        if len(self.messages) <= max_messages:
            return

        start_idx = max(0, len(self.messages) - max_messages)

        # If the cutoff lands mid-turn, extend backward to the nearest user turn.
        while start_idx > 0 and self.messages[start_idx].get("role") != "user":
            start_idx -= 1

        retained = self.messages[start_idx:]

        # Mirror get_history(): avoid persisting orphan tool results at the front.
        start = self._find_legal_start(retained)
        if start:
            retained = retained[start:]

        # Mirror get_history(): trim trailing orphaned tool_calls from the back.
        end = self._find_legal_end(retained)
        if end < len(retained):
            retained = retained[:end]

        dropped = len(self.messages) - len(retained)
        self.messages = retained
        self.staged_boundary = max(0, self.staged_boundary - dropped)
        self.updated_at = datetime.now()


class SessionManager:
    """
    Manages conversation sessions.

    Sessions are stored as JSONL files in the sessions directory.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(self.workspace / "sessions")
        self.legacy_sessions_dir = get_legacy_sessions_dir()
        self._cache: dict[str, Session] = {}

    def _get_session_path(self, key: str) -> Path:
        """Get the file path for a session."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"

    def _get_legacy_session_path(self, key: str) -> Path:
        """Legacy global session path (~/.nanobot/sessions/)."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.legacy_sessions_dir / f"{safe_key}.jsonl"

    def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.

        Args:
            key: Session key (usually channel:chat_id).

        Returns:
            The session.
        """
        from loguru import logger

        logger.info(f"GET OR CREATE {key}, cache: {list(self._cache.keys())}")
        if key in self._cache:
            return self._cache[key]

        session = self._load(key)
        logger.info(f"LOADED {key}: {session is not None}")
        if session is None:
            session = Session(key=key)

        self._cache[key] = session
        return session

    def _load(self, key: str) -> Session | None:
        """Load a session from disk."""
        path = self._get_session_path(key)
        if not path.exists():
            legacy_path = self._get_legacy_session_path(key)
            if legacy_path.exists():
                try:
                    shutil.move(str(legacy_path), str(path))
                    logger.info("Migrated session {} from legacy path", key)
                except Exception:
                    logger.exception("Failed to migrate session {}", key)

        if not path.exists():
            return None

        try:
            messages = []
            metadata = {}
            created_at = None
            staged_boundary = 0

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)

                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = (
                            datetime.fromisoformat(data["created_at"])
                            if data.get("created_at")
                            else None
                        )
                        # Migration (D10): sessions written before unification
                        # only have last_consolidated. Its staged prefix maps
                        # directly onto staged_boundary — those messages were
                        # distilled but never dropped.
                        staged_boundary = data.get(
                            "staged_boundary", data.get("last_consolidated", 0)
                        )
                    else:
                        messages.append(data)

            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                metadata=metadata,
                staged_boundary=staged_boundary,
            )
        except Exception as e:
            logger.warning("Failed to load session {}: {}", key, e)
            return None

    def save(self, session: Session) -> None:
        """Save a session to disk."""
        path = self._get_session_path(session.key)

        with open(path, "w", encoding="utf-8") as f:
            metadata_line = {
                "_type": "metadata",
                "key": session.key,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
                "staged_boundary": session.staged_boundary,
            }
            f.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")
            for msg in session.messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        self._cache[session.key] = session

    def invalidate(self, key: str) -> None:
        """Remove a session from the in-memory cache."""
        self._cache.pop(key, None)

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.

        Returns:
            List of session info dicts.
        """
        sessions = []

        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                # Read just the metadata line
                with open(path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            key = data.get("key") or path.stem.replace("_", ":", 1)
                            sessions.append(
                                {
                                    "key": key,
                                    "created_at": data.get("created_at"),
                                    "updated_at": data.get("updated_at"),
                                    "path": str(path),
                                }
                            )
            except Exception:
                continue

        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
