"""Memory system for persistent agent memory."""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import os
import re
import tempfile
import time
import weakref
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from loguru import logger

from nanobot.services.transcription import TranscriptionService
from nanobot.session.manager import Session
from nanobot.utils.helpers import (
    _extract_text,
    ensure_dir,
    estimate_message_tokens,
    estimate_prompt_tokens_chain,
)

if TYPE_CHECKING:
    from nanobot.providers.base import LLMClient
    from nanobot.session.manager import Session, SessionManager


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "New facts, updates, or changes to be added to long-term memory. Use a bulleted list. Do NOT return the entire memory file.",
                    },
                    "session_context": {
                        "type": "string",
                        "description": "A brief one-sentence summary of the current session's ongoing context "
                        "and working state. Use this to preserve continuity after older messages are dropped. "
                        "Example: 'Implementing unified session consolidation, Phase 1 Task 1: adding session_context field.'",
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]


def _ensure_text(value: Any) -> str:
    """Normalize tool-call payload values to text for file storage."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _normalize_save_memory_args(args: Any) -> dict[str, Any] | None:
    """Normalize provider tool-call arguments to the expected dict shape."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            # Salvage: find the first JSON object embedded in conversational text
            match = re.search(r"(\{.*?\})", args, re.DOTALL)
            if match:
                try:
                    args = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            if isinstance(args, str):
                return None
    if isinstance(args, list):
        return args[0] if args and isinstance(args[0], dict) else None
    return args if isinstance(args, dict) else None


_TOOL_CHOICE_ERROR_MARKERS = (
    "tool_choice",
    "toolchoice",
    "does not support",
    'should be ["none", "auto"]',
)


def _is_tool_choice_unsupported(content: str | None) -> bool:
    """Detect provider errors caused by forced tool_choice being unsupported."""
    text = (content or "").lower()
    return any(m in text for m in _TOOL_CHOICE_ERROR_MARKERS)


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    _MAX_FAILURES_BEFORE_RAW_ARCHIVE = 3

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self.staging_file = self.memory_dir / "STAGING.md"
        self.recovery_dir = ensure_dir(self.memory_dir / "recovery")
        self._consecutive_failures = 0

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def read_staging(self) -> str:
        if self.staging_file.exists():
            return self.staging_file.read_text(encoding="utf-8")
        return ""

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        """Return the memory block for the system prompt: long-term + staging.

        STAGING.md holds consolidated knowledge that hasn't been synthesized into
        MEMORY.md yet. Including it here makes consolidation output immediately
        visible (and cross-session visible), instead of waiting for synthesis.
        """
        long_term = self.read_long_term()
        staging = self.read_staging().strip()

        parts = []
        if long_term:
            parts.append(f"## Long-term Memory\n{long_term}")
        if staging:
            parts.append(f"## Staging Notes (pending synthesis)\n{staging}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        lines = []
        for message in messages:
            content = message.get("content")
            if not content:
                continue
            tools = (
                f" [tools: {', '.join(message['tools_used'])}]" if message.get("tools_used") else ""
            )
            text = _extract_text(content)
            lines.append(
                f"[{message.get('timestamp', '?')[:16]}] {message['role'].upper()}{tools}: {text}"
            )
        return "\n".join(lines)

    @staticmethod
    def _extract_text(content) -> str:
        """Extract text from message content, handling both string and list-of-blocks formats."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(parts) if parts else ""
        return str(content)

    def _write_wal(self, messages: list[dict]) -> Path:
        """Write-ahead log: dump messages to a temp recovery file before the LLM call."""
        now = datetime.now()
        ts_file = now.strftime("%Y%m%d_%H%M%S_%f")
        filename = f"wal-{ts_file}.json"
        path = self.recovery_dir / filename
        path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug("WAL checkpoint written to {}", path)
        return path

    def _cleanup_wal(self, path: Path) -> None:
        """Remove a WAL checkpoint file after successful consolidation."""
        try:
            if path.exists():
                path.unlink()
                logger.debug("WAL checkpoint cleaned up: {}", path)
        except Exception:
            logger.warning("Failed to cleanup WAL checkpoint {}", path)

    @staticmethod
    def _cleanup_stale_wals(recovery_dir: Path, max_age_hours: int = 24) -> int:
        """Remove WAL files older than max_age_hours."""
        cleaned = 0
        cutoff = time.time() - (max_age_hours * 3600)
        for path in recovery_dir.glob("wal-*.json"):
            if path.stat().st_mtime < cutoff:
                try:
                    path.unlink()
                    cleaned += 1
                except Exception:
                    pass
        if cleaned:
            logger.info("Cleaned up {} stale WAL file(s)", cleaned)
        return cleaned

    async def consolidate(
        self,
        messages: list[dict],
        provider: LLMClient,
        model: str,
    ) -> tuple[str, str | None] | None:
        """Consolidate the provided message chunk into MEMORY.md + HISTORY.md.

        Returns a tuple of (history_entry, session_context) on success, or None on failure.
        session_context may be None if the LLM did not provide one.
        """
        if not messages:
            return None

        current_memory = self.read_long_term()
        staging_content = ""
        if self.staging_file.exists():
            staging_content = self.staging_file.read_text(encoding="utf-8").strip()

        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

## Current Long-term Memory
{current_memory or "(empty)"}

## Pending Staging Notes
{staging_content or "(none)"}

## Conversation to Process
{self._format_messages(messages)}

## Guidance
- **history_entry**: Start with [YYYY-MM-DD HH:MM]. Include specific details (names, decisions, technical terms) that are useful for future grep searches.
- **memory_update**: New facts or changes as a bulleted list. Do NOT return the entire memory file.
- **session_context**: Provide a one-sentence summary of what the user is actively working on RIGHT NOW. This will be injected back into the conversation to maintain continuity after old messages are dropped. Focus on current goals and blockers, not just 'user said hello.' Example: 'Implementing unified session consolidation; Phase 2 Task 1 adds drop-and-inject logic for session context continuity.'"""

        chat_messages = [
            {
                "role": "system",
                "content": "You are a memory consolidation agent. Call the save_memory tool with your consolidation of the conversation.",
            },
            {"role": "user", "content": prompt},
        ]

        MemoryStore._cleanup_stale_wals(self.recovery_dir)
        wal_path = self._write_wal(messages)

        try:
            forced = {"type": "function", "function": {"name": "save_memory"}}
            response = await provider.chat_with_retry(
                messages=chat_messages,
                tools=_SAVE_MEMORY_TOOL,
                model=model,
                tool_choice=forced,
                max_tokens=8192,
            )

            if response.finish_reason == "error" and _is_tool_choice_unsupported(response.content):
                logger.warning("Forced tool_choice unsupported, retrying with auto")
                response = await provider.chat_with_retry(
                    messages=chat_messages,
                    tools=_SAVE_MEMORY_TOOL,
                    model=model,
                    tool_choice="auto",
                    max_tokens=8192,
                )

            if response.finish_reason == "length":
                logger.error(
                    "Memory consolidation TRUNCATED (finish_reason=length). "
                    "MEMORY.md update or history_entry was too long for the max_tokens limit ({}). "
                    "Aborting to avoid corrupting memory files.",
                    provider.generation.max_tokens,
                )
                self._fail_or_raw_archive(messages)
                return None

            if not response.has_tool_calls:
                # Last ditch effort: try to parse the content as a tool-call arguments string
                if response.content:
                    args = _normalize_save_memory_args(response.content)
                    if not args:
                        self._fail_or_raw_archive(messages)
                        return None
                else:
                    self._fail_or_raw_archive(messages)
                    return None
            else:
                args = _normalize_save_memory_args(response.tool_calls[0].arguments)

            if args is None:
                logger.warning("Memory consolidation: unexpected save_memory arguments")
                self._fail_or_raw_archive(messages)
                return None

            if (
                "history_entry" not in args
                or "memory_update" not in args
                or args["history_entry"] is None
                or args["memory_update"] is None
            ):
                logger.warning("Memory consolidation: save_memory payload missing required fields")
                self._fail_or_raw_archive(messages)
                return None

            entry = _ensure_text(args["history_entry"]).strip()
            update = _ensure_text(args["memory_update"])
            session_ctx = args.get("session_context") if "session_context" in args else None
            if session_ctx is not None:
                session_ctx = _ensure_text(session_ctx).strip() or None

            if not entry:
                logger.warning("Memory consolidation: history_entry is empty")
                self._fail_or_raw_archive(messages)
                return None

            self.append_history(entry)
            if update != current_memory:
                # Instead of overwriting MEMORY.md, we append these updates to STAGING.md
                # to be processed by the memory-synthesize skill later.
                staging_path = self.memory_dir / "STAGING.md"
                with open(staging_path, "a", encoding="utf-8") as f:
                    f.write(f"\n## Consolidation Update: {datetime.now().isoformat()}\n{update}\n")
                logger.info("Memory updates staged to STAGING.md")

            self._consecutive_failures = 0
            logger.info("Memory consolidation done for {} messages", len(messages))
            self._cleanup_wal(wal_path)
            return (entry, session_ctx)
        except asyncio.CancelledError:
            logger.warning("Memory consolidation cancelled (WAL preserved at {})", wal_path)
            self._fail_or_raw_archive(messages)
            raise
        except Exception:
            logger.exception("Memory consolidation failed")
            self._fail_or_raw_archive(messages)
            return None

    def _fail_or_raw_archive(self, messages: list[dict]) -> bool:
        """Increment failure count; after threshold, raw-archive messages and return True."""
        self._consecutive_failures += 1
        if self._consecutive_failures < self._MAX_FAILURES_BEFORE_RAW_ARCHIVE:
            return False
        self._raw_archive(messages)
        self._consecutive_failures = 0
        return True

    def _raw_archive(self, messages: list[dict]) -> None:
        """Fallback: dump raw messages to a recovery sidecar file.

        Extracts large audio blocks to separate binary files to prevent JSON bloat.
        """
        now = datetime.now()
        ts_human = now.strftime("%Y-%m-%d %H:%M")
        ts_file = now.strftime("%Y%m%d_%H%M%S")
        filename = f"raw-{ts_file}.txt"
        recovery_path = self.recovery_dir / filename

        # Deep copy messages to avoid mutating the original session history
        archived_messages = copy.deepcopy(messages)

        # Assets dir is created lazily — only when audio extraction actually needs it,
        # so an empty assets/ subdir doesn't linger in recovery/
        assets_dir: Path | None = None

        for msg in archived_messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict) or block.get("type") != "audio":
                    continue

                # Extract data: "data:<mime>;base64,<payload>"
                data_val = block.get("data", "")
                if not data_val or not data_val.startswith("data:"):
                    continue

                try:
                    # Split mime and base64 payload
                    if "," not in data_val:
                        # Try to just strip 'data:' if comma is missing
                        b64_payload = data_val.replace("data:", "")
                    else:
                        header, b64_payload = data_val.split(",", 1)

                    # Save binary file
                    if assets_dir is None:
                        assets_dir = ensure_dir(self.recovery_dir / "assets")
                    asset_name = f"asset_{uuid4().hex[:12]}.bin"
                    asset_path = assets_dir / asset_name

                    binary_data = base64.b64decode(b64_payload)
                    asset_path.write_bytes(binary_data)

                    # Replace block content with a pointer
                    block["data"] = f"FILE:{asset_path.name}"
                    block["recovery_asset"] = asset_path.name
                except Exception as e:
                    logger.error("Failed to extract audio asset during raw_archive: {}", e)

        # We use JSON for the recovery file now to preserve the structure with pointers
        recovery_path = recovery_path.with_suffix(".json")
        recovery_path.write_text(
            json.dumps(archived_messages, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        self.append_history(
            f"[{ts_human}] ⚠️ Synthesis failed; raw data preserved in recovery/{recovery_path.name}"
        )
        logger.warning(
            "Memory consolidation degraded: raw-archived {} messages to {}",
            len(messages),
            recovery_path,
        )


@dataclass
class DropResult:
    """Outcome of a unified drop dispatch (see ``MemoryConsolidator._dispatch_drop``)."""

    dropped: int = 0  # messages physically removed
    consolidated: bool = False  # LLM distillation ran (commit + drop case)
    summary: str | None = None  # history entry from the distillation
    error: str | None = None  # user-facing error (guard / failure); nothing dropped
    staged_boundary: int = 0  # resulting staged boundary (for caller reply text)


class MemoryConsolidator:
    """Owns consolidation policy, locking, and session offset updates."""

    _MAX_CONSOLIDATION_ROUNDS = 5
    _SAFETY_BUFFER = 1024  # extra headroom for tokenizer estimation drift
    _transcription_service: TranscriptionService | None = None

    def __init__(
        self,
        workspace: Path,
        provider: LLMClient,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        max_completion_tokens: int = 4096,
    ):
        self.store = MemoryStore(workspace)
        self.provider = provider
        self.model = model
        self.sessions = sessions

        # Override context_window_tokens with provider's max if available
        provider_context_max = (
            getattr(provider.generation, "context_max", None)
            if hasattr(provider, "generation")
            else None
        )
        self.context_window_tokens = (
            provider_context_max if provider_context_max is not None else context_window_tokens
        )

        self.max_completion_tokens = max_completion_tokens
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
        # Transient: stores the last session_context from a consolidation
        # so Phase 2 can inject it as a summary message after dropping old msgs.
        self._last_session_context: str | None = None

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    async def consolidate_range(self, session: Session, end_idx: int) -> DropResult:
        """Public entry: drop ``[ctx_end : end_idx]`` under the session lock.

        Used by the ``/compact`` command and the ``compact_history`` tool.
        The automatic token-budget path (``maybe_consolidate_by_tokens``)
        already holds the lock, so it calls ``_dispatch_drop`` directly.
        """
        async with self.get_lock(session.key):
            return await self._dispatch_drop(session, end_idx)

    async def _dispatch_drop(self, session: Session, end_idx: int) -> DropResult:
        """Core of the unified drop dispatch (D4). Assumes the session lock is held.

        3-case dispatch keyed on where ``end_idx`` falls relative to the
        staged boundary (SB):

        - free drop     — end_idx <= SB: physical delete, SB shift, no LLM
        - commit + drop — end_idx >  SB: distill fresh [SB:end_idx] first,
                          then drop [ctx_end:end_idx] together
        - nothing to do — empty chunk

        Never drops messages whose distillation failed (data-loss guard).
        At most one tagged context message exists in the leading pinned run:
        on commit + drop an already-injected context is replaced in place
        (older drifted copies removed) instead of stacked.
        """
        ctx_end = Session.context_prefix_len(session.messages)
        chunk = session.messages[ctx_end:end_idx]
        if not chunk:
            return DropResult(error="Nothing to summarize (session is already fully compacted).")

        if end_idx <= session.staged_boundary:
            # Free drop: [ctx_end:end_idx] was already distilled into
            # STAGING.md at stage time — reclaim it without any LLM call.
            del session.messages[ctx_end:end_idx]
            session.staged_boundary = ctx_end + (session.staged_boundary - end_idx)
            self.sessions.save(session)
            logger.info(
                "Free drop: removed {} already-staged messages (SB -> {}) for session {}",
                len(chunk),
                session.staged_boundary,
                session.key,
            )
            return DropResult(
                dropped=len(chunk),
                staged_boundary=session.staged_boundary,
            )

        # Commit + drop: distill only the fresh part [SB:end_idx].
        fresh = session.messages[session.staged_boundary : end_idx]

        # Early guard: if the fresh chunk is too large for the consolidator's
        # context, reject upfront rather than letting the LLM fail and losing data.
        chunk_tokens = sum(estimate_message_tokens(m) for m in fresh)
        if (
            self.context_window_tokens > 0
            and chunk_tokens > self.context_window_tokens * 0.8  # 80% threshold
        ):
            logger.warning(
                "dispatch_drop: fresh chunk ({} tokens) exceeds consolidator context "
                "({} x 80%) for session {} — skipping",
                chunk_tokens,
                self.context_window_tokens,
                session.key,
            )
            return DropResult(
                error=(
                    f"⚠️ Compact failed: {len(fresh)} fresh messages ({chunk_tokens:,} tokens) exceed "
                    f"the summarizer's context window ({self.context_window_tokens:,}). "
                    f"Session history preserved. Consider compacting with n_keep > 0 first."
                )
            )

        summary = await self._consolidate_with_retry(fresh)
        if not summary:
            # Summary generation failed — do NOT drop or advance SB.
            logger.warning(
                "dispatch_drop: summary failed for {} messages in session {} — "
                "nothing dropped to prevent data loss",
                len(fresh),
                session.key,
            )
            return DropResult(
                error=(
                    f"⚠️ Compact failed: summary generation returned no result for {len(fresh)} messages. "
                    f"Session history preserved. This may happen if the content exceeds the model's context window."
                )
            )

        del session.messages[ctx_end:end_idx]
        session.staged_boundary = ctx_end

        # Inject session context at the new pinned prefix so the agent keeps
        # continuity after the raw messages are gone.
        if self._last_session_context:
            summary_msg = {
                "role": "user",
                "content": f"[Previous session context]: {self._last_session_context}",
                "_session_context": True,
                "_compact_summary": True,
                "timestamp": datetime.now().isoformat(),
            }
            tagged = [
                i
                for i in range(ctx_end)
                if session.messages[i].get("_session_context")
                or session.messages[i].get("_compact_summary")
            ]
            if tagged:
                # Replace the newest tagged message in place; remove any older
                # drifted copies (legacy sessions may hold more than one).
                keep = tagged[-1]
                for i in tagged[:-1]:
                    del session.messages[i]
                    keep -= 1
                session.messages[keep] = summary_msg
                session.staged_boundary -= len(tagged) - 1
            else:
                session.messages.insert(ctx_end, summary_msg)
                session.staged_boundary += 1
            logger.info(
                "Injected (or replaced) session context at pinned prefix for session {}",
                session.key,
            )

        self.sessions.save(session)
        return DropResult(
            dropped=len(chunk),
            consolidated=True,
            summary=summary,
            staged_boundary=session.staged_boundary,
        )

    async def _consolidate_with_retry(self, messages: list[dict[str, object]]) -> str | None:
        """Distill a fresh chunk with guaranteed persistence.

        Same retry loop as :meth:`archive_messages` (retries until the
        raw-dump fallback engages), run against ``consolidate_messages`` so
        callers/tests can substitute that seam directly.
        """
        for _ in range(self.store._MAX_FAILURES_BEFORE_RAW_ARCHIVE):
            result = await self.consolidate_messages(messages)
            if result is not None:
                return result
        return None

    async def consolidate_messages(self, messages: list[dict[str, object]]) -> str | None:
        """Archive a selected message chunk into persistent memory.

        Returns the history_entry summary on success, or None on failure.
        If session_context was returned by the consolidation, it is stored in
        ``self._last_session_context`` for downstream injection (Phase 2).
        """
        messages = await self._transcribe_audio(messages)
        result = await self.store.consolidate(messages, self.provider, self.model)
        if result is None:
            return None
        if isinstance(result, tuple):
            entry, ctx = result
            self._last_session_context = ctx
            return entry
        # Backward compat: if store.consolidate was mocked/patched to return
        # a plain truthy value (e.g., True), pass it through.
        self._last_session_context = None
        return result

    async def _transcribe_audio(self, messages: list[dict[str, object]]) -> list[dict[str, object]]:
        """Transcribe audio blocks in messages before memory consolidation."""
        if not any(
            isinstance(m.get("content"), list)
            and any(isinstance(b, dict) and b.get("type") == "audio" for b in m["content"])
            for m in messages
        ):
            return messages

        provider = TranscriptionService(
            self.provider.generation.config
        )  # Use global config from provider
        if not provider.config.providers.groq.api_key:
            logger.warning(
                "No transcription provider available; audio blocks will be stripped from memory"
            )
            return self._strip_audio(messages)

        audio_blocks = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "audio":
                        audio_blocks.append(block)

        if not audio_blocks:
            return messages

        async def _transcribe_block(block: dict) -> str:
            data_uri = block.get("data", "")
            mime_type = block.get("mime_type", "audio/wav")
            ext = "wav"
            if "ogg" in mime_type:
                ext = "ogg"
            elif "mp3" in mime_type or "mpeg" in mime_type:
                ext = "mp3"
            elif "pcm" in mime_type:
                ext = "raw"

            try:
                base64_data = data_uri
                if "," in data_uri:
                    base64_data = data_uri.split(",", 1)[1]
                # Validate before decoding (b64decode silently ignores invalid chars)
                import re

                if not re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", base64_data) or len(base64_data) == 0:
                    raise ValueError("invalid base64 data")
                raw = base64.b64decode(base64_data)
            except Exception:
                return "[Transcription failed — invalid audio data]"

            tmp_path = None
            try:
                fd, tmp_path = tempfile.mkstemp(suffix=f".{ext}")
                with os.fdopen(fd, "wb") as f:
                    f.write(raw)
                text = await provider.transcribe(Path(tmp_path))
                return text if text else "[Transcription returned empty result]"
            except Exception as e:
                logger.warning("Audio transcription failed: {}", e)
                return f"[Transcription failed — {e}]"
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        transcriptions = await asyncio.gather(*[_transcribe_block(b) for b in audio_blocks])

        audio_index = 0
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                new_content = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "audio":
                        text = transcriptions[audio_index]
                        new_content.append({"type": "text", "text": f"[Transcribed Audio]: {text}"})
                        audio_index += 1
                    else:
                        new_content.append(block)
                msg["content"] = new_content

        return messages

    @staticmethod
    def _strip_audio(messages: list[dict[str, object]]) -> list[dict[str, object]]:
        """Remove audio blocks from messages when no transcription provider is available."""
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                msg["content"] = [
                    b for b in content if not (isinstance(b, dict) and b.get("type") == "audio")
                ]
        return messages

    def pick_consolidation_boundary(
        self,
        session: Session,
        tokens_to_remove: int,
    ) -> tuple[int, int] | None:
        """Pick a user-turn boundary that removes enough old prompt tokens.

        Starts the walk at the end of the pinned context prefix (tag-scan),
        so the picker only considers live (staged + fresh) messages.
        """
        start = Session.context_prefix_len(session.messages)
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        for idx in range(start, len(session.messages)):
            message = session.messages[idx]
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)

        return last_boundary

    def estimate_session_prompt_tokens(self, session: Session) -> tuple[int, str]:
        """Estimate current prompt size for the normal session history view."""
        history = session.get_history(max_messages=0)
        from nanobot.bus.events import Address

        try:
            addr = Address.from_uri(session.key)
            channel = addr.channel
            chat_id = addr.segments[0] if addr.segments else None
        except Exception:
            channel, chat_id = None, None
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
        )
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    async def archive_messages(self, messages: list[dict[str, object]]) -> str | None:
        """Archive messages with guaranteed persistence (retries until raw-dump fallback).

        Returns the history_entry summary on success, or None on failure.
        """
        if not messages:
            return None
        for _ in range(self.store._MAX_FAILURES_BEFORE_RAW_ARCHIVE):
            result = await self.consolidate_messages(messages)
            if result is not None:
                return result
        return None

    async def maybe_consolidate_by_tokens(
        self, session: Session, context_window_tokens: int
    ) -> None:
        """Archive old messages until prompt fits within safe budget.

        context_window_tokens is the *chat session's* context ceiling (from the
        active agent profile), used to calculate consolidation thresholds and
        how much history to keep.  This is distinct from the consolidator's own
        provider context limit, which constrains how much the consolidator can
        process in a single LLM call.
        """
        if not session.messages or context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        async with lock:
            budget = context_window_tokens - self.max_completion_tokens - self._SAFETY_BUFFER
            target = budget // 2
            estimated, source = self.estimate_session_prompt_tokens(session)
            if estimated <= 0:
                return
            if estimated < budget:
                return

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    return

                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))
                end_idx = boundary[0] if boundary else None

                # 3-case dispatch (D4), keyed on where the picked boundary B
                # falls relative to the staged boundary (SB). The core lives in
                # _dispatch_drop (shared with /compact and the compact_history
                # tool); the session lock is already held here.
                ctx_end = Session.context_prefix_len(session.messages)
                if end_idx is not None:
                    end = end_idx
                elif session.staged_boundary > ctx_end:
                    # Fresh region too small for the picker, but the already
                    # staged region is reclaimable for free (D4: pressure is
                    # never stuck waiting for fresh to grow).
                    end = session.staged_boundary
                else:
                    return

                result = await self._dispatch_drop(session, end)
                if result.error:
                    # Nothing was dropped (80% guard or distillation failure) —
                    # retrying this round would repeat the same failure.
                    logger.warning(
                        "Token-budget consolidation stopped for session {}: {}",
                        session.key,
                        result.error,
                    )
                    return

                estimated, source = self.estimate_session_prompt_tokens(session)
                if estimated <= 0:
                    return
