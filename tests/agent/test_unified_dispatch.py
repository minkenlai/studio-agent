"""Unit tests for the unified consolidation dispatch (D4).

Targets maybe_consolidate_by_tokens directly with the real picker and a
mocked LLM layer, covering the 3-case dispatch keyed on where the picked
boundary B falls relative to the staged boundary (SB):

- free drop      — B within staged region [ctx_end, SB]: physical delete, SB shift, no LLM
- commit + drop  — B in fresh region (SB, len]: distill [SB:B] first, then drop [ctx_end:B]
- nothing to do  — no boundary and no staged region to reclaim

Plus: free-reclaim fallback (picker empty, SB > ctx_end), consolidate
failure (nothing dropped), and SB self-healing on commit + drop.

Math: window=3000, max_completion=128, SAFETY_BUFFER=1024 →
budget = 1848, target = 924. Message tokens are monkeypatched to 500 each.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import nanobot.agent.memory as memory_module
from nanobot.agent.memory import MemoryConsolidator
from nanobot.session.manager import Session

WINDOW = 3000
# budget = 3000 - 128 - 1024 = 1848 (trigger threshold), target = 924
BUDGET = 1848
TARGET = 924
MSG_TOKENS = 400
# With 400-token messages the picker's boundary is idx 4 (u2): pre-boundary
# removed = 1600 >= 924, and 800 < 924 at the previous user boundary (u1).


def _make_consolidator(tmp_path: Path) -> MemoryConsolidator:
    """Consolidator with real picker, mocked LLM-facing methods patched per test."""
    provider = MagicMock()
    provider.generation = None  # no context_max override
    return MemoryConsolidator(
        workspace=tmp_path,
        provider=provider,
        model="test-model",
        sessions=MagicMock(),
        context_window_tokens=WINDOW,
        build_messages=lambda *args, **kwargs: [],
        get_tool_definitions=lambda: [],
        max_completion_tokens=128,
    )


def _pairs_session(key: str, n_pairs: int, sb: int = 0) -> Session:
    s = Session(key=key)
    for i in range(n_pairs):
        s.add_message("user", f"u{i}")
        s.add_message("assistant", f"a{i}")
    s.staged_boundary = sb
    return s


def _patched_estimate(values: list[tuple[int, str]]):
    """Scripted estimate_session_prompt_tokens: one value per call."""
    calls = [0]

    def _est(_session: Session):
        idx = min(calls[0], len(values) - 1)
        calls[0] += 1
        return values[idx]

    return _est


class TestUnifiedDispatch:
    """maybe_consolidate_by_tokens 3-case dispatch."""

    @pytest.mark.asyncio
    async def test_under_budget_no_dispatch(self, tmp_path: Path, monkeypatch) -> None:
        """est < budget → return immediately, no LLM work, session untouched."""
        consolidator = _make_consolidator(tmp_path)
        session = _pairs_session("test:under", 4)
        monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _m: MSG_TOKENS)
        mock_consolidate = AsyncMock(return_value="entry")

        with (
            patch.object(
                consolidator,
                "estimate_session_prompt_tokens",
                side_effect=_patched_estimate([(100, "m")]),
            ),
            patch.object(consolidator, "consolidate_messages", new=mock_consolidate),
        ):
            await consolidator.maybe_consolidate_by_tokens(session, WINDOW)
            mock_consolidate.assert_not_awaited()

        assert len(session.messages) == 8
        assert session.staged_boundary == 0
        consolidator.sessions.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_commit_drop_with_context_injection(self, tmp_path: Path, monkeypatch) -> None:
        """B in fresh region → distill [SB:B], drop [ctx_end:B], inject dual-tagged ctx."""
        consolidator = _make_consolidator(tmp_path)
        session = _pairs_session("test:commit", 4)  # u0 a0 u1 a1 u2 a2 u3 a3, SB=0
        monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _m: MSG_TOKENS)
        mock_consolidate = AsyncMock()

        async def _consolidate(chunk):
            consolidator._last_session_context = "Working on dispatch tests."
            return "History entry 1."

        mock_consolidate.side_effect = _consolidate

        with (
            patch.object(
                consolidator,
                "estimate_session_prompt_tokens",
                side_effect=_patched_estimate([(BUDGET, "m"), (100, "m")]),
            ),
            patch.object(consolidator, "consolidate_messages", new=mock_consolidate),
        ):
            await consolidator.maybe_consolidate_by_tokens(session, WINDOW)
            # est0 = budget (triggers, not < budget); ttr = 924 → picker returns
            # user boundary idx 4 (pre-boundary tokens 1600 >= 924).
            mock_consolidate.assert_awaited_once()

        chunk = mock_consolidate.call_args[0][0]
        assert [m["content"] for m in chunk] == ["u0", "a0", "u1", "a1"]  # fresh [SB=0 : 4]

        # [0:4] dropped; ctx injected at ctx_end=0; SB = 0 + 1
        injected = session.messages[0]
        assert injected["role"] == "user"
        assert injected["content"] == "[Previous session context]: Working on dispatch tests."
        assert injected["_session_context"] is True
        assert injected["_compact_summary"] is True
        assert [m["content"] for m in session.messages[1:]] == ["u2", "a2", "u3", "a3"]
        assert session.staged_boundary == 1
        consolidator.sessions.save.assert_called_once_with(session)

    @pytest.mark.asyncio
    async def test_commit_drop_without_context_no_injection(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Plain str result (no session_context) → drop happens, no injected message."""
        consolidator = _make_consolidator(tmp_path)
        session = _pairs_session("test:plain", 4)
        monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _m: MSG_TOKENS)
        mock_consolidate = AsyncMock(return_value="History entry 2.")

        with (
            patch.object(
                consolidator,
                "estimate_session_prompt_tokens",
                side_effect=_patched_estimate([(BUDGET, "m"), (100, "m")]),
            ),
            patch.object(consolidator, "consolidate_messages", new=mock_consolidate),
        ):
            await consolidator.maybe_consolidate_by_tokens(session, WINDOW)
            mock_consolidate.assert_awaited_once()

        assert [m["content"] for m in session.messages] == ["u2", "a2", "u3", "a3"]
        assert not any(m.get("_compact_summary") for m in session.messages)
        assert session.staged_boundary == 0

    @pytest.mark.asyncio
    async def test_free_drop_no_llm_when_boundary_in_staged(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """B within staged region → physical delete + SB shift, consolidate NOT called."""
        consolidator = _make_consolidator(tmp_path)
        session = _pairs_session("test:free", 4, sb=5)  # staged [0:5]
        monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _m: MSG_TOKENS)
        mock_consolidate = AsyncMock(return_value="should not run")

        with (
            patch.object(
                consolidator,
                "estimate_session_prompt_tokens",
                side_effect=_patched_estimate([(BUDGET, "m"), (100, "m")]),
            ),
            patch.object(consolidator, "consolidate_messages", new=mock_consolidate),
        ):
            await consolidator.maybe_consolidate_by_tokens(session, WINDOW)
            # Picker boundary 4 <= SB=5 → free drop.
            mock_consolidate.assert_not_awaited()

        # [0:4] removed; SB' = ctx_end + (SB - end_idx) = 0 + (5 - 4) = 1
        assert [m["content"] for m in session.messages] == ["u2", "a2", "u3", "a3"]
        assert session.staged_boundary == 1
        consolidator.sessions.save.assert_called_once_with(session)

    @pytest.mark.asyncio
    async def test_free_reclaim_fallback_when_picker_empty(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Picker finds no user boundary, but SB > ctx_end → reclaim staged region for free."""
        consolidator = _make_consolidator(tmp_path)
        session = Session(key="test:reclaim")
        session.messages = [
            {
                "role": "user",
                "content": "[Previous session context]: old",
                "_session_context": True,
                "_compact_summary": True,
                "timestamp": "2026-01-01T00:00:00",
            },
            {"role": "user", "content": "u0", "timestamp": "2026-01-01T00:00:01"},
            {"role": "assistant", "content": "a0", "timestamp": "2026-01-01T00:00:02"},
        ]
        session.staged_boundary = 3
        monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _m: MSG_TOKENS)
        mock_consolidate = AsyncMock(return_value="should not run")

        with (
            patch.object(
                consolidator,
                "estimate_session_prompt_tokens",
                side_effect=_patched_estimate([(BUDGET, "m"), (100, "m")]),
            ),
            patch.object(consolidator, "consolidate_messages", new=mock_consolidate),
        ):
            await consolidator.maybe_consolidate_by_tokens(session, WINDOW)
            # Picker: start=1 (tag scan), no user at idx>1 → None. SB=3 > ctx_end=1
            # → end_idx=SB=3, staged free drop of [1:3].
            mock_consolidate.assert_not_awaited()

        assert [m["content"] for m in session.messages] == ["[Previous session context]: old"]
        assert session.staged_boundary == 1  # ctx_end + (SB - end_idx) = 1 + 0


class TestUnifiedDispatchFailureModes:
    """Nothing is lost on failure; SB self-heals on commit + drop."""

    @pytest.mark.asyncio
    async def test_consolidate_failure_preserves_messages(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """consolidate_messages falsy (retries exhausted) → return before delete:
        session fully intact."""
        consolidator = _make_consolidator(tmp_path)
        session = _pairs_session("test:fail", 4)
        monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _m: MSG_TOKENS)
        mock_consolidate = AsyncMock(return_value=None)

        with (
            patch.object(
                consolidator,
                "estimate_session_prompt_tokens",
                side_effect=_patched_estimate([(BUDGET, "m")]),
            ),
            patch.object(consolidator, "consolidate_messages", new=mock_consolidate),
        ):
            await consolidator.maybe_consolidate_by_tokens(session, WINDOW)
            # The dispatch retries the distillation until the raw-dump fallback count.
            from nanobot.agent.memory import MemoryStore

            assert mock_consolidate.await_count == MemoryStore._MAX_FAILURES_BEFORE_RAW_ARCHIVE

        assert [m["content"] for m in session.messages] == [
            "u0",
            "a0",
            "u1",
            "a1",
            "u2",
            "a2",
            "u3",
            "a3",
        ]
        assert session.staged_boundary == 0
        consolidator.sessions.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_commit_drop_heals_drifted_staged_boundary(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Drifted SB is reset to ctx_end on commit + drop; only fresh [SB:B] distilled."""
        consolidator = _make_consolidator(tmp_path)
        session = _pairs_session("test:heal", 4, sb=2)  # SB drifted to 2
        monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _m: MSG_TOKENS)
        mock_consolidate = AsyncMock(return_value="entry")

        with (
            patch.object(
                consolidator,
                "estimate_session_prompt_tokens",
                side_effect=_patched_estimate([(BUDGET, "m"), (100, "m")]),
            ),
            patch.object(consolidator, "consolidate_messages", new=mock_consolidate),
        ):
            await consolidator.maybe_consolidate_by_tokens(session, WINDOW)
            # Picker boundary 4 > SB=2 → commit + drop.
            mock_consolidate.assert_awaited_once()

        # Only the fresh part [SB=2 : 4] was distilled — staged [0:2] untouched.
        chunk = mock_consolidate.call_args[0][0]
        assert [m["content"] for m in chunk] == ["u1", "a1"]

        # [0:4] dropped; SB healed to ctx_end (0), no context → no injection.
        assert [m["content"] for m in session.messages] == ["u2", "a2", "u3", "a3"]
        assert session.staged_boundary == 0

    @pytest.mark.asyncio
    async def test_no_boundary_and_no_staged_region_noop(self, tmp_path: Path, monkeypatch) -> None:
        """Single user msg: picker None and SB == ctx_end → nothing to do, no delete."""
        consolidator = _make_consolidator(tmp_path)
        session = Session(key="test:noop")
        session.add_message("user", "only")
        session.staged_boundary = 0
        monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _m: MSG_TOKENS)
        mock_consolidate = AsyncMock(return_value="should not run")

        with (
            patch.object(
                consolidator,
                "estimate_session_prompt_tokens",
                side_effect=_patched_estimate([(BUDGET, "m")]),
            ),
            patch.object(consolidator, "consolidate_messages", new=mock_consolidate),
        ):
            await consolidator.maybe_consolidate_by_tokens(session, WINDOW)
            mock_consolidate.assert_not_awaited()

        assert [m["content"] for m in session.messages] == ["only"]
        assert session.staged_boundary == 0
        consolidator.sessions.save.assert_not_called()


def _tagged_msg(content: str) -> dict:
    return {
        "role": "user",
        "content": content,
        "_session_context": True,
        "_compact_summary": True,
        "timestamp": "2026-01-01T00:00:00",
    }


class TestInjectionDedup:
    """S2 invariant: at most ONE tagged context message in the leading pinned run."""

    @pytest.mark.asyncio
    async def test_commit_drop_replaces_existing_tagged_context(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Two consecutive commit+drop rounds must REPLACE the injected context,
        not stack a second tagged message."""
        consolidator = _make_consolidator(tmp_path)
        session = _pairs_session("test:dedup", 4)  # u0 a0 u1 a1 u2 a2 u3 a3, SB=0
        monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _m: MSG_TOKENS)
        mock_consolidate = AsyncMock()
        round_no = [0]

        async def _consolidate(_chunk):
            round_no[0] += 1
            consolidator._last_session_context = f"CTX-R{round_no[0]}"
            return f"History entry {round_no[0]}."

        mock_consolidate.side_effect = _consolidate

        with (
            patch.object(
                consolidator,
                "estimate_session_prompt_tokens",
                side_effect=_patched_estimate([(BUDGET, "m"), (BUDGET, "m"), (100, "m")]),
            ),
            patch.object(consolidator, "consolidate_messages", new=mock_consolidate),
        ):
            await consolidator.maybe_consolidate_by_tokens(session, WINDOW)
            # est0 = budget → round 1 (boundary 4, commit+drop); after the drop the
            # re-estimate is still BUDGET → round 2 (boundary 4 on the shifted list,
            # commit+drop again); est2 = 100 <= target stops the loop.
            assert round_no[0] == 2

        # Round 2 distilled only the fresh tail: [new SB=1 : 4] = u2 a2.
        second_chunk = mock_consolidate.call_args_list[1][0][0]
        assert [m["content"] for m in second_chunk] == ["u2", "a2"]

        # Exactly ONE tagged message in the whole session, holding the LATEST context.
        tagged = [
            m for m in session.messages if m.get("_session_context") or m.get("_compact_summary")
        ]
        assert len(tagged) == 1
        assert tagged[0]["content"] == "[Previous session context]: CTX-R2"

        assert [m["content"] for m in session.messages] == [
            "[Previous session context]: CTX-R2",
            "u3",
            "a3",
        ]
        assert session.staged_boundary == 1

    @pytest.mark.asyncio
    async def test_legacy_drift_collapses_to_one_tagged(self, tmp_path: Path, monkeypatch) -> None:
        """Pre-existing drift (TWO tagged msgs in the leading run) collapses to one
        on the next commit+drop."""
        consolidator = _make_consolidator(tmp_path)
        session = Session(key="test:drift")
        session.messages = [
            _tagged_msg("[Previous session context]: old1"),
            _tagged_msg("[Previous session context]: old2"),
            {"role": "user", "content": "u0", "timestamp": "2026-01-01T00:00:03"},
            {"role": "assistant", "content": "a0", "timestamp": "2026-01-01T00:00:04"},
            {"role": "user", "content": "u1", "timestamp": "2026-01-01T00:00:05"},
            {"role": "assistant", "content": "a1", "timestamp": "2026-01-01T00:00:06"},
            {"role": "user", "content": "u2", "timestamp": "2026-01-01T00:00:07"},
            {"role": "assistant", "content": "a2", "timestamp": "2026-01-01T00:00:08"},
        ]
        session.staged_boundary = 2  # consistent with the two tagged msgs
        monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _m: MSG_TOKENS)
        mock_consolidate = AsyncMock()

        async def _consolidate(_chunk):
            consolidator._last_session_context = "CTX-FRESH"
            return "History entry fresh."

        mock_consolidate.side_effect = _consolidate

        with (
            patch.object(
                consolidator,
                "estimate_session_prompt_tokens",
                side_effect=_patched_estimate([(BUDGET, "m"), (100, "m")]),
            ),
            patch.object(consolidator, "consolidate_messages", new=mock_consolidate),
        ):
            await consolidator.maybe_consolidate_by_tokens(session, WINDOW)
            mock_consolidate.assert_awaited_once()

        # Drift collapsed: exactly one tagged msg (the replacement), and SB accounts
        # for the prefix shrinking by one.
        tagged = [
            m for m in session.messages if m.get("_session_context") or m.get("_compact_summary")
        ]
        assert len(tagged) == 1
        assert tagged[0]["content"] == "[Previous session context]: CTX-FRESH"

        assert [m["content"] for m in session.messages] == [
            "[Previous session context]: CTX-FRESH",
            "u2",
            "a2",
        ]
        assert session.staged_boundary == 1  # ctx_end(2) - (2 tagged - 1)


class TestSharedDispatchEntry:
    """consolidate_range: the public entry point acquires the session lock (F1)."""

    @pytest.mark.asyncio
    async def test_consolidate_range_acquires_lock(self, tmp_path: Path, monkeypatch) -> None:
        """While another holder keeps the consolidator lock, consolidate_range
        blocks (proves it routes through get_lock)."""
        consolidator = _make_consolidator(tmp_path)
        session = _pairs_session("test:lock", 4)
        monkeypatch.setattr(memory_module, "estimate_message_tokens", lambda _m: 10)

        lock = consolidator.get_lock(session.key)
        await lock.acquire()
        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(consolidator.consolidate_range(session, 4), 0.1)
        finally:
            lock.release()

        assert len(session.messages) == 8  # untouched
        assert session.staged_boundary == 0
