# SPEC — Unified Consolidation Refactor (Review-Findings PR)

Worktree: `/home/nanobot/.nanobot/workspace/nanobot-dev` (branch `dev`, base `f37a55a`).
Fixes the adjudicated review findings: F1/S1 (locking), F4 (guard divergence),
S2 (injection stacking), S3 (docstring). F3 = deliberately skipped (see below),
S4 = deferred to release.

## Design (decided — do not re-litigate)

### 1. Shared dispatch on `MemoryConsolidator` (memory.py)
Add two methods:

```python
@dataclass
class DropResult:
    dropped: int = 0            # messages physically removed
    consolidated: bool = False  # LLM path ran (commit+drop)
    summary: str | None = None  # history entry from consolidation
    error: str | None = None    # user-facing error (guard/failure)
    staged_boundary: int = 0    # resulting SB (for caller reply text)

async def consolidate_range(self, session, end_idx: int) -> DropResult:
    """Public entry: acquires get_lock(session.key), calls _dispatch_drop."""

async def _dispatch_drop(self, session, end_idx: int) -> DropResult:
    """3-case core, assumes lock already held."""
```

`_dispatch_drop` logic (move from current `maybe_consolidate_by_tokens` lines
697-752 + the compact tool's guard):

1. `ctx_end = Session.context_prefix_len(session.messages)`
2. `chunk = messages[ctx_end:end_idx]`; empty → `DropResult(error="Nothing to summarize...")`
3. `end_idx <= SB` → **free drop**: delete, `SB = ctx_end + (SB - end_idx)`, save, no LLM.
4. Else **commit+drop**:
   - `fresh = messages[SB:end_idx]`
   - **80% guard** (moved here from compact.py:89-106): if
     `sum(estimate_message_tokens(m) for m in fresh) > self.context_window_tokens * 0.8`
     → `DropResult(error="...exceed the summarizer's context window...")`, NOTHING dropped.
     (Only applies when `context_window_tokens > 0`.)
   - `summary = await self.archive_messages(fresh)`  ← archive_messages, NOT
     consolidate_messages (retry + raw-dump fallback). This upgrades the
     token-path from 1 attempt to full durability — intentional unification.
     Falsy → `DropResult(error="summary generation failed...")`, nothing dropped.
   - `del messages[ctx_end:end_idx]`; `SB = ctx_end`
   - **Injection with dedup** (see §2) when `_last_session_context` truthy.
   - save.
5. Return result.

### 2. S2 injection dedup — "at most one tagged message in the leading run"
Replace the current insert-at-ctx_end block (memory.py:736-745). After the drop:

```python
if self._last_session_context:
    summary_msg = {...}  # unchanged dual-tagged dict
    tagged = [i for i in range(ctx_end)
              if session.messages[i].get("_session_context")
              or session.messages[i].get("_compact_summary")]
    if tagged:
        # replace the newest tagged msg in place, delete the older ones
        keep = tagged[-1]
        for i in tagged[:-1]:
            del session.messages[i]
            keep -= 1
        session.messages[keep] = summary_msg
        session.staged_boundary -= (len(tagged) - 1)   # prefix shrank
    else:
        session.messages.insert(ctx_end, summary_msg)
        session.staged_boundary += 1
```

Normal cases: 0 tagged → insert (+1); 1 tagged → replace, SB unchanged.
Pathological (N tagged, legacy drift) → collapse to 1.

### 3. Wire the callers
- `maybe_consolidate_by_tokens` (memory.py:661): keep the estimate loop +
  picker; replace its inline free-drop/commit+drop branch (lines 698-752) with
  `res = await self._dispatch_drop(session, end_idx)`; on `res.error` return
  (failure); loop re-estimates after each successful round. Lock stays where it
  is (public wrapper NOT used here — lock already held; asyncio.Lock is not
  reentrant).
- `nanobot/agent/tools/compact.py` `execute()`: replace body after chunk check
  with `res = await self._consolidator.consolidate_range(session, end_idx)`;
  build reply from `res` (free-drop text as now, summary text as now). Delete
  the inline guard + inline drop/inject code.
- `nanobot/command/builtin.py` `cmd_compact` (line 338): same as tool —
  `await loop.memory_consolidator.consolidate_range(session, end_idx)`, map
  result to reply text. Gains the 80% guard for free (F4 fixed).
- `cmd_forget` (456), `cmd_new` (617), `cmd_stage` (530): different semantics
  (forget drops fresh by design; stage drops nothing) — they do NOT go through
  the dispatch. Wrap their read-modify-write+save blocks in
  `async with loop.memory_consolidator.get_lock(session.key):` (F1 for them).

### 4. S3 docstring (session/manager.py:49)
`context_prefix_len`: state that a message counts if it has EITHER tag (OR),
and that all producers set both tags.

### 5. F3 — SKIPPED, with reason
Keep re-estimating each round: the estimate covers probe msg + tool defs +
injected context, so manual add/subtract bookkeeping would drift. Path is rare
(≤5 rounds, local tiktoken/provider-counter). Noted in commit message.

### 6. S4 — deferred
da068a7 debug logging stays; remove/gate at release.

## Test plan (TDD: write/red first, then implement)

A. `tests/agent/test_unified_dispatch.py` — extend:
   1. `test_commit_drop_replaces_existing_tagged_context` (S2): 8 pairs,
      SB=0, `_last_session_context` set by a side_effect that also distinguishes
      round 1 vs 2 (e.g. "CTX-R1" / "CTX-R2"). Estimates: (BUDGET, BUDGET, 100)
      → two commit+drop rounds (round 1 boundary 4, round 2 boundary 4 on the
      shifted list). Assert: consolidate awaited TWICE; 2nd chunk ==
      [u2,a2,u3]; final messages[0] is the ONLY tagged msg and its content
      contains "CTX-R2" (replaced, not stacked); exactly one tagged msg in the
      whole session; SB == 1.
   2. `test_legacy_drift_collapses_to_one_tagged`: pre-seed TWO tagged msgs in
      the leading run (ctx_end=2, SB=2), one commit+drop round → final leading
      run has exactly 1 tagged msg; SB consistent (SB = 1 + 0 or 1+1 per math:
      drop [2:end], SB=2, dedup removes 1 older tagged → SB=1).
   3. `test_consolidate_range_acquires_lock`: hold `get_lock(key)` in the test,
      `asyncio.wait_for(consolidator.consolidate_range(...), 0.1)` →
      `asyncio.TimeoutError` (proves it waits on the lock).
B. `tests/cli/test_compact_command.py` — extend:
   4. `test_cmd_compact_rejects_chunk_over_80pct_of_consolidator_context`
      (F4): fresh chunk > 80% of `context_window_tokens` → error reply,
      messages + SB unchanged, no consolidation awaited.
   5. `test_cmd_compact_acquires_consolidator_lock` (F1): hold the lock,
      dispatch cmd_compact via wait_for → TimeoutError.
   6. Existing tests: re-point only if the dispatch move breaks a mock seam
      (archive_messages/consolidate_messages mocks should survive — dispatch
      still calls them).
C. `tests/agent/test_compact_tool.py` — existing 25 should stay green
   (mock seams unchanged); spot-check the free-drop + injection tests.
D. `tests/cli/test_forget_command.py` / `test_new_command.py` /
   `test_stage_command.py` — add a lock-acquired assertion per file (same
   wait_for pattern as A.3, via the command entry point).
E. Full suite + ruff (check + format) green; then commit
   (`git commit -F <tmpfile>`, never -m).

## DoD
- One shared dispatch; zero duplicated free-drop/commit+drop logic outside it.
- All 6 session-mutating paths (token loop, tool, compact, forget, new, stage)
  hold the consolidator lock.
- Tagged-context invariant: ≤1 tagged msg in the leading run, tested.
- 80% guard lives in exactly one place, applies to both compact paths.
- 1078+ tests, 0 failed; ruff clean.
- PLAN/Backlog items F1,F4,S2,S3,S1 checked; F3 marked skipped-with-reason;
  S4 deferred.
- Delete this spec file on commit (or commit it, then delete in follow-up —
  prefer committing it as docs/spec and keeping it in git).
