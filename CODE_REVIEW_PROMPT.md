You are performing a **fresh, adversarial code review (ITERATION 3)** of the
"unified session consolidation" feature in the `nanobot` Python repo. You have
no prior context — everything you need is below. Be skeptical; your job is to
find real bugs, subtle state-corruption paths, race conditions, and spec
violations, **not to rubber-stamp**.

This is iteration 3. Two prior iterations exist:
- **Iteration 1** reviewed the original feature (findings F1/S1, F4, S2, S3,
  F3, S4) — all FIXED and re-verified in iteration 2 (see
  `kb/01_Projects/Nanobot/reviews/iteration-2/REVIEW_ITER2.md` §2 if you have
  KB access; the proofs are listed again below).
- **Iteration 2** reviewed the refactor PR `268613b` and produced B-1 (blocker),
  A-1 (medium), A-2 (low). **B-1 and A-1 are now fixed in commit `e863cb7`.**

Your two jobs:
(a) **Verify B-1 and A-1 are actually fixed** (not just claimed) in `e863cb7`,
and that the fix did not break the previously-verified invariants;
(b) **Find NEW bugs** introduced by the fix commit (it touched the core seams —
`consolidate_messages` / `_consolidate_with_retry` return types changed from
`str | None` to `tuple[str | None, str | None]` — and every caller/mocked seam).
Do NOT re-derive the original feature design — that was iteration 1.

## Where to look
- **Worktree (active dev branch):** `/home/nanobot/.nanobot/workspace/nanobot-dev`
- **Branch:** `dev`
- **THE CHANGE UNDER REVIEW IS `268613b..e863cb7`:**
  ```
  git -C /home/nanobot/.nanobot/workspace/nanobot-dev diff 268613b..e863cb7
  git -C /home/nanobot/.nanobot/workspace/nanobot-dev show --stat e863cb7
  ```
  12 files: `nanobot/agent/memory.py`, `nanobot/command/builtin.py`, and 10 test
  files. The production diff is small; the test diff is large (seam migration).
- **Full-feature context (orientation only):**
  ```
  git -C /home/nanobot/.nanobot/workspace/nanobot-dev log --oneline 4874e73..e863cb7
  ```
  (`6807f54` Phase 1 … `f37a55a` Phase 4 tests, `da068a7` channels debug logging,
  `268613b` unified dispatch, `e863cb7` the fix under review.)
- **Authoritative docs — read in this order:**
  1. `SPEC_consolidation_refactor.md` (worktree root, git-tracked) — refactor
     design; note its §1/§2/§4 references to `_last_session_context` are
     SUPERSEDED by A-1 (see "Post-iteration-2 state" below).
  2. `PLAN_unified_consolidation.md` (worktree root, git-tracked) — the original
     D1–D10 decision table and invariants the code must not regress.
- Prefer reading the actual diff and current files over trusting this summary.
  If the worktree isn't accessible to you, say so explicitly.

## What this feature does (so you can judge it) — POST-`e863cb7`
Sessions store a `staged_boundary` (SB): `[0 : SB]` distilled into STAGING.md
("staged"); `[SB : len]` is "fresh". A pinned context prefix (injected
session-context summaries) sits at the front, marked by tags
`_session_context` / `_compact_summary`. `Session.context_prefix_len()`
(`session/manager.py:49`) counts that *leading tagged run* (tag-scan) — a
message counts if it has **either** tag; it is NOT a stored pointer.

The **unified drop dispatch** is a single locked code path on
`MemoryConsolidator` (`nanobot/agent/memory.py`):
- `consolidate_range(session, end_idx)` (:488) — **public** entry; acquires
  `get_lock(session.key)`, then calls the core.
- `_dispatch_drop(session, end_idx)` (:498) — **3-case core**, assumes lock held:
  - **free drop** — `end_idx <= SB`: `del messages[ctx_end:end_idx]`,
    `SB = ctx_end + (SB - end_idx)`, **no LLM call**, `save()`.
  - **commit + drop** — `end_idx > SB`: 80% guard on `fresh = messages[SB:end_idx]`,
    then `summary, session_ctx = await self._consolidate_with_retry(fresh)`.
    On success: `del messages[ctx_end:end_idx]`, `SB = ctx_end`, then
    **replace-in-place injection** of `session_ctx` (see invariant 3). On ANY
    failure (guard trip / no summary): **nothing dropped, SB untouched**,
    `DropResult.error` set.
  - **nothing to do** — empty chunk → `DropResult(error=...)`.
- `DropResult` (:436): `dropped`, `consolidated`, `summary`, `error`,
  `staged_boundary`.
- **SEAM (post-A-1):** `consolidate_messages` (:644) returns
  **`(history_entry, session_context)` tuple** — `(None, None)` on failure; a
  plain truthy store result is passed through as `(result, None)` (backward
  compat with legacy mocks). `_consolidate_with_retry` (:624) returns the tuple
  and retries while `entry is None` (a `(None, None)` tuple IS a failure —
  it must NOT short-circuit the loop). `archive_messages` (:813) keeps its
  external `str | None` contract: it unpacks the tuple, returns the entry,
  discards the context by design (standalone archival drops nothing).
  **There is NO instance-wide `_last_session_context` field anymore** (deleted
  in A-1); the context travels through the return value into `_dispatch_drop`'s
  local `session_ctx`.
- `maybe_consolidate_by_tokens` (:829): acquires the lock itself, up to
  `_MAX_CONSOLIDATION_ROUNDS=5` rounds, calls `_dispatch_drop` **directly**
  (lock already held; asyncio.Lock is not reentrant). Free-reclaim fallback:
  picker returns None but `SB > ctx_end` → `end = SB`.

**All mutation entry points route through this one lock:**
- `/compact` (`builtin.py:338`): `end_idx = len - n_keep` → `consolidate_range`.
- `compact_history` tool (`tools/compact.py`): → `consolidate_range`.
- `/forget` (`builtin.py:398`): under lock, `messages = messages[:ctx_end]`,
  `SB = ctx_end`, save + invalidate.
- `/stage [N]` (`builtin.py:475`): **everything session-read — `sb`, `desired`,
  the picker, `chunk` slice, and the logger — runs INSIDE the lock** (B-1 fix);
  only `n_keep` parsing (from `ctx.raw`, session-independent) stays outside.
  Distillation + SB-advance + save inside the lock. Drops nothing.
- `/new` (`builtin.py:567`): snapshot+clear+save+invalidate **under the lock**;
  background `archive_messages(snapshot)` scheduled **OUTSIDE** the lock via
  `loop._schedule_background` (archive takes no lock, mutates no session).

## Iteration-2 findings to VERIFY AS FIXED (don't assume; prove it)
- **B-1 (blocker) — `/stage` TOCTOU:** pre-fix, `sb`/`desired`/`new_sb`/`chunk`
  were read pre-lock; a concurrent drop could let a stale `new_sb` be written
  over a shrunken list (`SB > len`) → later `/new` (snapshot = `messages[SB:]`
  empty → clears unarchived tail) or free-reclaim (`end = SB`) silently deleted
  live messages. VERIFY: the pick+slice is now inside the locked section, no
  session read happens between the lock-free prefix and the lock, and the
  regression test (`test_cmd_stage_recomputes_boundary_inside_lock_toctou`,
  `tests/cli/test_stage_command.py`) actually fails on the pre-fix code (check
  the test mechanics: it holds the lock, spawns `cmd_stage`, asserts the task
  parks, mutates the session, releases, and asserts `SB <= len`).
- **A-1 (medium) — shared `_last_session_context` deleted:** VERIFY:
  (a) the field is gone from `MemoryConsolidator.__init__` and no production
  reference remains (`grep -rn _last_session_context nanobot/` must be empty);
  (b) `_dispatch_drop` injects from the local `session_ctx` returned by
  `_consolidate_with_retry`; (c) `archive_messages` still returns `str | None`
  (the /new + /stage seam is unchanged externally); (d) the regression test
  `test_no_cross_session_context_leak` (`tests/memory/test_drop_inject.py`)
  proves a session-A archive cannot leak context into session-B's dispatch.
- **A-2 (low, KNOWN, optional):** `archive_messages` and `_consolidate_with_retry`
  duplicate the same 3-attempt retry loop. Intentionally kept for now (the seams
  differ: `archive_messages` funnels to the raw-dump fallback via `store`;
  `_consolidate_with_retry` is the mockable dispatch seam). Note if it was
  cleaned up; do not block on it.
- **S4 (deferred to release):** `nanobot/channels/manager.py` debug logging
  (commit `da068a7`). Known + deferred — note it is still present.

## Invariants to verify (the important stuff)
1. **Data loss is the #1 risk.** On ANY failure path (guard trip, entry None,
   exception, empty chunk) in `_dispatch_drop`, no messages deleted and SB must
   not move. Check `del session.messages[...]` only happens after a successful
   `_consolidate_with_retry` in the commit+drop branch; free drop is lossless.
2. **Tag-scan correctness + either-tag.** `context_prefix_len` counts only the
   *leading* run, stops at first message lacking both tags. Every producer of
   injected context sets the tags it relies on; nothing inserts an untagged
   message *before* a tagged one.
3. **SB bookkeeping on every branch.** Free-drop shift
   (`ctx_end + (SB - end_idx)`) never negative. Commit+drop resets SB to
   `ctx_end`, then replace-in-place injection: NEW tagged msg at `ctx_end` →
   `SB += 1`; existing tagged run replaced → older copies deleted and
   `SB -= (len(tagged) - 1)`. **Scrutinize the drift-collapse index math**
   (deletes earlier tagged indices while decrementing `keep`).
4. **Injection replace-in-place correctness.** Finds all tagged indices in
   `range(ctx_end)`, keeps the last, deletes earlier (shifting `keep`), replaces
   in place. (a) handles a 2+ tagged run without index corruption, (b) SB
   correction matches messages removed from the pinned region, (c) never deletes
   a live untagged message.
5. **Tool-call exchange integrity.** `pick_consolidation_boundary`
   (`memory.py`) cuts at user-turn boundaries; `_pick_stage_boundary`
   (`builtin.py`) clamps to a cut where every `tool_call` in `messages[:k]` has
   its matching `tool` result. Neither leaves a dangling pair in the suffix.
6. **Budget math + free-reclaim.** `budget = window - max_completion -
   _SAFETY_BUFFER(1024)`, `target = budget // 2`. Trigger guard, `estimated <= 0`
   guard, negative budget can't cause a runaway loop. Free-reclaim (`SB > ctx_end`)
   can't drop fresh messages.
7. **80% consolidator-context guard (shared).** In `_dispatch_drop`, fresh chunk
   > `consolidator_context * 0.8` → skipped with user-facing message, nothing
   dropped. Token loop, `/compact`, AND tool all honor it.
8. **Concurrency / lock lifetime.** (a) No session read-modify-write escapes the
   lock on ANY path (the B-1 class of bug — re-check `/stage` specifically:
   `n_keep` parsing outside the lock is fine only because `ctx.raw` is
   session-independent). (b) `/new`'s background `archive_messages` takes no
   lock and mutates no session — confirm it truly can't corrupt state, and is
   safe firing after the session is cleared. (c) **NEW for this iteration:**
   `_dispatch_drop` now unpacks `session_ctx` from the seam's tuple — confirm
   no await between the distillation return and the injection lets another
   task's state interfere (the A-1 class of bug: shared mutable across
   sessions). (d) Lock registry is a `WeakValueDictionary` (`memory.py`) — can a
   lock be GC'd while another coroutine waits, breaking mutual exclusion?
   (e) No path acquires the same non-reentrant lock twice.
9. **Seam contract consistency (NEW for this iteration).** The return type of
   `consolidate_messages` / `_consolidate_with_retry` changed to a tuple.
   Verify: (a) every PRODUCTION caller handles the tuple (grep for callers —
   there are exactly two: `_dispatch_drop` and `archive_messages`); (b)
   `_consolidate_with_retry` retries on `(None, None)` and returns `(None, None)`
   after exhaustion (a failure tuple must not short-circuit the loop);
   (c) `archive_messages` correctly maps `(entry, ctx)` → `entry` and keeps
   retrying while `entry is None`; (d) the plain-truthy backward-compat branch
   (`(result, None)`) can't mask a real failure (e.g. `store.consolidate`
   returning a falsy-but-not-None value); (e) the raw-dump fallback path in
   `store.consolidate` (which returns `None`) still flows to `(None, None)` end
   to end.
10. **Persistence & migration (D10).** `save()` writes `staged_boundary`;
    `_load()` falls back to legacy `last_consolidated` only at load time. No
    stale `last_consolidated` references in production code.
11. **get_history() semantics.** Drops the consolidated slice, surfaces only the
    tagged prefix as pinned context. Cross-check `test_session_manager_history.py`.

## Secondary / smaller items
- `nanobot/channels/telegram.py` + `manager.py`: 3-layer command plumbing for
  `/stage` and `/new` — all three layers present + consistent, group-gate works.
- `_consolidate_with_retry` vs `archive_messages` duplicated retry loop (A-2) —
  minor; confirm the two seams behave the same.
- **Test-seam migration hygiene:** 10 test files were re-pointed at the tuple
  seam. Spot-check that no test now mocks the seam with a bare string (which
  would silently break the unpack in `_dispatch_drop` at runtime, not in the
  test) and that failure-mode tests assert `await_count == 3`
  (`MemoryStore._MAX_FAILURES_BEFORE_RAW_ARCHIVE`), proving the retry loop still
  engages on `(None, None)`.

## What to deliver
1. **Verdict:** is `e863cb7` correct and safe to merge to main? (yes / no /
   yes-with-fixes)
2. **v2-finding verification** — for each of B-1, A-1, A-2, S4: FIXED /
   PARTIAL / NOT-FIXED / N/A, with the exact file:line that proves it.
3. **Findings table (NEW)** — for each: severity (blocker / high / med / low),
   file:line, what's wrong, and a concrete suggested fix.
4. **Invariants checklist** — mark each of the 11 invariants PASS / FAIL /
   NEEDS-REVIEW with one-line justification and the exact line you checked.
5. **Spec-vs-code deltas** — anywhere `SPEC_consolidation_refactor.md` or
   `PLAN_unified_consolidation.md` says one thing and the code does another
   (known deltas: SPEC §1/§2/§4 reference the deleted `_last_session_context`
   and `archive_messages`-based dispatch; PLAN line ~234 references
   `_last_session_context` unset — these are superseded by A-1, confirm no
   OTHER deltas).
6. **Test-gap list** — behaviors not covered by the tests, especially the B-1
   TOCTOU mechanics, the A-1 cross-session isolation, the tuple seam's
   failure-mode retry, and the `/new` background archive.

Prefer reading the actual diff and current files over trusting this summary.
If the worktree isn't accessible to you, say so explicitly rather than guessing.

## METHOD (for the orchestrator — not for a single reviewer)
A single full-diff reviewer on a local model TIMES OUT (verified: 24 min, no
output). Split into two surgical reviewers, run SEQUENTIALLY (single-slot host):
- **Reviewer A — core:** `nanobot/agent/memory.py` (dispatch, seams, locking,
  tuple plumbing, invariants 1-4, 6, 8, 9) + `tests/memory/` +
  `tests/agent/test_unified_dispatch.py`.
- **Reviewer B — commands/seams:** `nanobot/command/builtin.py` (`/stage`
  in-lock restructure), `nanobot/agent/tools/compact.py`,
  `nanobot/channels/{telegram,manager}.py` + `tests/cli/` +
  `tests/agent/test_compact_tool.py` + `tests/agent/test_loop_consolidation_tokens.py`.
Give each `context_scope: minimal` with a surgical line-range spec (inlined
source snippets for the exact functions under review, not file-wide reads), and
require them to write their report to a file INCREMENTALLY (skeleton first, one
section at a time) so partial output survives timeouts. Then the principal
adjudicates the merged reports against the code.
