# Unified Session Consolidation — Design Plan (v2)

> **v2 revisions (2026-08-26):** Converged on a single-pointer model. `forget_boundary`
> was dropped (redundant after always-drop; it also contained a visibility bug in v1 §6.3/§6.6).
> `last_consolidated` is replaced by `staged_boundary` (SB) plus a tag-scan for the pinned
> context prefix. Added the `/stage` command and the 3-case consolidation dispatch
> (free-drop / commit-and-drop / stage-only). STAGING.md is injected into the prompt so the
> staged tier is a real cross-session knowledge channel.

## Problem Statement

Current system has two parallel consolidation paths with divergent behavior:

| Aspect | `/compact` (manual) | Auto-pressure (`maybe_consolidate_by_tokens`) |
|--------|---------------------|-----------------------------------------------|
| Boundary selection | Naive: `messages[last_consolidated : end_idx]` | Smart: `pick_consolidation_boundary()` (user-turn boundaries) |
| Summary in session | ✅ Replaced in place (1 `_compact_summary` msg) | ❌ Summary goes to STAGING.md only |
| Messages dropped | ✅ Chunk deleted, summary spliced in | ❌ Only `last_consolidated` advanced; msgs retained (disk grows unbounded) |
| Token budget guard | ✅ 80% threshold check | ❌ None |
| Pointer model | `last_consolidated` points **at** the summary | `last_consolidated` advanced past retained msgs |

**Consequences:**
- Auto-pressure never frees stored memory — the session JSONL grows unbounded even though the
  prompt is bounded (the prefix before `last_consolidated` is dead in every code path).
- `/new` silently discards that dead prefix (no archive) — the system already treats it as disposable.
- STAGING.md was never in the prompt → the staged tier is not a working cross-session channel.
- `last_consolidated` conflates two different roles: "where the pinned context ends" and
  "what has already been distilled to the staging tier."

## Core Design Decisions (converged)

### D1 — Always drop (no retained dead prefix)
Once a chunk is successfully reduced (summary → HISTORY.md/STAGING.md), the raw messages are
deleted. The stored prefix is dead in every path (`get_history`, picker, `/forget`, `/new` all
start at/after it), and `save()` rewrites the whole JSONL each turn — a bloated prefix is O(n)
I/O tax for zero benefit. The "append-only for KV-cache" docstring does not protect it: the
prompt already starts after it, so dropping is cache-neutral.

### D2 — No `forget_boundary`; tag-scan replaces it
After an always-drop consolidation, the only messages left at the front are the injected
`session_context` message(s) (tagged). Their presence is already in the payload — the tag
(`_compact_summary`, to be renamed `_session_context`). "End of pinned prefix" = scan to the
first untagged message. A stored pointer carries no information the tags don't, and an
upward-pointer error would be **silent data loss**, whereas a tag-scan error is benign.

> **v1 bug fixed:** §6.3 inserted the context msg at FB and advanced FB *past* it, while §6.6
> started `get_history()` at FB — making the injected context invisible. The tag-scan model
> keeps the context prefix *inside* `get_history()` (it is intentional context), which is what
> the current working code does.

### D3 — Single stored pointer: `staged_boundary` (SB)
SB marks "everything before this index has already been distilled into STAGING.md." It is the
**only** stored pointer because "already staged" is a fact about the outside world (STAGING.md),
not about the message — it cannot be derived from tags (staged messages are untagged live msgs).

**Regions:**
```
0        ctx_end        SB            len
├────────────┼────────────────────┼──────────────┤
pinned ctx   staged, live         fresh
(tag-scan)   (distilled to         (never staged
(0..5 msgs)  STAGING.md, but       since SB)
             still in session)
```
- `ctx_end` = count of leading tagged context messages (tag-scan, **not** stored).
- Invariant: `0 ≤ ctx_end ≤ SB ≤ len(messages)`.
- **Self-healing:** every drop recomputes SB from the surviving layout, so drift cannot
  accumulate across rounds.

### D4 — Consolidation = three-case dispatch
The boundary picker is **unchanged** (starts at `ctx_end`, walks to a user-turn boundary that
removes enough tokens). The *action* branches on where the picked boundary `B` falls relative to
SB:

| Case | Condition | LLM work | Dropped | SB after |
|------|-----------|----------|---------|----------|
| **Free drop** | `B ≤ SB` | none (its `history_entry` already landed at stage time) | `[ctx_end : B]` | `ctx_end + (SB − B)` |
| **Commit + drop** | `B > SB` | summarize the *fresh* part only, `[SB : B]` → HISTORY+STAGING | `[ctx_end : B]` | `ctx_end_new` (reset) |
| **Stage only** (`/stage`) | no picker | stage `[SB : len−N]` → HISTORY+STAGING | nothing | `len − N` |

- **No double-staging:** free-drop writes nothing (already persisted); commit-and-drop stages
  only the fresh `[SB:B]`, never the already-staged `[ctx_end:SB]`.
- **Continuity on free-drop:** no `session_context` is injected (none produced; the staged
  region is *past* by definition). The agent keeps bearings from the live tail + STAGING.md in
  the prompt (D6).
- **Pressure is never stuck:** even when the fresh region is too small for the picker, the
  already-staged region `[ctx_end:SB]` can be reclaimed *immediately* and for free — this
  dissolves the "wait for fresh to grow" wrinkle.

### D5 — `/stage` (new command): share cross-session knowledge without amnesia
`/stage [N]` runs the consolidation LLM over the unstaged tail `[SB : len−N]`, writes
`history_entry` + `memory_update` to HISTORY.md/STAGING.md, **advances SB, drops nothing**, and
discards `session_context` (nothing was dropped, so the live conversation is still the context —
injecting a summary of live msgs would be noise).

This is the cross-session channel the staging tier was built for: once STAGING.md is in the
prompt (D6), staging conversation A's state makes it visible to conversation B **without**
`/new`-blasting A. The watermark (SB) makes staging idempotent — the same messages are never
distilled twice, which a nightly-synthesis dedup cannot guarantee (synthesis is an LLM, not a
diff, and re-staging produces differently-worded duplicates of the same facts).

**Trigger policy:** start **command-only** (human-gated). Staging is *public* to every session
once STAGING.md is prompted, so a blast radius exists that periodic consolidation doesn't.
An LLM `stage_memory` tool can be added later without touching the pointer model — SB is
trigger-agnostic.

**`/stage` constraint:** SB must never point mid tool-call exchange (a later free-drop would cut
mid-exchange). `/stage` therefore picks a user-turn boundary (same picker logic) or clamps via
`_find_legal_start`.

### D6 — STAGING.md in the prompt
`get_memory_context()` returns `MEMORY.md` **and** `STAGING.md`. This (a) makes consolidation
output immediately visible, and (b) is what turns the staged tier into a real cross-session
channel (D5). Periodic `memory-synthesize` merges STAGING.md → MEMORY.md and prunes STAGING.md,
keeping its token cost bounded.

### D7 — Consolidation output (unchanged from v1, 3 fields)
`save_memory` keeps `history_entry`, `memory_update`, and the optional `session_context`
(already implemented). `session_context` is the payload for **drops** only (continuity after
raw msgs are gone). `/stage` uses the same 3-field call but discards `session_context`.

### D8 — Command behavior (revised)

- **`/compact [N]`** — run picker + 3-case dispatch keeping the last `N` live messages.
  Same path as auto-pressure (unified). Returns the summary to the user.
- **`/forget`** — rewind to `ctx_end`: `messages = messages[:ctx_end]`, `SB = ctx_end`.
  Pinned context is preserved; the staged tail is already in STAGING.md, so dropping it is lossless.
- **`/new`** — snapshot = `messages[SB:]` (only the *unstaged* tail — the staged region is
  already persisted, re-archiving it would burn an LLM call producing a duplicate). Clear the
  session; background-archive the snapshot.
- **`/stage [N]`** — see D5.

### D9 — `get_history()` simplified
No pointer skip. Everything from index 0 is either pinned context (tagged, keep) or live.
```python
def get_history(self, max_messages=500):
    active = self.messages            # no [last_consolidated:] slice
    sliced = active[-max_messages:]
    # keep leading tagged context prefix; drop leading non-user untagged msgs
    # (existing _compact_summary/_session_context preservation logic stays)
    # _find_legal_start / _find_legal_end unchanged
```

### D10 — Migration
Load-time: `staged_boundary = metadata.staged_boundary ?? metadata.last_consolidated ?? 0`.
Existing dead prefixes (auto-pressure never dropped them) are trimmed by the next consolidation
or a one-shot backfill (deferred; see Q4). `last_consolidated` is removed from the JSONL
metadata line once the cutover lands.

## Implementation Plan

### Phase 1: Foundation (non-breaking)
1. **STAGING.md in prompt** — `MemoryStore.get_memory_context()` returns MEMORY.md + STAGING.md. (D6)
2. **Add `staged_boundary`** — additive field on `Session`; load-time migration from
   `last_consolidated`; written to JSONL. No behavior change yet. (D3, D10)
   (The `session_context` field in `save_memory` — D7 — is already done and tested.)

### Phase 2: Unified consolidation (the cutover)
3. **Refactor `maybe_consolidate_by_tokens`** — picker starts at `ctx_end`; 3-case dispatch;
   physically drop the chunk; recompute SB. (D1, D3, D4)
4. **Refactor `/compact` + `CompactHistoryTool`** — same dispatch, unified path. (D8)
5. **`get_history()`** — drop the `last_consolidated` slice; tag-scan the context prefix. (D9)

### Phase 3: Cleanup + new command
6. **Remove `last_consolidated`** — all refs → SB/tag-scan; `clear()`, `retain_recent_legal_suffix()`,
   `save()`, `_load()`; drop it from the JSONL metadata line.
7. **Update `/forget` and `/new`** to the D8 semantics.
8. **Add `/stage [N]`** — command + (optionally) 3-layer plumbing; SB-advancing, non-dropping. (D5)
9. **`retain_recent_legal_suffix()`** — adjust SB alongside message trimming.

### Phase 4: Tests
10. Update `test_session_manager_history.py`, `test_compact_tool.py`, `test_drop_inject.py`,
    `test_consolidation_full_cycle.py`, `test_forget_command.py`, `test_compact_command.py`,
    `test_loop_consolidation_tokens.py`, `test_consolidate_offset.py`.
11. New: `/stage` command test; 3-case dispatch unit tests; SB self-healing test; tag-scan test.

## Open Questions / Resolutions

| # | Question | Resolution |
|---|----------|------------|
| 1 | session_context quality control | Handled in the consolidation prompt (terse memory update vs. continuation details). |
| 2 | Multiple context msgs | Accumulate a single tagged prefix, cap ~5; excess folds back into the forgettable region. |
| 3 | STAGING.md token cap | Deferred — no cap. Nightly `memory-synthesize` + manual trigger; user sees bloat before it bites. |
| 4 | Migration of existing dead prefixes | Deferred — one-shot backfill during a maintenance pass: dump `[last_consolidated:]`-free prefixes to `sessions/archive/<key>/<ts>.jsonl`, then trim. |
| 5 | Cheaper consolidation model | Pre-existing optimization; follow-up by injecting a secondary "fast/cheap" model into `MemoryConsolidator`. |
| 6 | Should free-drop also re-stage its `history_entry`? | **No** — its `history_entry` already landed at stage time; rewriting would duplicate. |
| 7 | `/stage` trigger: command vs. LLM tool | **Command-only first** (blast-radius control); LLM tool later, trigger-agnostic to SB. |

## Ready for Implementation
Phase 1 proceeds now. Phase 2 is the cutover (highest care). Q4/Q5 are post-implementation.

## Implementation Status (2026-08-26)

- **Phase 1 — done** (`6807f54`): STAGING.md in prompt; `staged_boundary` field + load-time
  migration from `last_consolidated`; written to JSONL metadata.
- **Phase 2 — done** (split, as planned):
  - `98a2dbd` (2a+2b): `pick_consolidation_boundary` walks from `ctx_end` (tag-scan);
    `maybe_consolidate_by_tokens` 3-case dispatch (free drop / commit+drop / free-reclaim
    fallback when the fresh region is too small for the picker); context injection dual-tags
    `_session_context` + legacy `_compact_summary`, inserted at `ctx_end`.
  - `53ba5d0` (2c+2d): `get_history()` drops the `last_consolidated` slice (D9);
    `cmd_compact` + `CompactHistoryTool` unified through the same dispatch (compacting an
    already-staged tail is now a free drop — no LLM call).
  - Deviation from plan text: `Session` is now imported at runtime in `agent/memory.py`
    (required by the tag-scan; no import cycle).
  - Test baseline post-cutover: 1050 passed / 21 failed / 6 skipped — all 21 failures are in
    the 5 files Phase 4 rewrites (they assert pre-cutover retain/pointer semantics).
- **Phase 3 — done** (`e4c8c6c`): `last_consolidated` removed from `Session`
  (field, `clear()`, `retain_recent_legal_suffix()`, `save()`); `_load()` keeps
  the D10 migration (`staged_boundary` falls back to the legacy
  `last_consolidated` metadata key on load only). `/forget` rewinds to
  `ctx_end` (tag-scan) with `SB = ctx_end`; `/new` snapshots only
  `messages[SB:]`. `/stage [N]` added (D5): consolidates `[SB : legal cut]`,
  advances SB, drops nothing, discards `session_context`, and does not
  advance SB on consolidation failure (idempotent re-stage). Picker
  (`_pick_stage_boundary`) never lands SB mid tool-call exchange. Telegram
  3-layer plumbing added. Tests: `test_forget_command.py` rewritten for D8,
  `test_stage_command.py` added (12 tests).
  Suite: 1048 passed / 35 failed / 6 skipped — all 35 failures are in the 7
  stale files Phase 4 rewrites (pre-cutover retain/pointer/
  `last_consolidated` semantics).
- **Phase 4 — done**: all stale test files rewritten for post-cutover
  `staged_boundary`/tag-scan semantics; no test code asserts
  `last_consolidated` anywhere anymore.
  - Deleted `test_consolidate_offset.py` (only tested the removed field;
    `/new` archival semantics now covered by `test_new_command.py`).
  - `test_session_manager_history.py`: retain→SB rewrites + tag-scan
    (`context_prefix_len`, `get_history` leading-run rules) + D10
    load/save migration tests (real `SessionManager._load` on tmp JSONL).
  - `test_compact_command.py`: full rewrite — free drop / commit+drop /
    failure-no-loss / dual-tag injection via lightweight ctx mocks.
  - `test_compact_tool.py`: full rewrite (25 tests) — unified dispatch
    paths incl. 80% consolidator-context guard, pinned-prefix free drop,
    staged-boundary self-reset, no-injection when `_last_session_context`
    unset, repeated-call behavior.
  - `test_drop_inject.py`: full rewrite — injection only on commit+drop
    through the real `maybe_consolidate_by_tokens` dispatch.
  - `test_consolidation_full_cycle.py`: full rewrite — round 1 commit+drop
    injects one dual-tagged context msg; round 2 over the staged region is
    a free drop (no second LLM call, no stacking).
  - `test_loop_consolidation_tokens.py`: SB/physical-drop assertions
    replace pointer assertions (AgentLoop harness kept, preflight-order
    test intact).
  - New `test_new_command.py` (4 tests: D8 `/new` snapshots only
    `messages[SB:]`) and new `test_unified_dispatch.py` (8 tests: 3-case
    dispatch, free-reclaim fallback, consolidate-failure no-loss, SB
    self-heal on commit+drop, tag-scan pinned-prefix interactions).
  - Side fix: `MemoryConsolidator` factory tests pass
    `max_completion_tokens=128` so small context windows keep positive
    budgets (budget = window − completion − 1024).
  - Suite: **1078 passed / 6 skipped / 0 failed**; ruff check + format
    clean.
