from __future__ import annotations

import asyncio
import logging
import os
import sys
import traceback
from datetime import datetime
from typing import TYPE_CHECKING

from nanobot.bus.events import OutboundMessage

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nanobot.command import CommandContext
    from nanobot.command.router import CommandRouter


async def cmd_fallback(ctx: CommandContext) -> OutboundMessage:
    """Set or inspect the fallback provider slot index.

    Usage: /fallback [n]
    - No argument: show current provider chain and active slot.
    - With argument: set the active slot to index n.
    """
    from nanobot.providers.fallback import FallbackClient

    if ctx.loop is None:
        return OutboundMessage(
            address=ctx.msg.address,
            content="Error: agent loop not available.",
        )

    pinned_profile = ctx.msg.metadata.get("agent_profile")
    if pinned_profile:
        provider = ctx.loop.registry.get_provider(pinned_profile)
    else:
        provider = ctx.loop.provider

    if not isinstance(provider, FallbackClient):
        return OutboundMessage(
            address=ctx.msg.address,
            content="Current provider is not a FallbackClient.",
        )

    args = ctx.args.strip()
    if not args:
        # Show current status
        idx = provider._active_index
        slot = provider._slots[idx]
        lines = [
            f"Provider: {slot[2]}",
            f"Active slot: {idx}",
            "All slots:",
        ]
        for i, s in enumerate(provider._slots):
            marker = " <--" if i == idx else ""
            lines.append(f"  [{i}] {s[2]} ({s[1]}){marker}")
        return OutboundMessage(
            address=ctx.msg.address,
            content="\n".join(lines),
        )

    # Parse and set new index
    try:
        n = int(args)
    except ValueError:
        return OutboundMessage(
            address=ctx.msg.address,
            content=f"Error: expected integer, got '{args}'",
        )

    total = len(provider._slots)
    if n < 0 or n >= total:
        return OutboundMessage(
            address=ctx.msg.address,
            content=f"Error: slot index must be 0-{total - 1}, got {n}",
        )

    provider._active_index = n
    slot = provider._slots[n]
    return OutboundMessage(
        address=ctx.msg.address,
        content=f"Set provider to slot [{n}] {slot[2]} ({slot[1]})",
    )


async def cmd_speak(ctx: CommandContext) -> OutboundMessage:
    """Synthesize text to speech.

    Usage:
      /speak "Hello world"   — synthesize the given text
      /speak                 — synthesize the last agent message from session
    """
    from nanobot.services.tts_service import TTSService

    msg = ctx.msg
    loop = ctx.loop

    if loop is None:
        return OutboundMessage(address=msg.address, content="Error: agent loop not available.")

    tts_service = TTSService(loop)

    text = ctx.args.strip() or None  # empty string → None triggers last-message fallback
    ok, result = await tts_service.speak(
        text=text, session_key=ctx.key if hasattr(ctx, "key") else None, msg_address=msg.address
    )

    if not ok:
        return OutboundMessage(
            address=msg.address,
            content=f"🔇 TTS failed: {result}",
        )

    return OutboundMessage(
        address=msg.address,
        content="",
        media=[result],
    )


async def cmd_help(ctx: CommandContext) -> OutboundMessage:
    """Show help for builtin commands."""
    msg = ctx.msg
    content = "**Available Commands:**\n"
    content += "- `/new`: Start a fresh conversation (clears context)\n"
    content += "- `/compact [N]`: Consolidate history, keeping last N messages (default all)\n"
    content += "- `/stage [N]`: Stage unstaged messages to memory, keeping last N live (default all; drops nothing)\n"
    content += "- `/help`: Show this help\n"
    content += "- `/status`: Show system status and version\n"
    content += "- `/fallback [n]`: Inspect or set the fallback provider slot index\n"
    content += "- `/tasks`: List current and recent background tasks\n"
    content += "- `/usage`: Show token usage for the current session\n"
    content += "- `/stop`: Cancel all active tasks in this session\n"
    content += "- `/speak [msg]`: Synthesize text to speech (no args = speak last agent message)\n"
    content += "- `/repl`: Execute Python code in the current environment\n"
    content += "- `/restart`: Refresh code in-place (preserves PID)\n"
    content += "- `/restart --full`: Graceful full reboot (requires systemd)\n"
    content += "- `/halt`: Stop the gateway process (Manual mode only)\n"
    return OutboundMessage(address=msg.address, content=content)


async def cmd_status(ctx: CommandContext) -> OutboundMessage:
    """Show system status and version."""
    import time

    msg = ctx.msg
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)

    sha = "unknown"
    branch = "unknown"
    try:
        import subprocess

        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=loop.workspace / "nanobot-dev", text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=loop.workspace / "nanobot-dev",
            text=True,
        ).strip()
    except Exception:
        pass

    status = "🟢 Running"
    if loop.is_supervised:
        status += " (Supervised)"
    else:
        status += " (Manual)"

    uptime_s = int(time.time() - getattr(loop, "_start_time", time.time()))
    uptime_str = f"{uptime_s // 60}m {uptime_s % 60}s"

    usage = loop.get_usage(msg.session_key)

    agent_id = session.metadata.get("agent", "defaults")
    pinned_profile = msg.metadata.get("agent_profile")

    window = getattr(loop, "context_window_tokens", 64000)
    if pinned_profile:
        profile_limit = None
        try:
            profile_cfg = loop.config.agents.get_agent(pinned_profile)
            profile_limit = getattr(profile_cfg, "context_window_tokens", None)
        except (ValueError, AttributeError):
            pass

        try:
            from nanobot.utils.helpers import resolve_context_window

            runner = loop.registry.get_runner(pinned_profile)
            provider_max = getattr(runner.provider.generation, "context_max", None)
            window = resolve_context_window(profile_limit, provider_max, window)
        except Exception:
            window = profile_limit or window

    try:
        context_tokens, _ = loop.memory_consolidator.estimate_session_prompt_tokens(session)
        pct = int(context_tokens / window * 100) if window else 0
        context_str = f"{context_tokens // 1000}k/{window // 1000}k ({pct}%)"
    except Exception:
        context_str = "unknown"

    running_tasks = getattr(loop.subagents, "get_running_count", lambda: 0)()

    if pinned_profile:
        agent_provider = loop.registry.get_provider(pinned_profile)
    else:
        agent_provider = loop.provider

    from nanobot.providers.fallback import FallbackClient

    if isinstance(agent_provider, FallbackClient):
        model_id = agent_provider.active_identifier
        model_name = agent_provider.active_model
        fallbacks = agent_provider.fallback_identifiers
        if len(fallbacks) > 1:
            chain = []
            for fid in fallbacks:
                if fid == model_id:
                    chain.append(f"**{fid}**")
                else:
                    chain.append(fid)
            model_info = f"`{model_name}` ({' → '.join(chain)})"
        else:
            model_info = f"`{model_name}` (`{model_id}`)"
    else:
        model_name = getattr(agent_provider, "get_default_model", lambda: loop.model)()
        model_id = "unknown"
        if loop.config and loop.config.models:
            for k, v in loop.config.models.items():
                if v.model == model_name:
                    model_id = k
                    break
        model_info = f"`{model_name}` (`{model_id}`)"

    content = f"**System Status:** {status}\n"
    content += f"**Branch:** `{branch}`\n"
    content += f"**Commit:** `{sha}`\n"
    content += f"**PID:** `{os.getpid()}`\n\n"
    content += f"**Agent:** `{agent_id}`\n"
    if pinned_profile:
        content += f"**Pinned Profile:** `{pinned_profile}`\n"
    content += f"**Model:** {model_info}\n"
    content += (
        f"**Tokens:** {usage.get('input_tokens', 0):,} in / {usage.get('output_tokens', 0):,} out\n"
        f"**Context:** {context_str}\n"
        f"**Session:** {len(session.messages)} messages\n"
        f"**Tasks:** {running_tasks} active\n"
        f"**Uptime:** {uptime_str}"
    )

    return OutboundMessage(address=msg.address, content=content, metadata={"render_as": "text"})


async def cmd_stop(ctx: CommandContext) -> OutboundMessage:
    """Stop all active tasks in the current session."""
    from loguru import logger

    logger.error(
        f"CMD STOP session_key: {ctx.msg.session_key}, active_tasks keys: {list(ctx.loop._active_tasks.keys())}"
    )
    msg = ctx.msg
    loop = ctx.loop
    cancelled = await loop.stop_tasks_by_session(msg.session_key)
    sub_cancelled = await loop.subagents.cancel_by_session(msg.session_key)
    total = cancelled + sub_cancelled
    content = f"Stopped {total} task(s)." if total else "No active task to stop."
    return OutboundMessage(address=msg.address, content=content)


async def cmd_restart(ctx: CommandContext) -> OutboundMessage:
    """Restart the process (in-place or full via systemd)."""
    msg = ctx.msg
    loop = ctx.loop
    raw = ctx.raw or ""
    args = raw.split()
    is_full = "--full" in args or "-f" in args

    if is_full and not loop.is_supervised:
        return OutboundMessage(
            address=msg.address,
            content="🛑 **Abort:** I am not running under a supervisor (systemd). "
            "Use `/restart` (in-place) to refresh code, or `/shutdown` to stop.",
        )

    async def _do_restart():
        log_file = loop.workspace / "logs" / "lifecycle.log"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        target = msg.address.to_uri()

        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("a", encoding="utf-8") as f:
                type_str = "FULL" if is_full else "IN-PLACE"
                f.write(f"[{now}] SHUTDOWN: RESTART_REQ {target} ({type_str})\n")
        except Exception:
            pass

        await asyncio.sleep(1)

        if is_full:
            loop.stop()
        else:
            os.execv(sys.executable, [sys.executable, "-m", "nanobot"] + sys.argv[1:])

    asyncio.create_task(_do_restart())
    content = (
        "🔄 Restarting (full)... See you in a moment!" if is_full else "🔄 Restarting in-place..."
    )
    return OutboundMessage(address=msg.address, content=content)


async def cmd_halt(ctx: CommandContext) -> OutboundMessage:
    """Gracefully shut down the process."""
    msg = ctx.msg
    loop = ctx.loop

    if loop.is_supervised:
        return OutboundMessage(
            address=msg.address,
            content="🛑 **Abort:** I am running under systemd. "
            "Please stop the service via systemd if you want me to stay off.",
        )

    async def _do_halt():
        await asyncio.sleep(1)
        loop.stop()

    asyncio.create_task(_do_halt())
    return OutboundMessage(address=msg.address, content="👋 Goodbye! Halting...")


async def cmd_compact(ctx: CommandContext) -> OutboundMessage:
    """Manually consolidate session history, keeping the last N messages.
    Default N=0 means summarize ALL messages.

    The summary is injected as a new assistant message so the conversation
    can continue from the compacted state.
    """
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)

    # Parse N from raw input (default 0 = compact everything)
    n_keep = 0
    parts = (ctx.raw or "").split()
    if len(parts) > 1:
        try:
            n_keep = abs(int(parts[1]))
        except ValueError:
            pass

    # Unified dispatch (D4), shared with the compact_history tool and the
    # token-budget auto-consolidation: drop [ctx_end : end_idx] where
    #   free drop     — end_idx ≤ SB (already staged, no LLM work)
    #   commit + drop — end_idx >  SB (80% guard + distill fresh [SB:end_idx])
    # ctx_end = end of the pinned tagged context prefix (tag-scan).
    from nanobot.session.manager import Session

    ctx_end = Session.context_prefix_len(session.messages)
    end_idx = max(0, len(session.messages) - n_keep)

    logger.info(
        "cmd_compact: session=%s, total=%d, ctx_end=%d, staged_boundary=%d, end_idx=%d, n_keep=%d",
        session.key,
        len(session.messages),
        ctx_end,
        session.staged_boundary,
        end_idx,
        n_keep,
    )

    result = await loop.memory_consolidator.consolidate_range(session, end_idx)

    if result.error:
        return OutboundMessage(address=ctx.msg.address, content=result.error)

    if result.consolidated:
        if n_keep > 0:
            reply = f"✅ Consolidated history, keeping last {n_keep} messages."
        else:
            reply = "✅ Consolidated all history."
        reply += f"\n\n---\n\n{result.summary}"
        return OutboundMessage(address=ctx.msg.address, content=reply)

    # Free drop: [ctx_end:end_idx] was already distilled into STAGING.md
    # at stage time — reclaimed without any LLM call.
    return OutboundMessage(
        address=ctx.msg.address,
        content=f"✅ Dropped {result.dropped} already-staged message(s), keeping last {n_keep}.",
    )


async def cmd_forget(ctx: CommandContext) -> OutboundMessage:
    """Forget everything after the pinned context prefix (D8).

    ``messages = messages[:ctx_end]``, ``SB = ctx_end``. The staged region is
    already in STAGING.md, so dropping it is lossless; the fresh tail is
    intentionally forgotten.
    """
    from nanobot.session.manager import Session

    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)

    # Serialize with background/token-budget consolidation (F1): the staged
    # region it reclaims is defined by the same SB this command resets.
    async with loop.memory_consolidator.get_lock(session.key):
        ctx_end = Session.context_prefix_len(session.messages)
        num_forgotten = len(session.messages) - ctx_end
        session.messages = session.messages[:ctx_end]
        session.staged_boundary = ctx_end

        loop.sessions.save(session)
        loop.sessions.invalidate(session.key)

    return OutboundMessage(
        address=ctx.msg.address,
        content=f"Rewound to pinned context. Forgot {num_forgotten} message(s).",
    )


def _pick_stage_boundary(messages: list[dict], sb: int, desired: int) -> int | None:
    """Pick a legal cut point for advancing SB toward ``desired`` (D5).

    A cut at index k is legal iff no tool-call exchange straddles it — i.e.
    every tool_call declared in ``messages[:k]`` is resolved there. A later
    free-drop may reclaim ``[ctx_end : k]``, so the live suffix must start
    clean.

    Preference order:
    1. Largest user-turn cut at or below ``desired`` (picker logic).
    2. Largest legal cut below ``desired`` (``_find_legal_start``-style clamp:
       stage slightly less, keep more live).
    3. First legal cut at or above ``desired`` (stage slightly more).
    Returns None if no legal cut exists above SB (tail is one open exchange).
    """
    total = len(messages)
    desired = min(max(desired, sb), total)

    def balanced(k: int) -> bool:
        declared: set[str] = set()
        for i in range(k):
            msg = messages[i]
            role = msg.get("role")
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        declared.add(str(tc["id"]))
            elif role == "tool":
                tid = msg.get("tool_call_id")
                if tid:
                    declared.discard(str(tid))
        return not declared

    # Backward: largest user-turn cut at or below desired (never scan index total).
    for u in range(min(desired, total - 1), sb, -1):
        if messages[u].get("role") == "user" and balanced(u):
            return u
    # Backward clamp: largest legal cut below desired.
    for k in range(min(desired, total) - 1, sb, -1):
        if balanced(k):
            return k
    # Forward: first legal cut at or above desired (stage slightly more).
    for k in range(max(desired, sb), total + 1):
        if balanced(k):
            return k
    return None


async def cmd_stage(ctx: CommandContext) -> OutboundMessage:
    """Stage the unstaged tail into persistent memory, keeping last N live (D5).

    Runs the consolidation LLM over ``[SB : len-N]``, writes to HISTORY.md and
    STAGING.md, and advances the staged boundary — **drops nothing**, so the
    live conversation stays the working context. ``session_context`` is
    discarded (nothing was dropped; a summary of live msgs would be noise).

    Usage: /stage [N]  (N=0 stages everything not yet staged)
    """
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)

    # Parse N from raw input (default 0 = stage everything unstaged)
    n_keep = 0
    parts = (ctx.raw or "").split()
    if len(parts) > 1:
        try:
            n_keep = abs(int(parts[1]))
        except ValueError:
            pass

    # Same consolidation pipeline as /compact; session_context is discarded
    # by design — the live messages are still the context. The boundary pick
    # + chunk slice run INSIDE the lock (B-1): a concurrent drop between a
    # pre-lock read and the lock would let a stale ``new_sb`` be written
    # against a shrunken list (SB > len) and later free-drops/`/new` would
    # delete live messages as if they were already staged.
    async with loop.memory_consolidator.get_lock(session.key):
        sb = session.staged_boundary
        desired = len(session.messages) - n_keep

        if desired <= sb:
            return OutboundMessage(
                address=ctx.msg.address,
                content="Nothing to stage (no unstaged messages).",
            )

        # D5 constraint: SB must never land mid tool-call exchange — a later
        # free-drop would then cut the live history mid-exchange.
        new_sb = _pick_stage_boundary(session.messages, sb, desired)
        if new_sb is None or new_sb <= sb:
            return OutboundMessage(
                address=ctx.msg.address,
                content=(
                    "⚠️ No safe staging boundary found (tail is mid tool-call exchange). "
                    "Try again once the current exchange completes."
                ),
            )

        chunk = session.messages[sb:new_sb]
        logger.info(
            "cmd_stage: session=%s, total=%d, staged_boundary=%d, desired=%d, new_boundary=%d, chunk_size=%d, n_keep=%d",
            session.key,
            len(session.messages),
            sb,
            desired,
            new_sb,
            len(chunk),
            n_keep,
        )

        summary = await loop.memory_consolidator.archive_messages(chunk)

        if not summary:
            # Summary generation failed — do NOT advance SB (idempotency: the
            # same messages stay stageable, nothing is lost or double-staged).
            logger.warning(
                "cmd_stage: summary generation failed for %d messages in session %s — "
                "staged boundary not advanced",
                len(chunk),
                session.key,
            )
            return OutboundMessage(
                address=ctx.msg.address,
                content=f"⚠️ Stage failed: summary generation returned no result for {len(chunk)} messages. "
                f"Nothing was staged. This may happen if the content exceeds the model's context window.",
            )

        session.staged_boundary = new_sb
        loop.sessions.save(session)

    reply = f"✅ Staged {len(chunk)} message(s) to memory (SB → {new_sb}). Nothing dropped."
    if new_sb != desired:
        if new_sb > desired:
            reply += f" (Boundary snapped to a clean cut point; {new_sb - desired} message(s) kept live.)"
        else:
            reply += f" (Boundary clamped to a clean cut point; {desired - new_sb} message(s) remain unstaged.)"
    reply += f"\n\n---\n\n{summary}"
    return OutboundMessage(address=ctx.msg.address, content=reply)


async def cmd_new(ctx: CommandContext) -> OutboundMessage:
    """Start a fresh session (D8).

    Only the *unstaged* tail ``messages[SB:]`` is archived — the staged region
    is already in STAGING.md, re-archiving it would duplicate.
    """
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    # Serialize with background/token-budget consolidation (F1): the archived
    # tail is defined by SB, which the consolidator may be moving. The
    # background archive itself runs OUTSIDE the lock (it acquires its own).
    async with loop.memory_consolidator.get_lock(session.key):
        snapshot = session.messages[session.staged_boundary :]
        session.clear()
        loop.sessions.save(session)
        loop.sessions.invalidate(session.key)
    if snapshot:
        loop._schedule_background(loop.memory_consolidator.archive_messages(snapshot))
    return OutboundMessage(
        address=ctx.msg.address,
        content="New session started.",
    )


async def cmd_repl(ctx: CommandContext) -> OutboundMessage:
    """Execute Python code in the current environment."""
    msg = ctx.msg
    loop = ctx.loop
    # Extract code from raw input to preserve whitespace
    raw_input = (ctx.raw or "").strip()
    if raw_input.startswith("/"):
        # Strip the command part (e.g., "/repl ")
        parts = raw_input.split(None, 1)
        code = parts[1] if len(parts) > 1 else ""
    else:
        code = raw_input

    repl_cfg = loop.config.repl

    # Check user-based ACL
    user_key = f"{msg.address.channel}:{msg.sender_id}"
    if user_key not in repl_cfg.allow_users:
        from loguru import logger

        logger.warning(
            "REPL access denied for user: {}. Allowed users: {}",
            user_key,
            repl_cfg.allow_users,
        )
        return OutboundMessage(
            address=msg.address,
            content=f"🛑 **Access Denied:** You are not authorized to use the REPL command.\n\n"
            f"Your identity string is: `{user_key}`\n\n"
            "Add this string to `repl.allowUsers` in `config.json` to enable it.",
        )

    if not code.strip():
        # If no code is provided, send the repl_recipes.md file if it exists.
        recipes_path = loop.workspace / "repl_recipes.md"
        if recipes_path.exists():
            return OutboundMessage(
                address=msg.address, content="📚 **REPL Recipes**", media=[str(recipes_path)]
            )
        return OutboundMessage(address=msg.address, content="Usage: `/repl <python statement>`")

    log_file = loop.workspace / "logs" / "repl.log"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    target = msg.address.to_uri()

    # Prepare environment for execution
    env = {
        "ctx": ctx,
        "loop": loop,
        "session": ctx.session,
        "msg": msg,
        "asyncio": asyncio,
        "os": os,
        "sys": sys,
    }

    import io
    from contextlib import redirect_stdout

    output = ""
    stdout_buf = io.StringIO()
    try:
        with redirect_stdout(stdout_buf):
            # Try to evaluate as an expression first
            try:
                result = eval(code, env)
                if asyncio.iscoroutine(result):
                    result = await result
                output = repr(result)
            except SyntaxError:
                # If eval fails, it might be a statement
                exec(code, env)
                if "result" in env:
                    result = env["result"]
                    if asyncio.iscoroutine(result):
                        result = await result
                    output = repr(result)
                else:
                    output = "Done (no result variable set)"
    except Exception:
        output = traceback.format_exc()

    stdout_val = stdout_buf.getvalue()
    if stdout_val:
        output = f"--- stdout ---\n{stdout_val}\n--- return ---\n{output}"

    # Log to repl.log
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"[{now}] REPL {target}\nIN:  {code}\nOUT: {output}\n{'-' * 40}\n")
    except Exception:
        pass

    return OutboundMessage(
        address=msg.address,
        content=f"**REPL Output:**\n```python\n{output}\n```",
    )


def register_builtin_commands(router: CommandRouter) -> None:
    """Register all builtin commands."""
    router.exact("/new", cmd_new)
    router.exact("/forget", cmd_forget)
    router.exact("/compact", cmd_compact)
    router.exact("/stage", cmd_stage)
    router.exact("/help", cmd_help)
    router.exact("/status", cmd_status)
    router.exact("/fallback", cmd_fallback)
    router.exact("/speak", cmd_speak)
    router.priority("/stop", cmd_stop)
    router.priority("/repl", cmd_repl)
    router.priority("/restart", cmd_restart)
    router.priority("/halt", cmd_halt)
