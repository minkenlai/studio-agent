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

    # Calculate boundary: keep the last N messages
    end_idx = max(0, len(session.messages) - n_keep)
    chunk = session.messages[session.last_consolidated : end_idx]

    logger.info(
        "cmd_compact: session=%s, total=%d, last_consolidated=%d, end_idx=%d, chunk_size=%d, n_keep=%d",
        session.key,
        len(session.messages),
        session.last_consolidated,
        end_idx,
        len(chunk),
        n_keep,
    )

    if not chunk:
        logger.info("cmd_compact: nothing to consolidate for session {}", session.key)
        return OutboundMessage(
            address=ctx.msg.address,
            content="Nothing to summarize (session is already fully compacted).",
        )

    # Use the existing consolidation pipeline
    summary = await loop.memory_consolidator.archive_messages(chunk)

    # Inject the summary as a prefix to the 'kept' messages and advance the pointer
    if summary:
        summary_msg = {
            "role": "assistant",
            "content": summary,
            "timestamp": datetime.now().isoformat(),
        }
        session.messages.insert(end_idx, summary_msg)
        # Set pointer to end_idx so the summary is the first message seen by get_history()
        session.last_consolidated = end_idx
    else:
        # Even if no summary is generated, we mark the chunk as consolidated
        session.last_consolidated = end_idx

    loop.sessions.save(session)

    logger.info(
        "cmd_compact: consolidated %d messages for session %s",
        len(chunk),
        session.key,
    )

    if n_keep > 0:
        reply = f"✅ Consolidated history, keeping last {n_keep} messages."
    else:
        reply = "✅ Consolidated all history."

    if summary:
        reply += f"\n\n---\n\n{summary}"

    return OutboundMessage(
        address=ctx.msg.address,
        content=reply,
    )


async def cmd_forget(ctx: CommandContext) -> OutboundMessage:
    """Forget all messages since the last consolidated marker."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)

    if session.last_consolidated >= len(session.messages):
        return OutboundMessage(
            address=ctx.msg.address,
            content="Nothing to forget. You are already at the last marker.",
        )

    num_forgotten = len(session.messages) - session.last_consolidated
    session.messages = session.messages[: session.last_consolidated]
    session.last_consolidated = len(session.messages)

    loop.sessions.save(session)
    loop.sessions.invalidate(session.key)

    return OutboundMessage(
        address=ctx.msg.address,
        content=f"Rewound to last marker. Forgot {num_forgotten} messages.",
    )


async def cmd_new(ctx: CommandContext) -> OutboundMessage:
    """Start a fresh session."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    snapshot = session.messages[session.last_consolidated :]
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
    router.exact("/help", cmd_help)
    router.exact("/status", cmd_status)
    router.exact("/fallback", cmd_fallback)
    router.exact("/speak", cmd_speak)
    router.priority("/stop", cmd_stop)
    router.priority("/repl", cmd_repl)
    router.priority("/restart", cmd_restart)
    router.priority("/halt", cmd_halt)
