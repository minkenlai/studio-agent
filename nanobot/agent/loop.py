"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from contextlib import AsyncExitStack, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.memory import MemoryConsolidator
from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.compact import CompactHistoryTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.speak import SpeakTool
from nanobot.agent.tools.task_status import TaskStatusTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import Address, InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.command import CommandContext, CommandRouter, register_builtin_commands
from nanobot.config.schema import Config
from nanobot.providers.base import LLMClient
from nanobot.providers.factory import AgentRegistry
from nanobot.session.manager import Session, SessionManager
from nanobot.utils.calibration import ModelCalibrationManager

# Lazy import for TYPE_CHECKING to break circular deps
if TYPE_CHECKING:
    from nanobot.providers.transcription import GroqTranscriptionProvider

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig, WebSearchConfig
    from nanobot.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 16_000

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMClient,
        workspace: Path,
        config: Config | None = None,
        model: str | None = None,
        max_iterations: int = 40,
        context_window_tokens: int = 65_536,
        web_search_config: WebSearchConfig | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        timezone: str | None = None,
        registry: AgentRegistry | None = None,
    ):
        from nanobot.config.schema import Config, ExecToolConfig, WebSearchConfig

        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.config = config or Config()
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.web_search_config = web_search_config or WebSearchConfig()
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self._start_time = time.time()
        self._last_usage: dict[str, int] = {}

        self.context = ContextBuilder(workspace, timezone=timezone)
        self.sessions = session_manager or SessionManager(workspace)
        self.calibration_mgr = ModelCalibrationManager()
        self.tools = ToolRegistry()
        self.registry = registry or AgentRegistry(self.config)

        # For the default runner, we use the provider passed in (if any),

        # otherwise we use the registry to get the default runner.
        if provider:
            self.runner = AgentRunner(provider)
        else:
            self.runner = self.registry.get_runner("defaults")

        self.subagents = SubagentManager(
            config=self.config,
            workspace=workspace,
            bus=bus,
            web_search_config=self.web_search_config,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
            registry=self.registry,
            provider=self.runner.provider,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._background_tasks: list[asyncio.Task] = []
        self._session_locks: dict[str, asyncio.Lock] = {}
        # NANOBOT_MAX_CONCURRENT_REQUESTS: <=0 means unlimited; default 3.
        _max = int(os.environ.get("NANOBOT_MAX_CONCURRENT_REQUESTS", "3"))
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_max) if _max > 0 else None
        )
        # Override context_window_tokens with provider's max if available
        provider_context_max = (
            getattr(self.runner.provider.generation, "context_max", None)
            if hasattr(self.runner.provider, "generation")
            else None
        )
        effective_context_window = (
            provider_context_max if provider_context_max is not None else context_window_tokens
        )
        self.context_window_tokens = effective_context_window

        self._transcription_provider: GroqTranscriptionProvider | None = None
        self._transcription_provider_lock = asyncio.Lock()

        # Lazy TTS client
        self._tts_client = None
        self._tts_client_lock = asyncio.Lock()

        self.memory_consolidator = MemoryConsolidator(
            workspace=workspace,
            provider=provider or self.runner.provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=effective_context_window,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
            max_completion_tokens=provider.generation.max_tokens,
        )
        self._register_default_tools()
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None
        self.tools.register(
            ReadFileTool(
                workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read
            )
        )
        for cls in (WriteFileTool, EditFileTool, ListDirTool, SearchFilesTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        if self.exec_config.enable:
            self.tools.register(
                ExecTool(
                    working_dir=str(self.workspace),
                    timeout=self.exec_config.timeout,
                    restrict_to_workspace=self.restrict_to_workspace,
                    path_append=self.exec_config.path_append,
                )
            )
        self.tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        self.tools.register(TaskStatusTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(
                CronTool(self.cron_service, default_timezone=self.context.timezone or "UTC")
            )
        self.tools.register(CompactHistoryTool(self.memory_consolidator, self.sessions))
        self.tools.register(SpeakTool(self))

    async def _get_transcription_provider(self) -> GroqTranscriptionProvider | None:
        """Lazily initialize the Groq transcription provider from config."""
        if self._transcription_provider is not None:
            return self._transcription_provider

        async with self._transcription_provider_lock:
            if self._transcription_provider is not None:
                return self._transcription_provider

            from nanobot.providers.transcription import GroqTranscriptionProvider

            api_key = None
            if self.config and self.config.providers:
                api_key = (
                    getattr(self.config.providers, "groq", None).api_key
                    if hasattr(self.config.providers, "groq")
                    else None
                )

            if not api_key:
                import os

                api_key = os.environ.get("GROQ_API_KEY")

            if api_key:
                self._transcription_provider = GroqTranscriptionProvider(api_key=api_key)
                logger.debug("Transcription provider initialized (Groq)")
            else:
                logger.warning(
                    "No Groq API key configured — audio transcription fallback will be unavailable"
                )
                self._transcription_provider = GroqTranscriptionProvider(api_key=None)

        return self._transcription_provider

    async def get_tts_client(self):
        """Lazily initialize the TTS client from config."""
        if self._tts_client is not None:
            return self._tts_client

        async with self._tts_client_lock:
            if self._tts_client is not None:
                return self._tts_client

            from nanobot.providers.tts import make_tts_provider

            if self.config and self.config.tts:
                try:
                    self._tts_client = make_tts_provider(self.config.tts)
                    logger.info("TTS client initialized (provider={})", self.config.tts.provider)
                except Exception as e:
                    logger.warning("Failed to initialize TTS client: {}", e)
            else:
                logger.warning("No TTS configuration found")

        return self._tts_client

    async def _route_audio(
        self,
        messages: list[dict[str, Any]],
        model: str,
        provider: LLMClient | None = None,
    ) -> list[dict[str, Any]]:
        """Route audio payloads based on the target model's capabilities.

        **Native path** — model supports audio. Pass messages through untouched.

        **Transcription path** — model doesn't support audio. Decode base64 audio
        blocks to temp files, send to Groq Whisper, replace blocks with transcribed
        text. Returns modified messages (or original if no audio found / no provider).
        """
        from nanobot.providers.registry import get_capabilities
        from nanobot.utils.helpers import has_audio_content

        if not has_audio_content(messages):
            return messages

        # If both the LLMClient and the model support native audio, then keep audio blocks
        if provider and provider.supports_native_audio and get_capabilities(model).audio:
            logger.debug("Model '{}' natively supports audio — passing through", model)
            return messages

        # Transcription path
        trans_provider = await self._get_transcription_provider()
        if trans_provider is None:
            logger.warning(
                "Audio content detected but no transcription provider available; "
                "replacing audio blocks with placeholders"
            )
            return LLMClient._strip_audio_content(messages) or messages
        return await self._transcribe_audio_blocks(messages, model, trans_provider)

    async def _transcribe_audio_blocks(
        self,
        messages: list[dict[str, Any]],
        model: str,
        trans_provider: GroqTranscriptionProvider,
    ) -> list[dict[str, Any]]:
        """Transcribe all audio blocks in *messages* via Groq Whisper."""
        import base64
        import tempfile

        # Collect all audio blocks
        audio_blocks: list[dict] = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "audio":
                    audio_blocks.append(block)

        if not audio_blocks:
            return messages

        logger.info(
            "Model '{}' does not support audio — routing {} audio block(s) through Groq Whisper",
            model,
            len(audio_blocks),
        )

        # Transcribe all audio blocks concurrently
        async def _transcribe_block(block: dict) -> str:
            """Decode a base64 audio block, write to temp file, transcribe."""
            data_field = block.get("data", "")
            mime = block.get("mime_type", "audio/ogg")

            # Strip "data:<mime>;base64," prefix
            if data_field.startswith("data:"):
                comma_idx = data_field.find(",")
                if comma_idx > 0:
                    data_field = data_field[comma_idx + 1 :]

            raw_bytes = base64.b64decode(data_field)

            # Determine extension from MIME
            ext_map = {
                "audio/ogg": ".ogg",
                "audio/mpeg": ".mp3",
                "audio/wav": ".wav",
                "audio/flac": ".flac",
                "audio/mp4": ".m4a",
                "audio/webm": ".webm",
            }
            ext = ext_map.get(mime, ".ogg")

            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(raw_bytes)
                tmp_path = tmp.name

            try:
                text = await trans_provider.transcribe(tmp_path)
                if not text:
                    path = (block.get("_meta") or {}).get("path", "")
                    return f"[audio transcription failed: {path or 'unknown'}]"
                return text
            finally:
                try:
                    import os

                    os.unlink(tmp_path)
                except OSError:
                    pass

        # Transcribe all blocks concurrently
        transcribed_texts: list[str] = await asyncio.gather(
            *[_transcribe_block(block) for block in audio_blocks]
        )

        # Replace audio blocks in messages
        modified = []
        text_idx = 0
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                modified.append(msg)
                continue

            new_content = []
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "audio"
                    and text_idx < len(transcribed_texts)
                ):
                    new_content.append(
                        {
                            "type": "text",
                            "text": f"[Transcribed Audio]: {transcribed_texts[text_idx]}",
                        }
                    )
                    text_idx += 1
                else:
                    new_content.append(block)

            modified.append({**msg, "content": new_content})

        logger.debug("Audio routing complete — {} audio block(s) transcribed", text_idx)
        return modified

    @property
    def is_supervised(self) -> bool:
        """Check if the process is running under a supervisor (e.g. systemd)."""
        import os

        return "INVOCATION_ID" in os.environ

    async def stop_tasks_by_session(self, session_key: str) -> int:
        """Cancel all active tasks for a given session."""
        tasks = self._active_tasks.get(session_key, [])
        if not tasks:
            return 0

        count = len(tasks)
        for task in list(tasks):
            if not task.done():
                task.cancel()

        return count

    def get_usage(self, session_key: str) -> dict[str, int]:
        """Get total token usage for the given session."""
        session = self.sessions.get_or_create(session_key)
        return session.metadata.get("usage", {})

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers

        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except BaseException as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(
        self,
        address: Address,
        session_key: str,
        message_id: str | None = None,
        agent_profile: str | None = None,
    ) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron", "compact_history", "speak"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    if name == "message":
                        tool.set_context(address, message_id)
                    elif name == "spawn":
                        # Pass the current agent profile to the spawn tool
                        if hasattr(tool, "set_agent_profile"):
                            tool.set_agent_profile(agent_profile)
                        tool.set_context(address, session_key)
                    else:
                        tool.set_context(address, session_key)

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        from nanobot.utils.helpers import strip_think

        return strip_think(text) or None

    async def _audit_lifecycle(self) -> None:
        """Log startup to lifecycle.log and check for pending notifications."""
        from datetime import datetime

        log_file = self.workspace / "logs" / "lifecycle.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # 1. Get current Git state
        try:
            # Use the directory of this file to find the nearest Git repository
            from pathlib import Path

            source_dir = Path(__file__).resolve().parent
            branch = (
                subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=source_dir
                )
                .decode()
                .strip()
            )
            sha = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=source_dir)
                .decode()
                .strip()
            )
        except Exception:
            branch, sha = "unknown", "unknown"

        # 2. Check for pending notification from a SHUTDOWN entry
        pending_notification = None
        if log_file.exists():
            try:
                # Read last 20 lines to find the last SHUTDOWN
                lines = log_file.read_text(encoding="utf-8").splitlines()[-20:]
                for line in reversed(lines):
                    if "SHUTDOWN:" in line and "RESTART_REQ" in line:
                        # Format: [timestamp] SHUTDOWN: RESTART_REQ {URI} to branch {branch}
                        parts = line.split("RESTART_REQ")[-1].strip().split()
                        if parts:
                            uri = parts[0]
                            try:
                                pending_notification = Address.from_uri(uri)
                                logger.debug(
                                    "Found pending notification target: {}", pending_notification
                                )
                            except Exception as e:
                                logger.warning("Failed to parse URI from log {}: {}", uri, e)
                        break
            except Exception as e:
                logger.error("Error reading lifecycle log for notifications: {}", e)

        # 3. Log current START
        import os

        pid = os.getpid()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"[{now}] START: branch={branch} sha={sha} pid={pid}\n")

        # 4. If a notification was pending, send it
        if pending_notification:
            try:
                msg = "🔄 **System bootstrapped successfully**\n"
                msg += f"**Branch:** `{branch}`\n"
                msg += f"**Commit:** `{sha}`"

                # Check for pending memory recovery files (top-level payload JSONs only;
                # ignore sub-dirs like assets/ which may exist even when empty)
                recovery_dir = self.workspace / "memory" / "recovery"
                recovery_files = (
                    sorted(p.name for p in recovery_dir.glob("*.json") if p.is_file())
                    if recovery_dir.exists()
                    else []
                )
                if recovery_files:
                    files_list = ", ".join(f"`{name}`" for name in recovery_files)
                    msg += (
                        f"\n\n⚠️ **Pending recovery data found** in `memory/recovery/` "
                        f"({files_list}). I can integrate these files into your history if you'd like."
                    )

                await self.bus.publish_outbound(
                    OutboundMessage(
                        address=pending_notification,
                        content=msg,
                    )
                )
            except Exception as e:
                logger.error("Failed to send lifecycle notification: {}", e)

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""

        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}

            # Special formatting for spawn
            if tc.name == "spawn":
                label = args.get("label") or args.get("task", "")[:30] + (
                    "..." if len(args.get("task", "")) > 30 else ""
                )
                agent = args.get("agent", "default")
                return f'spawn("{label}", agent="{agent}")'

            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'

        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        address: Address,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        *,
        message_id: str | None = None,
        agent_runner: AgentRunner | None = None,
        agent_profile: str | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop.

        *on_stream*: called with each content delta during streaming.
        *on_stream_end(resuming)*: called when a streaming session finishes.
        ``resuming=True`` means tool calls follow (spinner should restart);
        ``resuming=False`` means this is the final response.
        """
        loop_self = self
        session_key = address.to_uri()
        loop_self._set_tool_context(address, session_key, message_id, agent_profile)

        class _LoopHook(AgentHook):
            def __init__(self) -> None:
                self._stream_buf = ""

            def wants_streaming(self) -> bool:
                return on_stream is not None

            async def on_stream(self, context: AgentHookContext, delta: str) -> None:
                from nanobot.utils.helpers import strip_think

                prev_clean = strip_think(self._stream_buf)
                self._stream_buf += delta
                new_clean = strip_think(self._stream_buf)
                incremental = new_clean[len(prev_clean) :]
                if incremental and on_stream:
                    await on_stream(incremental)

            async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
                if on_stream_end:
                    await on_stream_end(resuming=resuming)
                self._stream_buf = ""

            async def before_execute_tools(self, context: AgentHookContext) -> None:
                if on_progress:
                    if not on_stream:
                        thought = loop_self._strip_think(
                            context.response.content if context.response else None
                        )
                        if thought:
                            logger.debug("Calling on_progress for thought")
                            await on_progress(thought)
                    tool_hint = loop_self._strip_think(loop_self._tool_hint(context.tool_calls))
                    logger.debug("Calling on_progress for tool_hint: {}", tool_hint)
                    await on_progress(tool_hint, tool_hint=True)
                for tc in context.tool_calls:
                    args_str = json.dumps(tc.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tc.name, args_str[:200])
                loop_self._set_tool_context(address, session_key, message_id, agent_profile)

            def finalize_content(
                self, context: AgentHookContext, content: str | None
            ) -> str | None:
                return loop_self._strip_think(content)

            async def after_iteration(self, context: AgentHookContext) -> None:
                # Release any subagent spawn gates so they can start
                # after this iteration completes (avoids model swap thrash)
                loop_self.subagents.release_pending_gates()

        # Use user-pinned agent provider if specified in msg metadata, otherwise default to main runner.
        if agent_runner:
            runner = agent_runner
            model = agent_runner.provider.get_default_model()
        else:
            runner = self.runner
            model = self.model

        # Estimate prompt tokens to enable proactive routing in FallbackProvider
        session = self.sessions.get_or_create(session_key)

        # --- Pre-call: Prefix-Anchor calibration ---
        from nanobot.utils.helpers import (
            calculate_base_hash,
            calculate_calibrated_prompt_tokens,
            calculate_history_hash,
        )

        system_prompt = ""
        for m in initial_messages:
            if m.get("role") == "system":
                system_prompt = m.get("content", "")
                break
        else:
            # Fallback: build from context builder if no system message in list
            system_prompt = self.context.build_system_prompt()

        tools_defs = self.tools.get_definitions() or None
        current_base_hash = calculate_base_hash(system_prompt, tools_defs)
        current_history_hash = calculate_history_hash(initial_messages)

        # Previous turn state stored in session metadata
        prev_cal = session.metadata.get("calibration", {})
        prev_actual = prev_cal.get("actual_prompt_tokens", 0)
        prev_base_hash = prev_cal.get("base_hash", "")
        prev_history_hash = prev_cal.get("history_hash", "")

        ratio = self.calibration_mgr.get_ratio(model)
        prompt_tokens = calculate_calibrated_prompt_tokens(
            initial_messages,
            tools_defs,
            ratio,
            current_base_hash=current_base_hash,
            current_history_hash=current_history_hash,
            prev_actual=prev_actual,
            prev_base_hash=prev_base_hash,
            prev_history_hash=prev_history_hash,
        )

        # Conditional audio routing (Phase 2.2):
        # Intercept messages before the LLM call. If the target model supports
        # audio natively, pass through. Otherwise, transcribe audio blocks via Groq.
        routed_messages = await self._route_audio(initial_messages, model, provider=runner.provider)

        result = await runner.run(
            AgentRunSpec(
                initial_messages=routed_messages,
                tools=self.tools,
                model=model,
                max_iterations=self.max_iterations,
                hook=_LoopHook(),
                error_message="Sorry, I encountered an error calling the AI model.",
                concurrent_tools=True,
            ),
            prompt_tokens=prompt_tokens,
        )
        self._last_usage = result.usage

        # --- Post-call: store anchor hashes + update calibration ratio ---
        actual_prompt_tokens = result.usage.get("prompt_tokens", 0)
        session.metadata["calibration"] = {
            "actual_prompt_tokens": actual_prompt_tokens,
            "base_hash": current_base_hash,
            "history_hash": current_history_hash,
        }
        model_used = result.messages[-1].get("_model_used") if result.messages else None
        if not model_used:
            model_used = getattr(runner.provider, "get_default_model", lambda: model)()
        if actual_prompt_tokens > 0 and prompt_tokens > 0:
            self.calibration_mgr.update_ratio(
                model_used, actual=actual_prompt_tokens, estimated=prompt_tokens
            )
        if result.stop_reason == "max_iterations":
            logger.warning("Max iterations ({}) reached", self.max_iterations)
        elif result.stop_reason == "error":
            logger.error("LLM returned error: {}", (result.final_content or "")[:200])
            raise RuntimeError(f"LLM Provider Error: {result.error or result.final_content}")
        return result.final_content, result.tools_used, result.messages

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        # Lifecycle Audit: Log start and handle pending notifications
        try:
            await self._audit_lifecycle()
        except Exception as e:
            logger.error("Lifecycle audit failed: {}", e)

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                # Preserve real task cancellation so shutdown can complete cleanly.
                # Only ignore non-task CancelledError signals that may leak from integrations.
                if not self._running or asyncio.current_task().cancelling():
                    raise
                continue
            except Exception as e:
                logger.warning("Error consuming inbound message: {}, continuing...", e)
                continue

            raw = msg.content.strip()
            if self.commands.is_priority(raw):
                ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, loop=self)
                result = await self.commands.dispatch_priority(ctx)
                if result:
                    await self.bus.publish_outbound(result)
                continue
            task = asyncio.create_task(self._dispatch(msg))
            self._active_tasks.setdefault(msg.session_key, []).append(task)
            task.add_done_callback(
                lambda t, k=msg.session_key: (
                    self._active_tasks.get(k, []) and self._active_tasks[k].remove(t)
                    if t in self._active_tasks.get(k, [])
                    else None
                )
            )

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message: per-session serial, cross-session concurrent."""
        lock = self._session_locks.setdefault(msg.session_key, asyncio.Lock())
        gate = self._concurrency_gate or nullcontext()
        async with lock, gate:
            try:
                on_stream = on_stream_end = None
                if msg.metadata.get("_wants_stream"):
                    # Split one answer into distinct stream segments.
                    stream_base_id = f"{msg.session_key}:{time.time_ns()}"
                    stream_segment = 0

                    def _current_stream_id() -> str:
                        return f"{stream_base_id}:{stream_segment}"

                    async def on_stream(delta: str) -> None:
                        await self.bus.publish_outbound(
                            OutboundMessage(
                                address=msg.address,
                                content=delta,
                                metadata={
                                    "_stream_delta": True,
                                    "_stream_id": _current_stream_id(),
                                },
                            )
                        )

                    async def on_stream_end(*, resuming: bool = False) -> None:
                        nonlocal stream_segment
                        await self.bus.publish_outbound(
                            OutboundMessage(
                                address=msg.address,
                                content="",
                                metadata={
                                    "_stream_end": True,
                                    "_resuming": resuming,
                                    "_stream_id": _current_stream_id(),
                                },
                            )
                        )
                        stream_segment += 1

                response = await self._process_message(
                    msg,
                    on_stream=on_stream,
                    on_stream_end=on_stream_end,
                )
                if response is not None:
                    await self.bus.publish_outbound(response)
                else:
                    # Always signal turn end to channels to clear progress/typing indicators
                    await self.bus.publish_outbound(
                        OutboundMessage(
                            address=msg.address,
                            content="",
                            metadata={**(msg.metadata or {}), "_turn_end": True},
                        )
                    )
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception as e:
                import datetime
                import traceback

                err_trace = traceback.format_exc()
                logger.exception("Error processing message for session {}", msg.session_key)

                log_file = self.workspace / "logs" / "error.log"
                try:
                    log_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(
                            f"--- Error on {datetime.datetime.now().isoformat()} for {msg.session_key} ---\n"
                        )
                        f.write(err_trace + "\n\n")
                except Exception as log_e:
                    logger.error("Failed to write to workspace log: {}", log_e)

                error_msg = f"⚠️ **System Error**\n```\n{type(e).__name__}: {str(e)}\n```\n_See `logs/error.log` in workspace for details._"

                await self.bus.publish_outbound(
                    OutboundMessage(
                        address=msg.address,
                        content=error_msg,
                    )
                )

    async def close_mcp(self) -> None:
        """Drain pending background archives, then close MCP connections."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def _schedule_background(self, coro) -> None:
        """Schedule a coroutine as a tracked background task (drained on shutdown)."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(self._background_tasks.remove)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        self.subagents.release_all_gates()
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Dispatch inbound messages to the appropriate handler."""
        if msg.channel == "system":
            return await self._process_system_message(
                msg, on_progress=on_progress, on_stream=on_stream, on_stream_end=on_stream_end
            )
        return await self._process_user_message(
            msg,
            session_key=session_key,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
        )

    def _make_progress_callback(
        self,
        address: Address,
        base_meta: dict | None = None,
    ) -> Callable[[str, bool], Awaitable[None]]:
        """Build a progress callback that publishes to the bus."""

        async def _callback(text: str, *, tool_hint: bool = False) -> None:
            meta = dict(base_meta or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(
                OutboundMessage(address=address, content=text, metadata=meta)
            )

        return _callback

    async def _process_system_message(
        self,
        msg: InboundMessage,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Handle system messages (restart notifications, subagent replies, etc.)."""
        # msg.chat_id was stored as "channel:chat_id" or a URI
        address = Address.from_uri(msg.chat_id)
        logger.info("Processing system message for {}", address)
        key = address.to_uri()
        session = self.sessions.get_or_create(key)
        await self.memory_consolidator.maybe_consolidate_by_tokens(
            session, self.context_window_tokens
        )

        history = session.get_history(max_messages=0)
        current_role = "assistant" if msg.sender_id == "subagent" else "user"
        messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            channel=address.channel,
            chat_id=address.segments[0] if address.segments else "",
            current_role=current_role,
        )

        # Warn on unexpected agent profile override from metadata
        agent_profile = msg.metadata.get("agent_profile")
        if agent_profile:
            agent_runner, success = self._apply_agent_profile(agent_profile)
            if not success:
                logger.warning("Invalid agent profile {}: falling back to defaults", agent_profile)
            current_runner = agent_runner
        else:
            current_runner = self.runner

        progress_cb = on_progress or self._make_progress_callback(address)

        final_content, _, all_msgs = await self._run_agent_loop(
            messages,
            address=address,
            message_id=msg.metadata.get("message_id"),
            on_progress=progress_cb,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            agent_runner=current_runner,
            agent_profile=agent_profile,
        )

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)
        self._schedule_background(
            self.memory_consolidator.maybe_consolidate_by_tokens(
                session, self.context_window_tokens
            )
        )

        return OutboundMessage(
            address=address,
            content=final_content or "Background task completed.",
        )

    def _apply_agent_profile(self, profile_name: str | None) -> tuple[AgentRunner, bool]:
        """Atomically update the runner and constraints. Returns (runner, success)."""
        target_profile = profile_name or "defaults"
        success = True
        try:
            runner = self.registry.get_runner(target_profile)
        except ValueError:
            logger.warning("Invalid agent profile {}: falling back to defaults", target_profile)
            runner = self.registry.get_runner("defaults")
            target_profile = "defaults"
            success = False

        if self.config.agents and hasattr(self.config.agents, target_profile):
            from nanobot.utils.helpers import resolve_context_window

            profile_cfg = getattr(self.config.agents, target_profile)
            provider_max = getattr(runner.provider.generation, "context_max", None)
            profile_limit = (
                profile_cfg.get("contextWindowTokens")
                if isinstance(profile_cfg, dict)
                else getattr(profile_cfg, "context_window_tokens", None)
            )
            self.context_window_tokens = resolve_context_window(
                profile_limit, provider_max, self.context_window_tokens
            )
            # Update max generation tokens (The "Floor")
            max_tokens = (
                profile_cfg.get("maxTokens")
                if isinstance(profile_cfg, dict)
                else getattr(profile_cfg, "max_tokens", None)
            )
            if max_tokens:
                self.max_tokens = max_tokens

        return runner, success

    async def _process_user_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Handle user messages (chat, slash commands, tool requests, etc.)."""
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        raw = msg.content.strip()
        ctx = CommandContext(msg=msg, session=session, key=key, raw=raw, loop=self)
        if result := await self.commands.dispatch(ctx):
            return result

        # Resolve agent profile override and synchronize constraints (Window/Limit)
        # This MUST happen before consolidation and build_messages so that the correct
        # truncation limit is used for both session pruning and message building.
        agent_profile = msg.metadata.get("agent_profile")
        agent_runner, success = self._apply_agent_profile(agent_profile)

        if not success and agent_profile:
            await self.bus.publish_outbound(
                OutboundMessage(
                    address=msg.address,
                    content=f"⚠️ The pinned profile `{agent_profile}` has issues. Falling back to default agent.",
                    metadata={
                        "system_event": "pin_invalid",
                        "message_id": msg.metadata.get("message_id"),
                    },
                )
            )

        # Consolidate with the correct context window tokens (after profile resolution)
        await self.memory_consolidator.maybe_consolidate_by_tokens(
            session, self.context_window_tokens
        )

        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=0)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )

        progress_cb = on_progress or self._make_progress_callback(msg.address, msg.metadata)

        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages,
            address=msg.address,
            on_progress=progress_cb,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            message_id=msg.metadata.get("message_id"),
            agent_runner=agent_runner,
            agent_profile=agent_profile,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)
        self._schedule_background(
            self.memory_consolidator.maybe_consolidate_by_tokens(
                session, self.context_window_tokens
            )
        )

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        final_content = self._maybe_add_frugality_nudge(session, final_content)

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        meta = dict(msg.metadata or {})
        if on_stream is not None:
            meta["_streamed"] = True
        return OutboundMessage(
            address=msg.address,
            content=final_content,
            metadata=meta,
        )

    @staticmethod
    def _image_placeholder(block: dict[str, Any]) -> dict[str, str]:
        """Convert an inline image block into a compact text placeholder."""
        path = (block.get("_meta") or {}).get("path", "")
        return {"type": "text", "text": f"[image: {path}]" if path else "[image]"}

    @staticmethod
    def _audio_placeholder(block: dict[str, Any]) -> dict[str, str]:
        """Convert an inline audio block into a compact text placeholder."""
        path = (block.get("_meta") or {}).get("path", "")
        mime = block.get("mime_type", "audio/unknown")
        label = f"[audio: {path} ({mime})]" if path else f"[audio ({mime})]"
        return {"type": "text", "text": label}

    def _sanitize_persisted_blocks(
        self,
        content: list[dict[str, Any]],
        *,
        truncate_text: bool = False,
        drop_runtime: bool = False,
    ) -> list[dict[str, Any]]:
        """Strip volatile multimodal payloads before writing session history."""
        filtered: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                filtered.append(block)
                continue

            if (
                drop_runtime
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
            ):
                continue

            if block.get("type") == "image_url" and block.get("image_url", {}).get(
                "url", ""
            ).startswith("data:image/"):
                filtered.append(self._image_placeholder(block))
                continue

            if block.get("type") == "audio":
                filtered.append(self._audio_placeholder(block))
                continue

            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text = block["text"]
                if truncate_text and len(text) > self._TOOL_RESULT_MAX_CHARS:
                    text = text[: self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
                filtered.append({**block, "text": text})
                continue

            filtered.append(block)

        return filtered

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime

        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool":
                if isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                    entry["content"] = content[: self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
                elif isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, truncate_text=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            elif role == "user":
                if isinstance(content, str) and content.startswith(
                    ContextBuilder._RUNTIME_CONTEXT_TAG
                ):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, drop_runtime=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli://direct",
        address: Address | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        # Legacy parameters for backward compatibility
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> OutboundMessage | None:
        """Process a message directly and return the outbound payload."""
        await self._connect_mcp()

        if address is None:
            if channel and chat_id:
                address = Address(channel=channel, segments=(str(chat_id),))
            else:
                address = Address.from_uri(session_key)

        msg = InboundMessage(
            address=address,
            sender_id="user",
            content=content,
        )
        return await self._process_message(
            msg,
            session_key=session_key,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
        )

    def _maybe_add_frugality_nudge(self, session: Session, content: str | None) -> str | None:
        """Append a non-intrusive nudge if the session is getting long/expensive."""
        if not content or not self.config.frugality.enabled:
            return content

        cfg = self.config.frugality
        msg_count = len(session.messages)

        # 1. Check message threshold
        if msg_count < cfg.message_floor:
            # We also check context floor if we're not past message floor yet
            try:
                tokens, _ = self.memory_consolidator.estimate_session_prompt_tokens(session)
                window = self.context_window_tokens
                pct = int(tokens / window * 100) if window else 0
                if pct < cfg.context_floor_pct:
                    return content
            except Exception:
                return content
        else:
            # Past message floor, now check if we should nudge based on interval
            pct = 0  # will calculate if needed below

        # 2. Check if we should nudge this time (modulus)
        last_nudge = session.metadata.get("last_frugality_nudge", 0)
        if msg_count - last_nudge < cfg.nudge_interval:
            return content

        # 3. If we haven't calculated pct yet, do it now for the nudge message
        if pct == 0:
            try:
                tokens, _ = self.memory_consolidator.estimate_session_prompt_tokens(session)
                window = self.context_window_tokens
                pct = int(tokens / window * 100) if window else 0
            except Exception:
                pct = 0

        # Update metadata to record this nudge
        session.metadata["last_frugality_nudge"] = msg_count
        self.sessions.save(session)

        # 4. Append the nudge
        nudge = f"\n\n---\n💡 **Frugality Tip:** This session has {msg_count} messages ({pct}% context). If starting a new topic, consider `/new` to save tokens!"
        return content + nudge
