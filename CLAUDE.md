# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project is using python3 .venv

```bash
# Install in dev mode
.venv/bin/pip install -e ".[dev]"

# Run all tests
.venv/bin/pytest

# Run a single test file
.venv/bin/pytest tests/agent/test_memory_consolidation_types.py

# Lint
.venv/bin/ruff check nanobot/

# Format
.venv/bin/ruff format nanobot/

# Type check (LSP / opencode)
.venv/bin/basedpyright
```

## Architecture

**nanobot** is an ultra-lightweight AI agent framework (~2,000 lines of core agent code). It receives messages from chat platforms, processes them through an LLM with tool execution, and streams responses back.

### Message Flow

```
Chat Channel → MessageBus → AgentLoop → AgentRegistry → LLMClient → ToolRegistry → MessageBus → Chat Channel
```

Messages (`InboundMessage`, `OutboundMessage`) use the `Address` interface (channel name + path segments). `AgentLoop` (`nanobot/agent/loop.py`) is the core orchestrator, which now leverages an optional `AgentRegistry`. It uses the registry to pull specialized Runners/Providers, or uses an explicitly supplied one. Message dispatch is split into `_process_system_message()` and `_process_user_message()` with a shared `_make_progress_callback()` factory for bus publishing. After each turn, `MemoryConsolidator` stages deltas to `STAGING.md` (non-destructive pipeline).

### Key Packages

| Package | Role |
|---------|------|
| `nanobot/agent/` | Core agent: loop, context building, memory, tools, subagents |
| `nanobot/providers/` | LLM backends (Anthropic, OpenAI-compat, Azure, OAuth) |
| `nanobot/channels/` | Chat platform integrations (12 platforms) |
| `nanobot/bus/` | Async message queue between channels and agent |
| `nanobot/session/` | Per-conversation history (JSONL append-only) |
| `nanobot/config/` | Pydantic config schema + workspace path resolution |
| `nanobot/cli/` | Typer CLI (`agent`, `gateway`, `onboard`, `status`, etc.) |
| `nanobot/skills/` | Bundled skills. Other skills are loaded from `~/.nanobot/workspace/skills/` |

### Provider System

`nanobot/providers/registry.py` is the single source of truth for 25+ providers. Each `ProviderSpec` defines name, keywords, `env_key`, default API base, and capability flags. Additional fields on `ProviderSpec`:
- `is_local: bool` — disables timeout retries (local providers should fail fast, not retry)
- `http_timeout: int` — per-provider HTTP timeout in seconds (default 120; local providers use 600)
- `max_concurrent: int` — max simultaneous requests (0 = unlimited; local providers use 1 to serialize)
- `strip_model_prefix: bool` — strip `"provider/"` routing prefix before sending model name to API

**To add a new provider**: (1) add a `ProviderSpec` to `PROVIDERS` in `registry.py`, (2) add a config field to `ProvidersConfig` in `nanobot/config/schema.py`.

Backend implementations include: `AnthropicClient`, `OpenAICompatClient`, `AzureOpenAIClient`, and `GeminiNativeClient`. A special `FallbackClient` wraps these, providing quota-awareness and automatic fallback chains that reset daily.

**FallbackClient failure modes:**
- **Quota exhaustion** → permanent slot advance (`self._active_index`); next reset scheduled per `quota_reset_timezone`
- **Connectivity/timeout on local slot** → request-scoped fallback only (`effective_index`); next request retries the local provider first
- **Context Overflow (Proactive)** → If `prompt_tokens` is provided, slots whose `context_max` is exceeded are proactively skipped in favor of larger-context fallbacks.

**Provider resolution** (`nanobot/config/schema.py`): `_match_provider` and all public resolution methods (`get_provider`, `get_api_base`, etc.) accept an explicit `provider_override` parameter. When `provider_override == "auto"`, resolution first checks `agent_config.provider` before falling back to keyword-based model matching. Pass `mc.provider` when resolving per-slot settings to avoid agent-level defaults leaking into per-model lookups. Note: `llama`/`llama.cpp` are NOT in inferencia's keywords (removed to prevent false-positive detection on `llama3.2`, etc.).

### Tool System

`nanobot/agent/tools/registry.py` holds `ToolRegistry`. Built-in tools: file operations, shell exec, web search/fetch, message sending, subagent spawn, cron scheduling, and MCP. File/shell tools respect `restrictToWorkspace` config to sandbox access.

**ExecTool safety guard** (`nanobot/agent/tools/shell.py`):
- Blocks dangerous patterns (rm -rf, dd, format, shutdown, fork bombs).
- Detects internal/private URLs via DNS resolution before execution.
- When `restrictToWorkspace` is enabled, extracts absolute paths from commands and blocks paths outside the working directory.
- **Slash command whitelist:** Known slash commands (`/new`, `/restart`, etc.) are filtered from path extraction to prevent false-positive workspace guard blocks (e.g., `echo "/new"` must not be mistaken for accessing `/new` filesystem path).

### Subagent System

The `SubagentManager` (`nanobot/agent/subagent.py`) handles background task execution. Subagents are spawned via the `spawn` tool and run in a separate `AgentRunner` instance. The `spawn` tool dynamically documents available agent profiles (e.g., `fast`, `deep`, `balanced`), enabling the main agent to autonomously select the appropriate model. Subagents inherit the workspace context (`SOUL.md`, `USER.md`, `SUBAGENT.md`) and report back to the main agent through the `MessageBus` upon completion. Subagents propagate the active agent profile through their entire lifecycle, ensuring that results are evaluated using the profile active at the time of spawn, regardless of subsequent topic pin changes.


### Memory System

Three-layer staging architecture:
- **`STAGING.md`** — Transient scratchpad. `MemoryConsolidator` appends deltas here after each turn (non-destructive).
- **`MEMORY.md`** — Long-term invariants. Updated via manual synthesis (`memory-synthesize` skill) which reads from staging.
- **`HISTORY.md`** — Append-only timestamped log. Searchable via grep-style tools.

`MemoryConsolidator` appends incremental updates to `STAGING.md` after each turn (never overwrites `MEMORY.md` directly). It implements a Write-Ahead Log (WAL) to checkpoint messages before the LLM call, preventing data loss during interruptions (e.g., `/restart`, `asyncio.CancelledError`). Strict truncation protection (`finish_reason == "length"`) prevents corrupting STAGING.md/HISTORY.md. Periodic `memory-synthesize` skill reads from staging to update `MEMORY.md`. It uses the loop's own provider (full fallback chain) and the provider's configured `max_tokens`. Messages are never modified after writing (cache-friendly for prompt caching).

### Skill System

`SkillsLoader` scans `~/.nanobot/workspace/skills/` for directories containing `SKILL.md`. Some skills are `always_skills` (always loaded into context); others are on-demand. Bundled skills live in `nanobot/skills/`.

### TTS System

Integrated Text-to-Speech capabilities via `AgentLoop.get_tts_client()`. Supported providers include `Chatterbox` (Local Nano/Turbo) and `OpenAI` (tts-1). Configuration is managed in `config.json` under the `tts` key. The `/speak` command triggers synthesis of specified text or the most recent assistant response.

Built-in slash commands are registered in `nanobot/command/builtin.py`. The `/status` command is a consolidated dashboard providing system uptime, token usage, context estimation, and active background tasks. Commands like `/new`, `/compact`, `/repl`, `/restart`, and `/speak` manage session lifecycle, audio synthesis, and development operations. The `/repl` command allows executing arbitrary Python code for debugging and system management. For security, it is restricted to users listed in `repl.allowUsers` (default: `["cli:user"]`). Access can be granted to other users using the format `"channel:sender_id"` (e.g., `"tg:12345678"`).

### Channel Integration (Telegram)

The Telegram channel implementation supports advanced interaction modes:
- **Topics as Sessions:** Threads/Topics represent isolated `Address` contexts (`tg://chat_id/topic_id`).
- **Profile Pinning:** Frugal profile switching is natively supported via pinned messages containing `Profile: <profile_name>`.
- **Audit Trails:** Progress/Status is streamed into a live message with proactive rollover (creating new messages gracefully before hitting Telegram's 4096-character limit).

## Configuration

User config normally lives at `~/.nanobot/config.json` but when run as service via systemd, it's mapped from `/etc/nanobot/config.json`
The workspace (agent files, memory, skills) defaults to `~/.nanobot/workspace/`.

## Testing Conventions

- All async tests use `pytest-asyncio` with `asyncio_mode = "auto"` (set in `pyproject.toml`)
- Tests live in `tests/` mirroring the source structure
- Integration tests hit real code paths; avoid mocking internals

## Deployment Architecture

Production environments use a symlink-based worktree architecture:
- **Workspace:** `~/.nanobot/workspace`
- **Active Code:** Symlink `~/.nanobot/workspace/current` points to a specific Git worktree (e.g., `nanobot-dev`, `nanobot-main`).
- **Service:** `nanobot.service` runs the gateway using the virtual environment inside the `current` symlink.
- **Environment Overrides:** Managed via `~/.nanobot/workspace/nanobot.env` (e.g., `EXTRA_ARGS` for config overlays).
- **Lifecycle Logging:** Restarts/shutdowns log intents to `~/.nanobot/workspace/logs/lifecycle.log`. The bot processes read this log on boot to report success to the user.

## Branching

`dev` is the working branch in the `workspace/nanobot-dev` worktree -- agents should perform all work in the worktree.
Actual repository root is at `workspace/nanobot-repo` -- user handles merging from `dev` to `staging` and/or `main`.
- remote `origin` links to user's own repository.
