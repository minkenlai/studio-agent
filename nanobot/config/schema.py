"""Configuration schema using Pydantic."""

from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from pydantic_settings import BaseSettings

from nanobot.providers.base import ModelCapabilities


class Base(BaseModel):
    """Base model that accepts both camelCase and snake_case keys."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ChannelsConfig(Base):
    """Configuration for chat channels.

    Built-in and plugin channel configs are stored as extra fields (dicts).
    Each channel parses its own config in __init__.
    Per-channel "streaming": true enables streaming output (requires send_delta impl).
    """

    model_config = ConfigDict(extra="allow")

    send_progress: bool = True  # stream agent's text progress to the channel
    send_tool_hints: bool = False  # stream tool-call hints (e.g. read_file("…"))
    send_max_retries: int = Field(
        default=3, ge=0, le=10
    )  # Max delivery attempts (initial send included)


class ModelConfig(Base):
    """Per-model configuration entry used in the top-level ``models`` dict."""

    model: str
    provider: str = "auto"
    temperature: float | None = None
    max_tokens: int | None = None
    context_max: int | None = None
    reasoning_effort: str | None = None
    prefill: bool | None = None


class AgentDefaults(Base):
    """Default agent configuration."""

    # The "defaults" workspace is the sandbox ROOT for the whole process.
    # For named profiles, workspace is the subagent *working directory*
    # (Option B): relative values resolve against the root, and the value is
    # rejected at spawn if it resolves outside the root — see
    # SubagentManager.resolve_profile_workspace. The containment boundary
    # itself never moves.
    workspace: str = "~/.nanobot/workspace"
    model: str = "anthropic/claude-opus-4-5"
    provider: str = (
        "auto"  # Provider name (e.g. "anthropic", "openrouter") or "auto" for auto-detection
    )
    max_tokens: int = 8192
    context_window_tokens: int = 65_536
    temperature: float = 1.0
    max_tool_iterations: int = 40
    reasoning_effort: str | None = None  # low / medium / high - enables LLM thinking mode
    grounding: str | None = None  # e.g. "google_search"
    timezone: str = "UTC"  # IANA timezone, e.g. "Asia/Shanghai", "America/New_York"
    description: str | None = None  # Optional description to help choose the agent

    fallback_models: list[str] = Field(
        default_factory=list
    )  # ordered keys into top-level models dict


class AgentsConfig(Base):
    """Agent configuration."""

    defaults: AgentDefaults = Field(default_factory=AgentDefaults)

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="allow")

    def get_agent(self, name: str) -> AgentDefaults:
        """Get an agent configuration by name. Extra configs are lazily parsed."""
        if name == "defaults":
            return self.defaults
        data = getattr(self, name, None)
        if data is None:
            raise ValueError(f"Agent '{name}' is not configured.")
        if isinstance(data, dict):
            # Lazily parse the extra dictionary into AgentDefaults
            parsed = AgentDefaults.model_validate(data)
            setattr(self, name, parsed)
            return parsed
        if isinstance(data, AgentDefaults):
            return data
        raise ValueError(f"Agent '{name}' configuration is invalid type: {type(data)}")


class ReplConfig(Base):
    """Configuration for /repl command."""

    allow_users: list[str] = Field(
        default_factory=lambda: ["cli:user"]
    )  # ["tg:user_id", "cli:user"]


class ProviderConfig(Base):
    """Configuration for an LLM provider endpoint."""

    api_key: str = ""
    quota_reset_timezone: str = "UTC"
    api_base: str | None = None
    extra_headers: dict[str, str] | None = None  # Custom headers (e.g. APP-Code for AiHubMix)


class ChatterboxTTSConfig(BaseModel):
    model_variant: str = "nano"  # "nano" or "turbo"
    device: str = "cpu"  # "cpu" or "cuda"
    audio_prompt_path: str | None = None  # optional voice cloning ref


class OpenAITTSConfig(BaseModel):
    api_key: str | None = None
    api_base: str | None = None
    default_voice: str = "alloy"


class TTSConfig(BaseModel):
    provider: str = "chatterbox"  # "chatterbox" or "openai"
    chatterbox: ChatterboxTTSConfig = Field(default_factory=ChatterboxTTSConfig)
    openai: OpenAITTSConfig = Field(default_factory=OpenAITTSConfig)


class ProvidersConfig(Base):
    """Configuration for LLM providers."""

    inferencia: ProviderConfig = Field(default_factory=ProviderConfig)  # Self-hosted llama.cpp
    custom: ProviderConfig = Field(default_factory=ProviderConfig)  # Any OpenAI-compatible endpoint
    azure_openai: ProviderConfig = Field(
        default_factory=ProviderConfig
    )  # Azure OpenAI (model = deployment name)
    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)
    groq: ProviderConfig = Field(default_factory=ProviderConfig)
    xai: ProviderConfig = Field(default_factory=ProviderConfig)
    zhipu: ProviderConfig = Field(default_factory=ProviderConfig)
    dashscope: ProviderConfig = Field(default_factory=ProviderConfig)
    vllm: ProviderConfig = Field(default_factory=ProviderConfig)
    ollama: ProviderConfig = Field(default_factory=ProviderConfig)  # Ollama local models
    ovms: ProviderConfig = Field(default_factory=ProviderConfig)  # OpenVINO Model Server (OVMS)
    gemini: ProviderConfig = Field(default_factory=ProviderConfig)
    moonshot: ProviderConfig = Field(default_factory=ProviderConfig)
    minimax: ProviderConfig = Field(default_factory=ProviderConfig)
    mistral: ProviderConfig = Field(default_factory=ProviderConfig)
    stepfun: ProviderConfig = Field(default_factory=ProviderConfig)  # Step Fun (阶跃星辰)
    aihubmix: ProviderConfig = Field(default_factory=ProviderConfig)  # AiHubMix API gateway
    siliconflow: ProviderConfig = Field(default_factory=ProviderConfig)  # SiliconFlow (硅基流动)
    volcengine: ProviderConfig = Field(default_factory=ProviderConfig)  # VolcEngine (火山引擎)
    volcengine_coding_plan: ProviderConfig = Field(
        default_factory=ProviderConfig
    )  # VolcEngine Coding Plan
    byteplus: ProviderConfig = Field(
        default_factory=ProviderConfig
    )  # BytePlus (VolcEngine international)
    byteplus_coding_plan: ProviderConfig = Field(
        default_factory=ProviderConfig
    )  # BytePlus Coding Plan
    openai_codex: ProviderConfig = Field(
        default_factory=ProviderConfig, exclude=True
    )  # OpenAI Codex (OAuth)
    github_copilot: ProviderConfig = Field(
        default_factory=ProviderConfig, exclude=True
    )  # Github Copilot (OAuth)


class HeartbeatConfig(Base):
    """Heartbeat service configuration."""

    enabled: bool = True
    interval_s: int = 30 * 60  # 30 minutes
    keep_recent_messages: int = 8


class FrugalityConfig(Base):
    """Configuration for frugality nudges."""

    enabled: bool = True
    message_floor: int = 50
    context_floor_pct: int = 60
    nudge_interval: int = 10


class GatewayConfig(Base):
    """Gateway/server configuration."""

    host: str = "0.0.0.0"
    port: int = 18790
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)


class WebSearchConfig(Base):
    """Web search tool configuration."""

    provider: str = "brave"  # brave, tavily, duckduckgo, searxng, jina
    api_key: str = ""
    base_url: str = ""  # SearXNG base URL
    max_results: int = 5


class WebToolsConfig(Base):
    """Web tools configuration."""

    proxy: str | None = (
        None  # HTTP/SOCKS5 proxy URL, e.g. "http://127.0.0.1:7890" or "socks5://127.0.0.1:1080"
    )
    search: WebSearchConfig = Field(default_factory=WebSearchConfig)


class ExecToolConfig(Base):
    """Shell exec tool configuration."""

    enable: bool = True
    timeout: int = 60
    path_append: str = ""


class MCPServerConfig(Base):
    """MCP server connection configuration (stdio or HTTP)."""

    type: Literal["stdio", "sse", "streamableHttp"] | None = None  # auto-detected if omitted
    command: str = ""  # Stdio: command to run (e.g. "npx")
    args: list[str] = Field(default_factory=list)  # Stdio: command arguments
    env: dict[str, str] = Field(default_factory=dict)  # Stdio: extra env vars
    url: str = ""  # HTTP/SSE: endpoint URL
    headers: dict[str, str] = Field(default_factory=dict)  # HTTP/SSE: custom headers
    tool_timeout: int = 30  # seconds before a tool call is cancelled
    enabled_tools: list[str] = Field(
        default_factory=lambda: ["*"]
    )  # Only register these tools; accepts raw MCP names or wrapped mcp_<server>_<tool> names; ["*"] = all tools; [] = no tools


class ToolsConfig(Base):
    """Tools configuration."""

    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False  # If true, restrict all tool access to workspace directory
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


def get_default_capabilities() -> dict[str, ModelCapabilities]:
    """Return the built-in capability map for well-known models.

    Users can override or extend this map via ``config.json`` by setting
    the top-level ``capabilities`` field.
    """
    return {
        # Gemma 4 Family
        "gemma4-e4b": ModelCapabilities(audio=True, vision=True),
        "gemma4-e4b-udq8": ModelCapabilities(audio=True, vision=True),
        "gemma4-e4b-bf16": ModelCapabilities(audio=True, vision=True),
        "gemma4-12b": ModelCapabilities(audio=True, vision=True),
        "gemma4-31b": ModelCapabilities(audio=False, vision=True),
        "gemma4-26b": ModelCapabilities(audio=False, vision=True),
        # Qwen Family
        "qwen36-27b-mm": ModelCapabilities(audio=False, vision=True),
        "qwen36-35b-udiq4": ModelCapabilities(audio=False, vision=True),
        "qwen35-9b": ModelCapabilities(audio=False, vision=True),
        "qwen35-9b-coder": ModelCapabilities(audio=False, vision=True),
        # Cloud / General
        "gpt-4o": ModelCapabilities(audio=True, vision=True),
        "gpt-4o-mini": ModelCapabilities(audio=True, vision=True),
        "claude-3-5-sonnet": ModelCapabilities(vision=True),
        "deepseek-r1": ModelCapabilities(reasoning=True),
    }


class OverlayPolicyConfig(Base):
    """Policy governing which parts of the config a config overlay may change.

    Only meaningful when a config overlay is active (``--config-overlay`` /
    ``NANOBOT_CONFIG_OVERLAY``). The policy itself lives in the *base* config
    (which is operator-owned and unreachable from the agent), so an overlay —
    which may be agent-writable — can never widen its own authority.

    - ``allowedSections``: top-level sections the overlay may modify (deep-merged).
      Any other top-level section is rejected.
    - ``deniedPaths``: dotted paths that must remain unchanged after merging
      (e.g. ``agents.defaults.workspace`` even when ``agents`` is allowed).
    - ``onViolation``: ``warn`` (reject the offending parts, boot continues —
      default) or ``fail`` (refuse to boot).
    """

    allowed_sections: list[str] = Field(default_factory=list)
    denied_paths: list[str] = Field(default_factory=list)
    # "warn" (default) or "fail"; anything else is treated as "warn"
    # (fail-soft: a policy typo must not brick the whole config load).
    on_violation: str = "warn"


class Config(BaseSettings):
    """Root configuration for nanobot."""

    models: dict[str, ModelConfig] = Field(default_factory=dict)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    repl: ReplConfig = Field(default_factory=ReplConfig)
    frugality: FrugalityConfig = Field(default_factory=FrugalityConfig)
    # Accept both spellings: the config template is serialized camelCase
    # (to_camel), but this root is a BaseSettings that does NOT inherit the
    # alias_generator from Base, so without an explicit alias the camelCase
    # ``overlayPolicy`` key from disk is rejected as an unknown field
    # (extra_forbidden) — which silently boots with the DEFAULT config.
    overlay_policy: OverlayPolicyConfig = Field(
        default_factory=OverlayPolicyConfig,
        validation_alias=AliasChoices("overlayPolicy", "overlay_policy"),
    )
    debug_llm: bool = False  # Log full responses and dump them to files in workspace/logs/llm_dumps
    capabilities: dict[str, ModelCapabilities] = Field(
        default_factory=get_default_capabilities,
        description="Mapping of canonical model IDs to their native capabilities.",
    )

    @property
    def workspace_path(self) -> Path:
        """Get expanded workspace path."""
        return Path(self.agents.defaults.workspace).expanduser()

    def _match_provider(
        self,
        model: str | None = None,
        agent_name: str = "defaults",
        provider_override: str = "auto",
    ) -> tuple["ProviderConfig | None", str | None]:
        """Match provider config and its registry name. Returns (config, spec_name).

        *provider_override* takes precedence over the agent's configured ``provider``
        field.  Pass it explicitly (e.g. from a ModelConfig entry) to avoid mutating
        shared agent state.
        """
        from nanobot.providers.registry import PROVIDERS, find_by_name

        agent_config = self.agents.get_agent(agent_name)
        effective_provider = (
            provider_override if provider_override != "auto" else agent_config.provider
        )
        if effective_provider != "auto":
            spec = find_by_name(effective_provider)
            if spec:
                p = getattr(self.providers, spec.name, None)
                return (p, spec.name) if p else (None, None)
            else:
                logger.warning(f"provider not found {effective_provider}")
            return None, None

        model_lower = (model or agent_config.model).lower()
        model_normalized = model_lower.replace("-", "_")
        model_prefix = model_lower.split("/", 1)[0] if "/" in model_lower else ""
        normalized_prefix = model_prefix.replace("-", "_")

        def _kw_matches(kw: str) -> bool:
            kw = kw.lower()
            return kw in model_lower or kw.replace("-", "_") in model_normalized

        # Explicit provider prefix wins — prevents `github-copilot/...codex` matching openai_codex.
        for spec in PROVIDERS:
            p = getattr(self.providers, spec.name, None)
            if p and model_prefix and normalized_prefix == spec.name:
                if spec.is_oauth or spec.is_local or p.api_key:
                    return p, spec.name

        # Match by keyword (order follows PROVIDERS registry)
        for spec in PROVIDERS:
            p = getattr(self.providers, spec.name, None)
            if p and any(_kw_matches(kw) for kw in spec.keywords):
                if spec.is_oauth or spec.is_local or p.api_key:
                    return p, spec.name

        # Fallback: configured local providers can route models without
        # provider-specific keywords (for example plain "llama3.2" on Ollama).
        # Prefer providers whose detect_by_base_keyword matches the configured api_base
        # (e.g. Ollama's "11434" in "http://localhost:11434") over plain registry order.
        local_fallback: tuple[ProviderConfig, str] | None = None
        for spec in PROVIDERS:
            if not spec.is_local:
                continue
            p = getattr(self.providers, spec.name, None)
            if not (p and p.api_base):
                continue
            if spec.detect_by_base_keyword and spec.detect_by_base_keyword in p.api_base:
                return p, spec.name
            if local_fallback is None:
                local_fallback = (p, spec.name)
        if local_fallback:
            return local_fallback

        # Fallback: gateways and direct providers only (follows registry order).
        # Standard cloud providers (Gemini, Anthropic, etc.) must match by keyword above —
        # they cannot serve arbitrary model names, so falling back to them silently would
        # route requests to the wrong API (e.g. Gemini when Ollama was intended).
        # OAuth providers are NOT valid fallbacks — they require explicit model selection.
        for spec in PROVIDERS:
            if spec.is_oauth or not (spec.is_gateway or spec.is_direct):
                continue
            p = getattr(self.providers, spec.name, None)
            if p and p.api_key:
                return p, spec.name
        return None, None

    def get_provider(
        self,
        model: str | None = None,
        agent_name: str = "defaults",
        provider_override: str = "auto",
    ) -> ProviderConfig | None:
        """Get matched provider config (api_key, api_base, extra_headers)."""
        p, _ = self._match_provider(model, agent_name, provider_override)
        return p

    def get_provider_name(
        self,
        model: str | None = None,
        agent_name: str = "defaults",
        provider_override: str = "auto",
    ) -> str | None:
        """Get the registry name of the matched provider (e.g. "deepseek", "openrouter")."""
        _, name = self._match_provider(model, agent_name, provider_override)
        return name

    def get_api_key(
        self,
        model: str | None = None,
        agent_name: str = "defaults",
        provider_override: str = "auto",
    ) -> str | None:
        """Get API key for the given model."""
        p = self.get_provider(model, agent_name, provider_override)
        return p.api_key if p else None

    def get_api_base(
        self,
        model: str | None = None,
        agent_name: str = "defaults",
        provider_override: str = "auto",
    ) -> str | None:
        """Get API base URL for the given model. Applies default URLs for gateway/local providers."""
        from nanobot.providers.registry import find_by_name

        p, name = self._match_provider(model, agent_name, provider_override)
        if p and p.api_base:
            return p.api_base
        # Only gateways get a default api_base here. Standard providers
        # resolve their base URL from the registry in the provider constructor.
        if name:
            spec = find_by_name(name)
            if spec and (spec.is_gateway or spec.is_local) and spec.default_api_base:
                return spec.default_api_base
        return None

    model_config = {"env_prefix": "NANOBOT_", "env_nested_delimiter": "__"}
