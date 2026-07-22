"""Base LLM provider interface."""

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel


@dataclass
class ToolCallRequest:
    """A tool call request from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]
    extra_content: dict[str, Any] | None = None
    provider_specific_fields: dict[str, Any] | None = None
    function_provider_specific_fields: dict[str, Any] | None = None

    def to_openai_tool_call(self) -> dict[str, Any]:
        """Serialize to an OpenAI-style tool_call payload."""
        tool_call = {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }
        if self.extra_content:
            tool_call["extra_content"] = self.extra_content
        if self.provider_specific_fields:
            tool_call["provider_specific_fields"] = self.provider_specific_fields
        if self.function_provider_specific_fields:
            tool_call["function"]["provider_specific_fields"] = (
                self.function_provider_specific_fields
            )
        return tool_call


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    reasoning_content: str | None = None  # Kimi, DeepSeek-R1 etc.
    thinking_blocks: list[dict] | None = None  # Anthropic extended thinking
    model_used: str | None = None  # The actual model ID returned by the provider

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


class ModelCapabilities(BaseModel):
    """Capabilities a model natively supports."""

    audio: bool = False
    vision: bool = False
    reasoning: bool = False

    model_config = {"frozen": True}


@dataclass(frozen=True)
class GenerationSettings:
    """Default generation parameters for LLM calls.

    Stored on the provider so every call site inherits the same defaults
    without having to pass temperature / max_tokens / reasoning_effort
    through every layer.  Individual call sites can still override by
    passing explicit keyword arguments to chat() / chat_with_retry().
    """

    temperature: float = 1.0
    max_tokens: int = 4096
    reasoning_effort: str | None = None
    context_max: int | None = None


class LLMClient(ABC):
    """
    Abstract base class for LLM providers.

    Implementations should handle the specifics of each provider's API
    while maintaining a consistent interface.
    """

    _CHAT_RETRY_DELAYS = (1, 2, 4)
    _TRANSIENT_ERROR_MARKERS = (
        "429",
        "rate limit",
        "500",
        "502",
        "503",
        "504",
        "overloaded",
        "timeout",
        "timed out",
        "connection",
        "server error",
        "temporarily unavailable",
    )

    _SENTINEL = object()

    def __init__(self, api_key: str | None = None, api_base: str | None = None):
        self.api_key = api_key
        self.api_base = api_base
        self.generation: GenerationSettings = GenerationSettings()
        self.debug: bool = False
        self.dump_dir: Path | None = None

    def _dump_debug_data(self, name: str, data: Any) -> None:
        """Dump raw LLM data to a file in workspace/logs/llm_dumps/."""
        if not self.debug or not self.dump_dir:
            return

        try:
            target_dir = self.dump_dir / "llm_dumps"
            target_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}_{name}.json"
            target_path = target_dir / filename

            # Handle non-serializable objects (like OpenAI SDK models)
            def _default(obj: Any) -> Any:
                if hasattr(obj, "model_dump"):
                    return obj.model_dump()
                return str(obj)

            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=_default)

            logger.debug("LLM debug data dumped to {}", target_path)
        except Exception as e:
            logger.warning("Failed to dump LLM debug data: {}", e)

    @property
    def supports_native_audio(self):
        spec = getattr(self, "_spec", None)
        return spec and getattr(spec, "supports_native_audio", False)

    @staticmethod
    def _sanitize_empty_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sanitize message content: fix empty blocks, strip internal _meta fields."""
        result: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")

            if isinstance(content, str) and not content:
                clean = dict(msg)
                clean["content"] = (
                    None
                    if (msg.get("role") == "assistant" and msg.get("tool_calls"))
                    else "(empty)"
                )
                result.append(clean)
                continue

            if isinstance(content, list):
                new_items: list[Any] = []
                changed = False
                for item in content:
                    if (
                        isinstance(item, dict)
                        and item.get("type") in ("text", "input_text", "output_text")
                        and not item.get("text")
                    ):
                        changed = True
                        continue
                    if isinstance(item, dict) and "_meta" in item:
                        new_items.append({k: v for k, v in item.items() if k != "_meta"})
                        changed = True
                    else:
                        new_items.append(item)
                if changed:
                    clean = dict(msg)
                    if new_items:
                        clean["content"] = new_items
                    elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                        clean["content"] = None
                    else:
                        clean["content"] = "(empty)"
                    result.append(clean)
                    continue

            if isinstance(content, dict):
                clean = dict(msg)
                clean["content"] = [content]
                result.append(clean)
                continue

            result.append(msg)
        return result

    @staticmethod
    def _sanitize_request_messages(
        messages: list[dict[str, Any]],
        allowed_keys: frozenset[str],
    ) -> list[dict[str, Any]]:
        """Keep only provider-safe message keys and normalize assistant content."""
        sanitized = []
        for msg in messages:
            clean = {k: v for k, v in msg.items() if k in allowed_keys}
            if clean.get("role") == "assistant" and "content" not in clean:
                clean["content"] = None
            sanitized.append(clean)
        return sanitized

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        prompt_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions.
            model: Model identifier (provider-specific).
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            tool_choice: Tool selection strategy ("auto", "required", or specific tool dict).
            prompt_tokens: Estimated prompt token count (used by FallbackClient for routing).

        Returns:
            LLMResponse with content and/or tool calls.
        """
        pass

    def _is_transient_error(self, content: str | None) -> bool:
        err = (content or "").lower()
        markers = self._TRANSIENT_ERROR_MARKERS
        # Local providers (Ollama, vLLM) are inherently slow — a timeout means
        # the model is still processing, not that the endpoint is flaky.
        # Retrying would pile up a second request on top of the first, making
        # things exponentially worse. Skip the timeout markers for local specs.
        spec = getattr(self, "_spec", None)
        if spec and getattr(spec, "is_local", False):
            markers = tuple(m for m in markers if m not in ("timeout", "timed out"))
        return any(marker in err for marker in markers)

    @staticmethod
    def _strip_image_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """Replace image_url blocks with text placeholder. Returns None if no images found."""
        found = False
        result = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                new_content = []
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "image_url":
                        path = (b.get("_meta") or {}).get("path", "")
                        placeholder = f"[image: {path}]" if path else "[image omitted]"
                        new_content.append({"type": "text", "text": placeholder})
                        found = True
                    else:
                        new_content.append(b)
                result.append({**msg, "content": new_content})
            else:
                result.append(msg)
        return result if found else None

    @staticmethod
    def _strip_audio_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """Replace audio blocks with text placeholder. Returns None if no audio found.

        Mirrors ``_strip_image_content`` — used when a non-transient error
        occurs and the target model may not support audio payloads.
        """
        found = False
        result = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                new_content = []
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "audio":
                        path = (b.get("_meta") or {}).get("path", "")
                        mime = b.get("mime_type", "audio/unknown")
                        placeholder = (
                            f"[audio: {path} ({mime})]" if path else f"[audio omitted ({mime})]"
                        )
                        new_content.append({"type": "text", "text": placeholder})
                        found = True
                    else:
                        new_content.append(b)
                result.append({**msg, "content": new_content})
            else:
                result.append(msg)
        return result if found else None

    async def _safe_chat(self, **kwargs: Any) -> LLMResponse:
        """Call chat() and convert unexpected exceptions to error responses."""
        try:
            # ENFORCING GLOBAL APPLICATION TIMEOUT: 10 minutes (600s)
            # This forces a clean exit path if the underlying provider locks up.
            return await asyncio.wait_for(self.chat(**kwargs), timeout=600.0)
        except asyncio.TimeoutError:
            print("Provider call timed out after 600 seconds.")
            return LLMResponse(content="Error: LLM generation timed out.", finish_reason="timeout")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Catch other provider-specific exceptions
            return LLMResponse(content=f"Error calling chat: {exc}", finish_reason="error")

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
        prompt_tokens: int | None = None,
    ) -> LLMResponse:
        """Stream a chat completion, calling *on_content_delta* for each text chunk.

        Returns the same ``LLMResponse`` as :meth:`chat`.  The default
        implementation falls back to a non-streaming call and delivers the
        full content as a single delta.  Providers that support native
        streaming should override this method.
        """
        response = await self.chat(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
            prompt_tokens=prompt_tokens,
        )
        if on_content_delta and response.content:
            await on_content_delta(response.content)
        return response

    async def _safe_chat_stream(self, **kwargs: Any) -> LLMResponse:
        """Call chat_stream() and convert unexpected exceptions to error responses."""
        try:
            return await asyncio.wait_for(self.chat_stream(**kwargs), timeout=600.0)
        except asyncio.TimeoutError:
            print("Provider call timed out after 600 seconds.")
            return LLMResponse(content="Error: LLM generation timed out.", finish_reason="timeout")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            msg = str(exc) or type(exc).__name__
            return LLMResponse(content=f"Error calling LLM: {msg}", finish_reason="error")

    async def chat_stream_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = _SENTINEL,
        temperature: object = _SENTINEL,
        reasoning_effort: object = _SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
        prompt_tokens: int | None = None,
    ) -> LLMResponse:
        """Call chat_stream() with retry on transient provider failures.

        Call chain: chat_stream_with_retry() → _safe_chat_stream() → chat_stream()
        The safe wrapper adds timeout and exception-to-response conversion.
        Subclasses override chat_stream() (e.g. FallbackClient for context-exceeded fallback).
        This method handles the retry loop for transient errors (rate limits, etc.).
        """
        if max_tokens is self._SENTINEL:
            max_tokens = self.generation.max_tokens
        if temperature is self._SENTINEL:
            temperature = self.generation.temperature
        if reasoning_effort is self._SENTINEL:
            reasoning_effort = self.generation.reasoning_effort

        kw: dict[str, Any] = dict(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
            on_content_delta=on_content_delta,
            prompt_tokens=prompt_tokens,
        )

        for attempt, delay in enumerate(self._CHAT_RETRY_DELAYS, start=1):
            response = await self._safe_chat_stream(**kw)

            if response.finish_reason != "error":
                return response

            if not self._is_transient_error(response.content):
                stripped = self._strip_image_content(messages)
                if stripped is not None:
                    logger.warning(
                        "Non-transient LLM error with image content, retrying without images: {}",
                        (response.content or "")[:120],
                    )
                    return await self._safe_chat_stream(**{**kw, "messages": stripped})
                stripped_audio = self._strip_audio_content(messages)
                if stripped_audio is not None:
                    logger.warning(
                        "Non-transient LLM error with audio content, retrying without audio: {}",
                        (response.content or "")[:120],
                    )
                    return await self._safe_chat_stream(**{**kw, "messages": stripped_audio})
                return response

            logger.warning(
                "LLM transient error (attempt {}/{}), retrying in {}s: {}",
                attempt,
                len(self._CHAT_RETRY_DELAYS),
                delay,
                (response.content or "")[:120].lower(),
            )
            await asyncio.sleep(delay)

        return await self._safe_chat_stream(**kw)

    async def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = _SENTINEL,
        temperature: object = _SENTINEL,
        reasoning_effort: object = _SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
        prompt_tokens: int | None = None,
    ) -> LLMResponse:
        """Call chat() with retry on transient provider failures.

        Call chain: chat_with_retry() → _safe_chat() → chat()
        The safe wrapper adds timeout and exception-to-response conversion.
        Subclasses override chat() (e.g. FallbackClient for context-exceeded fallback).
        This method handles the retry loop for transient errors (rate limits, etc.).
        Parameters default to ``self.generation`` when not explicitly passed,
        so callers no longer need to thread temperature / max_tokens /
        reasoning_effort through every layer.
        """
        if max_tokens is self._SENTINEL:
            max_tokens = self.generation.max_tokens
        if temperature is self._SENTINEL:
            temperature = self.generation.temperature
        if reasoning_effort is self._SENTINEL:
            reasoning_effort = self.generation.reasoning_effort

        kw: dict[str, Any] = dict(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
            prompt_tokens=prompt_tokens,
        )

        for attempt, delay in enumerate(self._CHAT_RETRY_DELAYS, start=1):
            response = await self._safe_chat(**kw)

            if response.finish_reason != "error":
                return response

            if not self._is_transient_error(response.content):
                stripped = self._strip_image_content(messages)
                if stripped is not None:
                    logger.warning(
                        "Non-transient LLM error with image content, retrying without images: {}",
                        (response.content or "")[:120],
                    )
                    return await self._safe_chat(**{**kw, "messages": stripped})
                stripped_audio = self._strip_audio_content(messages)
                if stripped_audio is not None:
                    logger.warning(
                        "Non-transient LLM error with audio content, retrying without audio: {}",
                        (response.content or "")[:120],
                    )
                    return await self._safe_chat(**{**kw, "messages": stripped_audio})
                return response

            logger.warning(
                "LLM transient error (attempt {}/{}), retrying in {}s: {}",
                attempt,
                len(self._CHAT_RETRY_DELAYS),
                delay,
                (response.content or "")[:120].lower(),
            )
            await asyncio.sleep(delay)

        return await self._safe_chat(**kw)

    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass
