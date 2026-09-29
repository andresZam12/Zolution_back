"""
Anthropic (Claude) LLM provider adapter.

Implements the `LLMProvider` interface using the official `anthropic` Python SDK.

Key feature — Prompt Caching:
    The system prompt for each tenant is long (400-1000 tokens) and identical
    across all messages in a conversation. Anthropic's prompt caching marks
    the system prompt block with `cache_control: {"type": "ephemeral"}`.
    On subsequent calls with the same system prompt, Anthropic serves the
    cached tokens at ~10% of the normal input price, reducing per-conversation
    cost by up to 90% on the system prompt portion.

    Cache lifetime: 5 minutes (Anthropic's current TTL).
    The caching is transparent to the caller — `LLMResponse.cached_tokens`
    reports how many tokens were served from cache for cost tracking.

Error mapping:
    All `anthropic.APIError` subclasses are caught and re-raised as the
    appropriate `LLMProviderError` subclass so the rest of the app never
    sees SDK-specific exceptions.
"""

import logging
from collections.abc import AsyncIterator

import anthropic

from app.llm_providers.base import ConversationMessage, LLMProvider, LLMResponse
from app.llm_providers.exceptions import (
    LLMProviderAuthError,
    LLMProviderContentFilterError,
    LLMProviderError,
    LLMProviderNotConfiguredError,
    LLMProviderRateLimitError,
    LLMProviderTimeoutError,
)

logger = logging.getLogger(__name__)

# Anthropic model identifiers
CLAUDE_HAIKU_4_5 = "claude-haiku-4-5"
CLAUDE_SONNET_4_5 = "claude-sonnet-4-5"


class AnthropicProvider(LLMProvider):
    """
    LLM provider adapter for Anthropic Claude models.

    Instantiate via `get_llm_provider("anthropic")` — do not instantiate
    directly in business logic.

    Args:
        api_key: Anthropic API key. Falls back to `ANTHROPIC_API_KEY` env var.
        model:   Model identifier. Defaults to claude-haiku-4-5 (best cost/quality
                 ratio for appointment scheduling as of the initial benchmark).
        enable_caching: Enable prompt caching on the system prompt block.
                        Should be True in production, False only in tests/benchmarks
                        that measure true cost without cache effects.
    """

    PROVIDER_NAME = "anthropic"
    DEFAULT_MODEL = CLAUDE_HAIKU_4_5

    def __init__(
        self,
        api_key: str = "",
        model: str = CLAUDE_HAIKU_4_5,
        *,
        enable_caching: bool = True,
    ) -> None:
        if not api_key:
            # Fall back to environment variable via the SDK's default behavior
            try:
                self._client = anthropic.AsyncAnthropic()
            except anthropic.AuthenticationError as exc:
                raise LLMProviderNotConfiguredError(
                    "ANTHROPIC_API_KEY is not set or is invalid.",
                    provider=self.PROVIDER_NAME,
                ) from exc
        else:
            self._client = anthropic.AsyncAnthropic(api_key=api_key)

        self._model = model
        self._enable_caching = enable_caching

    def _build_system_block(self, system_prompt: str) -> list[dict]:
        """
        Build the system parameter with optional cache_control.

        When caching is enabled, the system prompt block is marked with
        `cache_control: {"type": "ephemeral"}` so Anthropic caches it
        for up to 5 minutes. Subsequent calls with the same prompt are
        served from cache at ~10% input cost.
        """
        block: dict = {"type": "text", "text": system_prompt}
        if self._enable_caching:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    @staticmethod
    def _to_anthropic_messages(
        messages: list[ConversationMessage],
    ) -> list[dict]:
        """Convert ConversationMessage list to Anthropic's message format."""
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    @staticmethod
    def _map_exception(exc: anthropic.APIError, provider: str) -> LLMProviderError:
        """Map Anthropic SDK exceptions to domain exceptions."""
        if isinstance(exc, anthropic.AuthenticationError):
            return LLMProviderAuthError(str(exc), provider=provider, status_code=401)
        if isinstance(exc, anthropic.RateLimitError):
            return LLMProviderRateLimitError(str(exc), provider=provider, status_code=429)
        if isinstance(exc, anthropic.APITimeoutError):
            return LLMProviderTimeoutError(str(exc), provider=provider)
        if isinstance(exc, anthropic.BadRequestError):
            return LLMProviderContentFilterError(str(exc), provider=provider, status_code=400)
        return LLMProviderError(str(exc), provider=provider)

    async def generate(
        self,
        system_prompt: str,
        messages: list[ConversationMessage],
        *,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> LLMResponse:
        """Generate a response using Claude, with prompt caching on the system prompt."""
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=self._build_system_block(system_prompt),
                messages=self._to_anthropic_messages(messages),
            )
        except anthropic.APIError as exc:
            raise self._map_exception(exc, self.PROVIDER_NAME) from exc

        # Extract cache token counts (only present when caching is active)
        usage = response.usage
        cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0

        logger.debug(
            "Anthropic response | model=%s in=%d out=%d cached=%d",
            self._model,
            usage.input_tokens,
            usage.output_tokens,
            cached_tokens,
        )

        return LLMResponse(
            content=response.content[0].text,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            provider=self.PROVIDER_NAME,
            model=self._model,
            cached_tokens=cached_tokens,
        )

    async def stream(
        self,
        system_prompt: str,
        messages: list[ConversationMessage],
        *,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> AsyncIterator[str]:
        """Stream Claude's response token by token."""
        try:
            async with self._client.messages.stream(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=self._build_system_block(system_prompt),
                messages=self._to_anthropic_messages(messages),
            ) as stream:
                async for text_chunk in stream.text_stream:
                    yield text_chunk
        except anthropic.APIError as exc:
            raise self._map_exception(exc, self.PROVIDER_NAME) from exc
