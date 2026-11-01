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

CLAUDE_3_5_HAIKU = "claude-3-5-haiku-latest"
CLAUDE_3_5_SONNET = "claude-3-5-sonnet-latest"
CLAUDE_3_HAIKU_FALLBACK = "claude-3-haiku-20240307"


def _normalize_anthropic_model(model: str) -> str:
    cleaned = (model or "").strip().lower()
    if not cleaned or "haiku" in cleaned:
        return CLAUDE_3_5_HAIKU
    if "sonnet" in cleaned:
        return CLAUDE_3_5_SONNET
    return cleaned


class AnthropicProvider(LLMProvider):
    """
    LLM provider adapter for Anthropic Claude models.

    Instantiate via `get_llm_provider("anthropic")` — do not instantiate
    directly in business logic.

    Args:
        api_key: Anthropic API key. Falls back to `ANTHROPIC_API_KEY` env var.
        model:   Model identifier. Defaults to claude-3-5-haiku-latest.
        enable_caching: Enable prompt caching on the system prompt block.
    """

    PROVIDER_NAME = "anthropic"
    DEFAULT_MODEL = CLAUDE_3_5_HAIKU

    def __init__(
        self,
        api_key: str = "",
        model: str = CLAUDE_3_5_HAIKU,
        *,
        enable_caching: bool = True,
    ) -> None:
        if not api_key:
            import os

            api_key = os.getenv("ANTHROPIC_API_KEY", "")

        api_key = api_key.strip().strip('"\'')

        if not api_key:
            raise LLMProviderNotConfiguredError(
                "ANTHROPIC_API_KEY is not set or is invalid.",
                provider=self.PROVIDER_NAME,
            )

        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = _normalize_anthropic_model(model)
        self._enable_caching = enable_caching

    def _build_system_block(self, system_prompt: str) -> list[dict]:
        """
        Build the system parameter with optional cache_control.
        """
        block: dict = {"type": "text", "text": system_prompt}
        if self._enable_caching:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    @staticmethod
    def _to_anthropic_messages(
        messages: list[ConversationMessage],
    ) -> list[dict]:
        """
        Convert ConversationMessage list to Anthropic's message format.

        Anthropic requirements:
        1. First message MUST have role 'user'.
        2. Roles MUST alternate between 'user' and 'assistant'.
        3. Consecutive messages of the same role must be merged.
        4. Cannot be empty.
        """
        clean_turns: list[dict] = []
        for msg in messages:
            content = (msg.content or "").strip()
            if not content:
                continue
            role = "assistant" if msg.role in ("assistant", "model") else "user"

            if not clean_turns:
                # First turn must be user
                if role != "user":
                    continue
                clean_turns.append({"role": "user", "content": content})
            else:
                if clean_turns[-1]["role"] == role:
                    clean_turns[-1]["content"] = f"{clean_turns[-1]['content']}\n\n{content}"
                else:
                    clean_turns.append({"role": role, "content": content})

        if not clean_turns:
            clean_turns = [{"role": "user", "content": "Hola"}]

        return clean_turns

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
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Generate a response using Claude, with prompt caching on the system prompt."""
        if tools:
            logger.warning("Tool calling not yet implemented for Anthropic in this MVP.")

        anth_msgs = self._to_anthropic_messages(messages)
        response = None

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=self._build_system_block(system_prompt),
                messages=anth_msgs,
            )
        except anthropic.APIError as exc:
            # If latest haiku not found, fallback to claude-3-haiku-20240307
            if "not_found" in str(exc).lower() and self._model != CLAUDE_3_HAIKU_FALLBACK:
                logger.info("Claude %s not found; falling back to %s", self._model, CLAUDE_3_HAIKU_FALLBACK)
                try:
                    response = await self._client.messages.create(
                        model=CLAUDE_3_HAIKU_FALLBACK,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=self._build_system_block(system_prompt),
                        messages=anth_msgs,
                    )
                    self._model = CLAUDE_3_HAIKU_FALLBACK
                except anthropic.APIError as fb_exc:
                    raise self._map_exception(fb_exc, self.PROVIDER_NAME) from fb_exc
            else:
                raise self._map_exception(exc, self.PROVIDER_NAME) from exc

        # Extract cache token counts (only present when caching is active)
        usage = response.usage
        cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0

        # Extract reply text safely
        text_blocks = [
            getattr(block, "text", "") for block in response.content
            if getattr(block, "type", "") == "text" and hasattr(block, "text")
        ]
        reply_content = "".join(text_blocks).strip()

        logger.debug(
            "Anthropic response | model=%s in=%d out=%d cached=%d",
            self._model,
            usage.input_tokens,
            usage.output_tokens,
            cached_tokens,
        )

        return LLMResponse(
            content=reply_content,
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
        tools: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Stream Claude's response token by token."""
        if tools:
            logger.warning("Tool calling not yet implemented for Anthropic in this MVP.")

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
