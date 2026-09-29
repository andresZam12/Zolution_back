"""
OpenAI (GPT) LLM provider adapter.

Implements the `LLMProvider` interface using the official `openai` Python SDK
with its async client (`AsyncOpenAI`).

Notable differences from Anthropic's API that this adapter handles:
- System prompt goes as the first message with role "system" in the `messages`
  list (OpenAI does not have a separate `system` parameter in the chat API).
- Token counts are in `response.usage` (prompt_tokens, completion_tokens).
- Streaming uses `AsyncStream[ChatCompletionChunk]`.

Model recommendation:
    gpt-4o-mini — $0.25/$2.00 per 1M tokens input/output, 128K context.
    Strong ecosystem, good balance of cost and quality. Benchmark candidate.
"""

import logging
from collections.abc import AsyncIterator

import openai

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

GPT_4O_MINI = "gpt-4o-mini"
GPT_4O = "gpt-4o"


class OpenAIProvider(LLMProvider):
    """
    LLM provider adapter for OpenAI GPT models.

    Args:
        api_key: OpenAI API key. Falls back to `OPENAI_API_KEY` env var.
        model:   Model identifier. Defaults to gpt-4o-mini.
    """

    PROVIDER_NAME = "openai"
    DEFAULT_MODEL = GPT_4O_MINI

    def __init__(
        self,
        api_key: str = "",
        model: str = GPT_4O_MINI,
    ) -> None:
        if not api_key:
            import os  # noqa: PLC0415
            api_key = os.getenv("OPENAI_API_KEY", "")

        if not api_key:
            raise LLMProviderNotConfiguredError(
                "OPENAI_API_KEY is not set.",
                provider=self.PROVIDER_NAME,
            )

        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model

    @staticmethod
    def _build_messages(
        system_prompt: str,
        messages: list[ConversationMessage],
    ) -> list[dict]:
        """
        Build the OpenAI messages list.

        OpenAI places the system prompt as the first message with role "system".
        Conversation history follows in chronological order.
        """
        return [
            {"role": "system", "content": system_prompt},
            *[{"role": msg.role, "content": msg.content} for msg in messages],
        ]

    @staticmethod
    def _map_exception(exc: openai.OpenAIError, provider: str) -> LLMProviderError:
        """Map OpenAI SDK exceptions to domain exceptions."""
        if isinstance(exc, openai.AuthenticationError):
            return LLMProviderAuthError(str(exc), provider=provider, status_code=401)
        if isinstance(exc, openai.RateLimitError):
            return LLMProviderRateLimitError(str(exc), provider=provider, status_code=429)
        if isinstance(exc, openai.APITimeoutError):
            return LLMProviderTimeoutError(str(exc), provider=provider)
        if isinstance(exc, openai.BadRequestError):
            # Often indicates content policy violation
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
        """Generate a response using GPT."""
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=self._build_messages(system_prompt, messages),  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except openai.OpenAIError as exc:
            raise self._map_exception(exc, self.PROVIDER_NAME) from exc

        choice = response.choices[0]

        # Content filter check
        if choice.finish_reason == "content_filter":
            raise LLMProviderContentFilterError(
                "OpenAI content filter blocked the response.",
                provider=self.PROVIDER_NAME,
            )

        usage = response.usage
        logger.debug(
            "OpenAI response | model=%s in=%d out=%d",
            self._model,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )

        return LLMResponse(
            content=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            provider=self.PROVIDER_NAME,
            model=self._model,
            cached_tokens=0,
        )

    async def stream(
        self,
        system_prompt: str,
        messages: list[ConversationMessage],
        *,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> AsyncIterator[str]:
        """Stream GPT's response token by token."""
        try:
            async with await self._client.chat.completions.create(
                model=self._model,
                messages=self._build_messages(system_prompt, messages),  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            ) as stream:
                async for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
        except openai.OpenAIError as exc:
            raise self._map_exception(exc, self.PROVIDER_NAME) from exc
