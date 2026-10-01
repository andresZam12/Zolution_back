"""
Google (Gemini) LLM provider adapter.

Implements the `LLMProvider` interface using the `google-generativeai` SDK.

Notable differences from Anthropic's API that this adapter handles:
- System instruction is a separate parameter (`system_instruction`), not part
  of the `contents` list.
- Role name mapping: Google uses "model" for assistant turns; this adapter
  accepts the canonical "assistant" role and converts to "model" internally.
- Token counts are in `response.usage_metadata` (not `response.usage`).
- Streaming: `generate_content_async` with `stream=True` yields chunks.

Model recommendation:
    gemini-2.5-flash — 1M context window, strong function calling,
    $0.30/$2.50 per 1M tokens input/output. Competitive candidate in the
    LLM diagnostic benchmark (Section 5.2).
"""

import logging
from collections.abc import AsyncIterator

import google.generativeai as genai
from google.api_core.exceptions import (
    DeadlineExceeded,
    GoogleAPICallError,
    PermissionDenied,
    ResourceExhausted,
)

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

GEMINI_2_5_FLASH = "gemini-2.5-flash"
GEMINI_2_0_FLASH = "gemini-2.0-flash"


class GoogleProvider(LLMProvider):
    """
    LLM provider adapter for Google Gemini models.

    Args:
        api_key: Google AI Studio API key. Falls back to `GOOGLE_API_KEY` env var.
        model:   Model identifier. Defaults to gemini-2.5-flash.
    """

    PROVIDER_NAME = "google"
    DEFAULT_MODEL = GEMINI_2_5_FLASH

    def __init__(
        self,
        api_key: str = "",
        model: str = GEMINI_2_5_FLASH,
    ) -> None:
        if not api_key:
            import os

            api_key = os.getenv("GOOGLE_API_KEY", "")

        if not api_key:
            raise LLMProviderNotConfiguredError(
                "GOOGLE_API_KEY is not set.",
                provider=self.PROVIDER_NAME,
            )

        genai.configure(api_key=api_key)
        self._model_name = model
        self._generation_config_defaults = {
            "candidate_count": 1,
        }

    def _get_model(
        self,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> genai.GenerativeModel:
        """Create a GenerativeModel instance with system instruction and config."""
        return genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                **self._generation_config_defaults,
            ),
        )

    @staticmethod
    def _to_google_messages(messages: list[ConversationMessage]) -> list[dict]:
        """
        Convert ConversationMessage list to Google's `contents` format.

        Google uses "model" for assistant turns — not "assistant".
        """
        return [
            {
                "role": "model" if msg.role == "assistant" else "user",
                "parts": [{"text": msg.content}],
            }
            for msg in messages
        ]

    @staticmethod
    def _map_exception(exc: Exception, provider: str) -> LLMProviderError:
        """Map Google API exceptions to domain exceptions."""
        if isinstance(exc, PermissionDenied):
            return LLMProviderAuthError(str(exc), provider=provider, status_code=403)
        if isinstance(exc, ResourceExhausted):
            return LLMProviderRateLimitError(str(exc), provider=provider, status_code=429)
        if isinstance(exc, DeadlineExceeded):
            return LLMProviderTimeoutError(str(exc), provider=provider)
        if isinstance(exc, GoogleAPICallError):
            return LLMProviderError(str(exc), provider=provider)
        return LLMProviderError(str(exc), provider=provider)

    async def generate(
        self,
        system_prompt: str,
        messages: list[ConversationMessage],
        *,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> LLMResponse:
        """Generate a response using Gemini."""
        model = self._get_model(system_prompt, temperature, max_tokens)
        try:
            response = await model.generate_content_async(self._to_google_messages(messages))
        except Exception as exc:
            raise self._map_exception(exc, self.PROVIDER_NAME) from exc

        # Handle content filtering (Gemini may return empty candidates)
        if not response.candidates or not response.text:
            raise LLMProviderContentFilterError(
                "Gemini returned no content (possible content filter).",
                provider=self.PROVIDER_NAME,
            )

        usage = response.usage_metadata
        logger.debug(
            "Google response | model=%s in=%d out=%d",
            self._model_name,
            usage.prompt_token_count,
            usage.candidates_token_count,
        )

        return LLMResponse(
            content=response.text,
            input_tokens=usage.prompt_token_count,
            output_tokens=usage.candidates_token_count,
            provider=self.PROVIDER_NAME,
            model=self._model_name,
            cached_tokens=0,  # Google does not offer prompt caching at this time
        )

    async def stream(
        self,
        system_prompt: str,
        messages: list[ConversationMessage],
        *,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> AsyncIterator[str]:
        """Stream Gemini's response chunk by chunk."""
        model = self._get_model(system_prompt, temperature, max_tokens)
        try:
            async for chunk in await model.generate_content_async(
                self._to_google_messages(messages),
                stream=True,
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            raise self._map_exception(exc, self.PROVIDER_NAME) from exc
