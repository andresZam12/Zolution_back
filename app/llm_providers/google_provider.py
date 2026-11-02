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

import asyncio
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

GEMINI_3_8_FLASH = "gemini-3.8-flash"
GEMINI_FLASH_LATEST = "gemini-flash-latest"


def _normalize_model_name(model: str) -> str:
    cleaned = (model or "").strip().lower()
    if not cleaned or "2.0" in cleaned or "2.5" in cleaned or "3.8" in cleaned or "flash" in cleaned:
        return GEMINI_3_8_FLASH
    if "pro" in cleaned:
        return "gemini-pro-latest"
    return cleaned


class GoogleProvider(LLMProvider):
    """
    LLM provider adapter for Google Gemini models.

    Args:
        api_key: Google AI Studio API key. Falls back to `GOOGLE_API_KEY` env var.
        model:   Model identifier. Defaults to gemini-3.8-flash.
    """

    PROVIDER_NAME = "google"
    DEFAULT_MODEL = GEMINI_3_8_FLASH

    def __init__(
        self,
        api_key: str = "",
        model: str = GEMINI_3_8_FLASH,
    ) -> None:
        if not api_key:
            import os

            api_key = os.getenv("GOOGLE_API_KEY", "")

        api_key = api_key.strip().strip('"\'')

        if not api_key:
            raise LLMProviderNotConfiguredError(
                "GOOGLE_API_KEY is not set.",
                provider=self.PROVIDER_NAME,
            )

        genai.configure(api_key=api_key)
        self._model_name = _normalize_model_name(model)
        self._generation_config_defaults = {
            "candidate_count": 1,
        }

    def _get_model(
        self,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        model_name: str | None = None,
    ) -> genai.GenerativeModel:
        """Create a GenerativeModel instance with standard GenerationConfig."""
        return genai.GenerativeModel(
            model_name=model_name or self._model_name,
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

        Google Gemini requirements:
        1. First turn MUST be from 'user'.
        2. Turns MUST alternate between 'user' and 'model'.
        3. Consecutive turns with the same role must be merged.
        4. Cannot be empty.
        """
        clean_turns: list[dict] = []
        for msg in messages:
            content = (msg.content or "").strip()
            if not content:
                continue
            role = "model" if msg.role in ("assistant", "model") else "user"

            if not clean_turns:
                # First turn must be user
                if role != "user":
                    continue
                clean_turns.append({"role": "user", "parts": [{"text": content}]})
            else:
                if clean_turns[-1]["role"] == role:
                    prev_text = clean_turns[-1]["parts"][0]["text"]
                    clean_turns[-1]["parts"] = [{"text": f"{prev_text}\n\n{content}"}]
                else:
                    clean_turns.append({"role": role, "parts": [{"text": content}]})

        if not clean_turns:
            clean_turns = [{"role": "user", "parts": [{"text": "Hola"}]}]

        return clean_turns

    @staticmethod
    def _map_exception(exc: Exception, provider: str) -> LLMProviderError:
        """Map Google API exceptions to domain exceptions."""
        if isinstance(exc, (DeadlineExceeded, TimeoutError, asyncio.TimeoutError)):
            return LLMProviderTimeoutError(f"Google Gemini timed out: {exc}", provider=provider)
        if isinstance(exc, PermissionDenied):
            return LLMProviderAuthError(str(exc), provider=provider, status_code=403)
        if isinstance(exc, ResourceExhausted):
            return LLMProviderRateLimitError(str(exc), provider=provider, status_code=429)
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
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Generate a response using Gemini."""
        if tools:
            logger.warning("Tool calling not yet implemented for Gemini in this MVP.")

        google_msgs = self._to_google_messages(messages)
        model = self._get_model(system_prompt, temperature, max_tokens)
        response = None

        try:
            response = await asyncio.wait_for(
                model.generate_content_async(google_msgs),
                timeout=12.0,
            )
        except Exception as exc:
            # If 3.8 flash failed with not found/available, retry with gemini-flash-latest
            if (
                isinstance(exc, (DeadlineExceeded, TimeoutError, asyncio.TimeoutError))
                or "not found" in str(exc).lower()
                or "no longer available" in str(exc).lower()
            ) and self._model_name != GEMINI_FLASH_LATEST:
                logger.info("Gemini %s failed or timed out; falling back to %s", self._model_name, GEMINI_FLASH_LATEST)
                try:
                    fallback_model = self._get_model(system_prompt, temperature, max_tokens, GEMINI_FLASH_LATEST)
                    response = await asyncio.wait_for(
                        fallback_model.generate_content_async(google_msgs),
                        timeout=12.0,
                    )
                    self._model_name = GEMINI_FLASH_LATEST
                except Exception as fb_exc:
                    raise self._map_exception(fb_exc, self.PROVIDER_NAME) from fb_exc
            else:
                raise self._map_exception(exc, self.PROVIDER_NAME) from exc

        # Handle content extraction safely
        reply_text = ""
        try:
            if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                part_texts = [
                    getattr(p, "text", "") for p in response.candidates[0].content.parts
                    if getattr(p, "text", None)
                ]
                reply_text = "".join(part_texts).strip()
            if not reply_text and hasattr(response, "text"):
                reply_text = (response.text or "").strip()
        except Exception:
            try:
                reply_text = (response.text or "").strip()
            except Exception:
                reply_text = ""

        if not reply_text:
            raise LLMProviderContentFilterError(
                "Gemini returned no text content (possible content filter).",
                provider=self.PROVIDER_NAME,
            )

        usage = getattr(response, "usage_metadata", None)
        in_tokens = getattr(usage, "prompt_token_count", 0) or 0 if usage else 0
        out_tokens = getattr(usage, "candidates_token_count", 0) or 0 if usage else 0

        logger.debug(
            "Google response | model=%s in=%d out=%d",
            self._model_name,
            in_tokens,
            out_tokens,
        )

        return LLMResponse(
            content=reply_text,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            provider=self.PROVIDER_NAME,
            model=self._model_name,
            cached_tokens=0,
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
        """Stream Gemini's response chunk by chunk."""
        if tools:
            logger.warning("Tool calling not yet implemented for Gemini in this MVP.")

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
