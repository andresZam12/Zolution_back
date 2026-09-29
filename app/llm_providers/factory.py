"""
LLM provider factory.

Single point of instantiation for all `LLMProvider` concrete adapters.
Business logic should never import adapters directly — always use this factory.

This enables:
- Per-tenant provider routing (e.g. enterprise tenants use GPT-4o,
  standard tenants use Claude Haiku).
- Easy addition of new providers: add a class, register it in `_REGISTRY`.
- Consistent configuration: API keys are always sourced from `Settings`.
"""

import logging

from app.core.config import get_settings
from app.llm_providers.base import LLMProvider
from app.llm_providers.exceptions import LLMProviderNotConfiguredError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider registry
# Registry maps provider name -> (adapter class, api_key_getter)
# Lazy imports inside the tuples to avoid loading all SDKs at startup
# ---------------------------------------------------------------------------

def _get_anthropic_provider(model: str = "", *, enable_caching: bool = True) -> LLMProvider:
    from app.llm_providers.anthropic_provider import AnthropicProvider  # noqa: PLC0415

    settings = get_settings()
    api_key = settings.ANTHROPIC_API_KEY
    effective_model = model or AnthropicProvider.DEFAULT_MODEL

    if not api_key:
        raise LLMProviderNotConfiguredError(
            "ANTHROPIC_API_KEY is not configured.",
            provider="anthropic",
        )
    return AnthropicProvider(api_key=api_key, model=effective_model, enable_caching=enable_caching)


def _get_google_provider(model: str = "") -> LLMProvider:
    from app.llm_providers.google_provider import GoogleProvider  # noqa: PLC0415

    settings = get_settings()
    api_key = settings.GOOGLE_API_KEY
    effective_model = model or GoogleProvider.DEFAULT_MODEL

    if not api_key:
        raise LLMProviderNotConfiguredError(
            "GOOGLE_API_KEY is not configured.",
            provider="google",
        )
    return GoogleProvider(api_key=api_key, model=effective_model)


def _get_openai_provider(model: str = "") -> LLMProvider:
    from app.llm_providers.openai_provider import OpenAIProvider  # noqa: PLC0415

    settings = get_settings()
    api_key = settings.OPENAI_API_KEY
    effective_model = model or OpenAIProvider.DEFAULT_MODEL

    if not api_key:
        raise LLMProviderNotConfiguredError(
            "OPENAI_API_KEY is not configured.",
            provider="openai",
        )
    return OpenAIProvider(api_key=api_key, model=effective_model)


# Supported providers and their factory functions
_PROVIDER_FACTORIES = {
    "anthropic": _get_anthropic_provider,
    "google": _get_google_provider,
    "openai": _get_openai_provider,
}

SUPPORTED_PROVIDERS = frozenset(_PROVIDER_FACTORIES.keys())


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def get_llm_provider(
    provider_name: str,
    model: str = "",
    *,
    enable_caching: bool = True,
) -> LLMProvider:
    """
    Return a configured LLMProvider instance by provider name.

    Args:
        provider_name:   One of "anthropic", "google", "openai".
        model:           Optional model override. Uses provider default if empty.
        enable_caching:  Enable prompt caching (Anthropic only). Default True.

    Returns:
        A configured LLMProvider ready to call.

    Raises:
        LLMProviderNotConfiguredError: Provider name unknown or API key missing.

    Example:
        provider = get_llm_provider("anthropic")
        response = await provider.generate(system_prompt, messages)
    """
    if provider_name not in _PROVIDER_FACTORIES:
        raise LLMProviderNotConfiguredError(
            f"Unknown LLM provider: {provider_name!r}. "
            f"Supported: {sorted(SUPPORTED_PROVIDERS)}",
            provider=provider_name,
        )

    factory = _PROVIDER_FACTORIES[provider_name]

    # Only pass enable_caching to Anthropic (other providers ignore it)
    kwargs: dict = {"model": model}
    if provider_name == "anthropic":
        kwargs["enable_caching"] = enable_caching

    provider = factory(**kwargs)
    logger.debug("LLM provider instantiated: %r", provider)
    return provider


def get_provider_for_agent(
    llm_provider: str,
    llm_model: str = "",
) -> LLMProvider:
    """
    Return the LLM provider configured for a specific tenant agent.

    This is the primary entry point used by the agent orchestrator.
    It reads `llm_provider` and `llm_model` from the tenant's `AgentConfig`
    (stored in the DB) and returns a ready provider instance.

    Args:
        llm_provider: Provider name from `agent_configs.llm_provider`.
        llm_model:    Model name from `agent_configs.llm_model` (empty = default).

    Example:
        provider = get_provider_for_agent(
            agent_config.llm_provider,
            agent_config.llm_model,
        )
        response = await provider.generate(agent_config.system_prompt_generated, messages)
    """
    return get_llm_provider(llm_provider, model=llm_model, enable_caching=True)
