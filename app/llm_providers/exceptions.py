"""
Domain exceptions for the LLM provider layer.

All provider SDK exceptions are caught in the adapters and re-raised as
one of these typed exceptions. The rest of the application only ever sees
`LLMProviderError` subclasses — never Anthropic/Google/OpenAI SDK errors.

This decouples error handling from the specific provider in use and allows
the agent orchestrator to make routing decisions (e.g. retry with a different
provider on rate-limit) without knowing which SDK caused the error.
"""


class LLMProviderError(Exception):
    """
    Base exception for all LLM provider failures.

    Attributes:
        provider: Name of the provider that raised the error.
        message:  Human-readable error description.
        status_code: HTTP status code from the provider API (if applicable).
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code

    def __str__(self) -> str:
        base = super().__str__()
        parts = [base]
        if self.provider:
            parts.append(f"provider={self.provider!r}")
        if self.status_code is not None:
            parts.append(f"status={self.status_code}")
        return " | ".join(parts)


class LLMProviderNotConfiguredError(LLMProviderError):
    """
    Raised when a required API key is missing or empty.

    This is a configuration error, not a runtime failure. It should be
    caught at startup or at factory instantiation time, not mid-request.
    """


class LLMProviderRateLimitError(LLMProviderError):
    """
    Raised when the provider returns HTTP 429 (Too Many Requests).

    The agent orchestrator can use this to implement retry-with-backoff
    or route to a secondary provider.
    """


class LLMProviderTimeoutError(LLMProviderError):
    """
    Raised when the provider request exceeds the configured timeout.

    Distinct from rate-limit to allow different retry strategies.
    """


class LLMProviderAuthError(LLMProviderError):
    """
    Raised when the provider rejects the API key (HTTP 401/403).

    Indicates a configuration issue that retrying will not fix.
    """


class LLMProviderContentFilterError(LLMProviderError):
    """
    Raised when the provider's content filter blocks a request.

    The agent should respond with a generic fallback message rather than
    surfacing the error to the end user.
    """
