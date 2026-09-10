"""
core.exceptions — custom exception types for the LLM Cost Autopilot.

All provider-level errors are wrapped in ProviderError before being surfaced
to the rest of the application. This keeps httpx, API-specific, and timeout
errors from leaking out of the router_client layer.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Raised when an LLM provider call fails for any reason.

    Attributes:
        provider: The name of the provider that raised the error
                  (e.g. "openrouter", "groq", "gemini", "mistral", "ollama").
        original_error: The underlying exception that caused this failure.
        status_code: The HTTP status code returned by the provider, if any.
                     None for network-level failures (e.g. timeout, DNS error).
    """

    def __init__(
        self,
        provider: str,
        original_error: Exception,
        status_code: int | None = None,
    ) -> None:
        self.provider = provider
        self.original_error = original_error
        self.status_code = status_code

        status_part = f" (HTTP {status_code})" if status_code is not None else ""
        super().__init__(
            f"[{provider}]{status_part} {type(original_error).__name__}: {original_error}"
        )

    def __repr__(self) -> str:
        return (
            f"ProviderError(provider={self.provider!r}, "
            f"status_code={self.status_code!r}, "
            f"original_error={self.original_error!r})"
        )
