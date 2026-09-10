"""
core.router_client — unified async LLM call dispatcher.

This module is the *only* place in the codebase that knows how to talk to
specific LLM providers. Every other module calls ``send_request`` and receives
a normalized ``LLMResponse`` back.

Provider dispatch map
---------------------
  openrouter  →  _call_openrouter   (OpenAI-compatible, chat/completions)
  groq        →  _call_groq         (OpenAI-compatible, different base URL)
  gemini      →  _call_gemini       (Google GenerateContent REST API)
  mistral     →  _call_mistral      (OpenAI-compatible, mistral base URL)
  ollama      →  _call_ollama       (Ollama /api/chat, local)

Retry policy
------------
Rate-limit (HTTP 429) and server errors (HTTP 503) are retried up to 3 times
with exponential backoff (1 s → 2 s → 4 s), implemented via tenacity.
All other errors are wrapped in ProviderError and re-raised immediately.

Environment variables read (via python-dotenv)
----------------------------------------------
  OPENROUTER_API_KEY
  GROQ_API_KEY
  GEMINI_API_KEY
  MISTRAL_API_KEY
  OLLAMA_BASE_URL         (default: http://localhost:11434)
  HTTP_TIMEOUT_SECONDS    (default: 60)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from core.exceptions import ProviderError
from models.registry import ModelConfig
from models.response import LLMResponse

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _get_required_env(key: str) -> str:
    """Read a required environment variable.

    Args:
        key: The name of the environment variable.

    Returns:
        The variable's value as a string.

    Raises:
        EnvironmentError: If the variable is not set or is empty.
    """
    value = os.getenv(key, "").strip()
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Copy .env.example to .env and fill in your API keys."
        )
    return value


def _http_timeout() -> float:
    """Return the configured HTTP timeout in seconds (default 60)."""
    try:
        return float(os.getenv("HTTP_TIMEOUT_SECONDS", "60"))
    except ValueError:
        return 60.0


# ---------------------------------------------------------------------------
# Retry predicate — only retry on rate-limit or server errors
# ---------------------------------------------------------------------------

class _RetryableHTTPError(Exception):
    """Internal marker exception used by the retry predicate."""


def _is_retryable(exc: BaseException) -> bool:
    """Return True if the exception should trigger a retry attempt."""
    return isinstance(exc, _RetryableHTTPError)


def _retry_decorator() -> Any:
    """Build the tenacity retry decorator used by all provider functions."""
    return retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

def _calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model_config: ModelConfig,
) -> float:
    """Calculate the USD cost of a single API call.

    Args:
        input_tokens: Number of prompt / input tokens consumed.
        output_tokens: Number of generated / output tokens produced.
        model_config: The ModelConfig for the model that served the request.

    Returns:
        Estimated cost in USD. Always 0.0 for Ollama models.
    """
    return (
        input_tokens * model_config.cost_per_input_token
        + output_tokens * model_config.cost_per_output_token
    )


# ---------------------------------------------------------------------------
# Provider: OpenRouter (OpenAI-compatible)
# ---------------------------------------------------------------------------

@_retry_decorator()
async def _call_openrouter(
    prompt: str,
    model_config: ModelConfig,
    system_prompt: str | None,
    max_tokens: int,
) -> LLMResponse:
    """Call an OpenRouter-hosted model using the chat completions API.

    OpenRouter's API is fully OpenAI-compatible; only the base URL and the
    ``Authorization`` header differ.

    Args:
        prompt: User message content.
        model_config: Registry entry for the target model.
        system_prompt: Optional system instruction injected as the first message.
        max_tokens: Maximum number of output tokens to generate.

    Returns:
        Normalized LLMResponse.

    Raises:
        _RetryableHTTPError: On HTTP 429 or 503 — triggers tenacity retry.
        ProviderError: On all other failures (auth, bad request, network, etc.).
    """
    api_key = _get_required_env("OPENROUTER_API_KEY")
    base_url = "https://openrouter.ai/api/v1"

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_config.model_id,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/routemind-ai",  # OpenRouter best practice
        "X-Title": "LLM Cost Autopilot",
    }

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=_http_timeout()) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
            )

        if response.status_code in (429, 503):
            logger.warning(
                "OpenRouter returned %s — will retry", response.status_code
            )
            raise _RetryableHTTPError(f"HTTP {response.status_code}")

        if response.status_code != 200:
            raise ProviderError(
                provider="openrouter",
                original_error=ValueError(response.text),
                status_code=response.status_code,
            )

        latency_ms = (time.perf_counter() - start) * 1000
        raw: dict[str, Any] = response.json()

        # OpenAI-compatible usage block
        usage = raw.get("usage", {})
        input_tokens: int = usage.get("prompt_tokens", 0)
        output_tokens: int = usage.get("completion_tokens", 0)

        output_text: str = (
            raw["choices"][0]["message"]["content"].strip()
        )

        return LLMResponse(
            output_text=output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=_calculate_cost(input_tokens, output_tokens, model_config),
            model_id=model_config.model_id,
            provider="openrouter",
            raw_response=raw,
        )

    except ProviderError:
        raise
    except _RetryableHTTPError:
        raise
    except httpx.TimeoutException as exc:
        raise ProviderError(
            provider="openrouter", original_error=exc
        ) from exc
    except httpx.HTTPError as exc:
        raise ProviderError(
            provider="openrouter", original_error=exc
        ) from exc
    except (KeyError, IndexError, ValueError) as exc:
        raise ProviderError(
            provider="openrouter",
            original_error=exc,
        ) from exc


# ---------------------------------------------------------------------------
# Provider: Groq (OpenAI-compatible, different base URL)
# ---------------------------------------------------------------------------

@_retry_decorator()
async def _call_groq(
    prompt: str,
    model_config: ModelConfig,
    system_prompt: str | None,
    max_tokens: int,
) -> LLMResponse:
    """Call a Groq-hosted model using Groq's OpenAI-compatible API.

    Args:
        prompt: User message content.
        model_config: Registry entry for the target model.
        system_prompt: Optional system instruction.
        max_tokens: Maximum output tokens.

    Returns:
        Normalized LLMResponse.

    Raises:
        _RetryableHTTPError: On HTTP 429 or 503.
        ProviderError: On all other failures.
    """
    api_key = _get_required_env("GROQ_API_KEY")
    base_url = "https://api.groq.com/openai/v1"

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_config.model_id,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=_http_timeout()) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
            )

        if response.status_code in (429, 503):
            logger.warning("Groq returned %s — will retry", response.status_code)
            raise _RetryableHTTPError(f"HTTP {response.status_code}")

        if response.status_code != 200:
            raise ProviderError(
                provider="groq",
                original_error=ValueError(response.text),
                status_code=response.status_code,
            )

        latency_ms = (time.perf_counter() - start) * 1000
        raw: dict[str, Any] = response.json()

        usage = raw.get("usage", {})
        input_tokens: int = usage.get("prompt_tokens", 0)
        output_tokens: int = usage.get("completion_tokens", 0)

        output_text: str = raw["choices"][0]["message"]["content"].strip()

        return LLMResponse(
            output_text=output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=_calculate_cost(input_tokens, output_tokens, model_config),
            model_id=model_config.model_id,
            provider="groq",
            raw_response=raw,
        )

    except ProviderError:
        raise
    except _RetryableHTTPError:
        raise
    except httpx.TimeoutException as exc:
        raise ProviderError(provider="groq", original_error=exc) from exc
    except httpx.HTTPError as exc:
        raise ProviderError(provider="groq", original_error=exc) from exc
    except (KeyError, IndexError, ValueError) as exc:
        raise ProviderError(provider="groq", original_error=exc) from exc


# ---------------------------------------------------------------------------
# Provider: Google Gemini (distinct GenerateContent REST API)
# ---------------------------------------------------------------------------

@_retry_decorator()
async def _call_gemini(
    prompt: str,
    model_config: ModelConfig,
    system_prompt: str | None,
    max_tokens: int,
) -> LLMResponse:
    """Call a Google Gemini model using the GenerateContent REST API.

    Gemini uses a structurally different request and response schema from
    the OpenAI-compatible providers — it does NOT use the ``messages`` array
    format. This function handles the translation explicitly.

    Token counts are in ``usageMetadata.promptTokenCount`` /
    ``usageMetadata.candidatesTokenCount`` (not ``usage.prompt_tokens``).

    Args:
        prompt: User message content.
        model_config: Registry entry for the target model.
        system_prompt: Optional system instruction (passed via
                       ``system_instruction`` field).
        max_tokens: Maximum output tokens.

    Returns:
        Normalized LLMResponse.

    Raises:
        _RetryableHTTPError: On HTTP 429 or 503.
        ProviderError: On all other failures.
    """
    api_key = _get_required_env("GEMINI_API_KEY")
    model_name = model_config.model_id  # e.g. "gemini-2.0-flash"
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={api_key}"
    )

    # Gemini request schema
    payload: dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
        },
    }
    if system_prompt:
        payload["system_instruction"] = {
            "parts": [{"text": system_prompt}]
        }

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=_http_timeout()) as client:
            response = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

        if response.status_code in (429, 503):
            logger.warning("Gemini returned %s — will retry", response.status_code)
            raise _RetryableHTTPError(f"HTTP {response.status_code}")

        if response.status_code != 200:
            raise ProviderError(
                provider="gemini",
                original_error=ValueError(response.text),
                status_code=response.status_code,
            )

        latency_ms = (time.perf_counter() - start) * 1000
        raw: dict[str, Any] = response.json()

        # Gemini-specific token count location
        usage_meta = raw.get("usageMetadata", {})
        input_tokens: int = usage_meta.get("promptTokenCount", 0)
        output_tokens: int = usage_meta.get("candidatesTokenCount", 0)

        # Gemini response text location
        output_text: str = (
            raw["candidates"][0]["content"]["parts"][0]["text"].strip()
        )

        return LLMResponse(
            output_text=output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=_calculate_cost(input_tokens, output_tokens, model_config),
            model_id=model_config.model_id,
            provider="gemini",
            raw_response=raw,
        )

    except ProviderError:
        raise
    except _RetryableHTTPError:
        raise
    except httpx.TimeoutException as exc:
        raise ProviderError(provider="gemini", original_error=exc) from exc
    except httpx.HTTPError as exc:
        raise ProviderError(provider="gemini", original_error=exc) from exc
    except (KeyError, IndexError, ValueError) as exc:
        raise ProviderError(provider="gemini", original_error=exc) from exc


# ---------------------------------------------------------------------------
# Provider: Mistral (OpenAI-compatible)
# ---------------------------------------------------------------------------

@_retry_decorator()
async def _call_mistral(
    prompt: str,
    model_config: ModelConfig,
    system_prompt: str | None,
    max_tokens: int,
) -> LLMResponse:
    """Call a Mistral model using Mistral's OpenAI-compatible API.

    Args:
        prompt: User message content.
        model_config: Registry entry for the target model.
        system_prompt: Optional system instruction.
        max_tokens: Maximum output tokens.

    Returns:
        Normalized LLMResponse.

    Raises:
        _RetryableHTTPError: On HTTP 429 or 503.
        ProviderError: On all other failures.
    """
    api_key = _get_required_env("MISTRAL_API_KEY")
    base_url = "https://api.mistral.ai/v1"

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_config.model_id,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=_http_timeout()) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
            )

        if response.status_code in (429, 503):
            logger.warning("Mistral returned %s — will retry", response.status_code)
            raise _RetryableHTTPError(f"HTTP {response.status_code}")

        if response.status_code != 200:
            raise ProviderError(
                provider="mistral",
                original_error=ValueError(response.text),
                status_code=response.status_code,
            )

        latency_ms = (time.perf_counter() - start) * 1000
        raw: dict[str, Any] = response.json()

        usage = raw.get("usage", {})
        input_tokens: int = usage.get("prompt_tokens", 0)
        output_tokens: int = usage.get("completion_tokens", 0)

        output_text: str = raw["choices"][0]["message"]["content"].strip()

        return LLMResponse(
            output_text=output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=_calculate_cost(input_tokens, output_tokens, model_config),
            model_id=model_config.model_id,
            provider="mistral",
            raw_response=raw,
        )

    except ProviderError:
        raise
    except _RetryableHTTPError:
        raise
    except httpx.TimeoutException as exc:
        raise ProviderError(provider="mistral", original_error=exc) from exc
    except httpx.HTTPError as exc:
        raise ProviderError(provider="mistral", original_error=exc) from exc
    except (KeyError, IndexError, ValueError) as exc:
        raise ProviderError(provider="mistral", original_error=exc) from exc


# ---------------------------------------------------------------------------
# Provider: Ollama (local, no API key required)
# ---------------------------------------------------------------------------

@_retry_decorator()
async def _call_ollama(
    prompt: str,
    model_config: ModelConfig,
    system_prompt: str | None,
    max_tokens: int,
) -> LLMResponse:
    """Call a locally-running Ollama model via the Ollama /api/chat endpoint.

    Ollama uses a chat-style API similar to OpenAI but with ``stream: false``
    required for a single-shot JSON response. Token counts are in
    ``prompt_eval_count`` and ``eval_count`` (not ``usage.prompt_tokens``).

    The Ollama base URL defaults to http://localhost:11434 and can be overridden
    via the ``OLLAMA_BASE_URL`` environment variable (e.g. for Docker setups).

    Args:
        prompt: User message content.
        model_config: Registry entry for the target model.
        system_prompt: Optional system instruction.
        max_tokens: Maximum output tokens (passed as ``num_predict`` option).

    Returns:
        Normalized LLMResponse. cost_usd is always 0.0 for local models.

    Raises:
        _RetryableHTTPError: On HTTP 429 or 503.
        ProviderError: On all other failures (model not pulled, connection
                       refused, etc.).
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": model_config.model_id,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
        },
    }

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=_http_timeout()) as client:
            response = await client.post(
                f"{base_url}/api/chat",
                json=payload,
            )

        if response.status_code in (429, 503):
            logger.warning("Ollama returned %s — will retry", response.status_code)
            raise _RetryableHTTPError(f"HTTP {response.status_code}")

        if response.status_code != 200:
            raise ProviderError(
                provider="ollama",
                original_error=ValueError(response.text),
                status_code=response.status_code,
            )

        latency_ms = (time.perf_counter() - start) * 1000
        raw: dict[str, Any] = response.json()

        # Ollama-specific token count field names
        input_tokens: int = raw.get("prompt_eval_count", 0)
        output_tokens: int = raw.get("eval_count", 0)

        output_text: str = raw["message"]["content"].strip()

        return LLMResponse(
            output_text=output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=0.0,  # Local models have no API cost
            model_id=model_config.model_id,
            provider="ollama",
            raw_response=raw,
        )

    except ProviderError:
        raise
    except _RetryableHTTPError:
        raise
    except httpx.ConnectError as exc:
        # Ollama not running — give a clear, actionable error message
        raise ProviderError(
            provider="ollama",
            original_error=ConnectionError(
                f"Cannot connect to Ollama at {base_url}. "
                f"Is Ollama running? Start it with: ollama serve"
            ),
        ) from exc
    except httpx.TimeoutException as exc:
        raise ProviderError(provider="ollama", original_error=exc) from exc
    except httpx.HTTPError as exc:
        raise ProviderError(provider="ollama", original_error=exc) from exc
    except (KeyError, IndexError, ValueError) as exc:
        raise ProviderError(provider="ollama", original_error=exc) from exc


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_PROVIDER_DISPATCH: dict[
    str,
    Any,  # Callable[[str, ModelConfig, str | None, int], Awaitable[LLMResponse]]
] = {
    "openrouter": _call_openrouter,
    "groq": _call_groq,
    "gemini": _call_gemini,
    "mistral": _call_mistral,
    "ollama": _call_ollama,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def send_request(
    prompt: str,
    model_config: ModelConfig,
    system_prompt: str | None = None,
    max_tokens: int = 1024,
) -> LLMResponse:
    """Send a prompt to any registered model and return a normalized response.

    This is the *only* function that callers outside ``core/`` should use.
    It dispatches to the correct private provider function based on
    ``model_config.provider`` and always returns an ``LLMResponse``.

    Args:
        prompt: The user's message or instruction. Must not be empty.
        model_config: The ``ModelConfig`` instance for the target model,
                      obtained from ``models.registry.get_model()``.
        system_prompt: Optional system-level instruction injected before the
                       user message. Supported by all providers; Ollama and
                       Gemini handle it via provider-specific fields.
        max_tokens: Maximum number of output tokens to generate. Defaults to
                    1024. Adjust for tasks that require longer or shorter
                    responses.

    Returns:
        A normalized ``LLMResponse`` instance containing the generated text,
        token counts, latency, cost, and the full raw provider response.

    Raises:
        ValueError: If ``prompt`` is empty or ``model_config.provider`` is
                    not in the known dispatch table.
        ProviderError: If the provider call fails after exhausting retries.
                       Carries the provider name and the underlying exception.

    Example::

        from models.registry import get_model
        from core.router_client import send_request

        model = get_model("llama-3.1-8b-instant")
        response = await send_request(
            prompt="What is the capital of France?",
            model_config=model,
            max_tokens=256,
        )
        print(response.output_text)   # "Paris"
        print(f"Cost: ${response.cost_usd:.6f}")
    """
    if not prompt.strip():
        raise ValueError("'prompt' must not be empty.")

    provider = model_config.provider
    if provider not in _PROVIDER_DISPATCH:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported providers: {list(_PROVIDER_DISPATCH.keys())}"
        )

    logger.debug(
        "Dispatching request to %s / %s (max_tokens=%d)",
        provider,
        model_config.model_id,
        max_tokens,
    )

    handler = _PROVIDER_DISPATCH[provider]
    return await handler(prompt, model_config, system_prompt, max_tokens)
