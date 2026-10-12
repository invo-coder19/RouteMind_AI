"""
tests/test_api.py
=================
pytest tests for the LLM Cost Autopilot API.

All provider calls are mocked — no real API spend during testing.

Coverage:
- POST /v1/completions — successful routing
- POST /v1/completions — force_model override
- POST /v1/completions — invalid force_model
- POST /v1/completions — provider failure + automatic fallback
- POST /v1/completions — all providers fail → 502
- PUT  /v1/routing-config — invalid model_id → 422
- GET  /health
"""
from __future__ import annotations

import pathlib
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Mock AppState — no real DB or classifier needed
# ---------------------------------------------------------------------------

class _MockAppState:
    classifier_ready = True
    db_ready = True
    routing_map = {}


def _make_mock_llm_response(model_id: str = "mock-model", provider: str = "groq") -> MagicMock:
    """Return a MagicMock that looks like an LLMResponse."""
    r = MagicMock()
    r.output_text = "This is a mock answer."
    r.model_id = model_id
    r.provider = provider
    r.input_tokens = 10
    r.output_tokens = 20
    r.cost_usd = 0.000005
    r.latency_ms = 150.0
    return r


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """TestClient with mocked lifespan (no real DB or classifier load)."""
    with (
        patch("classifier.predict._load_model", return_value=MagicMock()),
        patch("classifier.predict._load_routing_map", return_value={}),
        patch("classifier.predict._MODEL", MagicMock()),
        patch("classifier.predict._ROUTING_MAP", {}),
    ):
        from api.main import app

        # Override app state dependency
        from api.dependencies import get_app_state
        app.dependency_overrides[get_app_state] = lambda: _MockAppState()

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    """GET /health should return 200 with classifier and db status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("ok", "degraded")
    assert "classifier_loaded" in body
    assert "db_connected" in body


def test_successful_completion(client: TestClient) -> None:
    """POST /v1/completions should return output_text and routing metadata."""
    mock_resp = _make_mock_llm_response()

    with (
        patch("api.routes.completions.classify_and_route",
              return_value=("simple", MagicMock(model_id="mock-model", quality_tier="low", provider="groq"))),
        patch("api.routes.completions.send_request", new_callable=AsyncMock, return_value=mock_resp),
        patch("api.routes.completions._background_logging_and_verification", new_callable=AsyncMock),
    ):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "What is 2 + 2?"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["output_text"] == "This is a mock answer."
    assert body["complexity_tier"] == "simple"
    assert body["fallback_used"] is False
    assert "request_id" in body
    assert resp.headers.get("X-Request-ID") == body["request_id"]


def test_force_model_override(client: TestClient) -> None:
    """force_model should bypass classify_and_route."""
    mock_resp = _make_mock_llm_response(model_id="llama-3.1-8b-instant", provider="groq")
    mock_model_config = MagicMock(model_id="llama-3.1-8b-instant", quality_tier="medium", provider="groq")

    with (
        patch("api.routes.completions.get_model", return_value=mock_model_config),
        patch("api.routes.completions.send_request", new_callable=AsyncMock, return_value=mock_resp),
        patch("api.routes.completions._background_logging_and_verification", new_callable=AsyncMock),
    ):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "Hello", "force_model": "llama-3.1-8b-instant"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model_used"] == "llama-3.1-8b-instant"
    assert body["complexity_tier"] == "forced"


def test_invalid_force_model(client: TestClient) -> None:
    """force_model with unknown model_id should return 422."""
    from models.registry import get_model as real_get_model
    with patch("api.routes.completions.get_model", side_effect=KeyError("no-such-model")):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "Hello", "force_model": "no-such-model"},
        )

    assert resp.status_code == 422
    assert "not found in registry" in resp.json()["detail"]


def test_provider_failure_triggers_fallback(client: TestClient) -> None:
    """When the primary model raises ProviderError, fallback to next tier."""
    from core.exceptions import ProviderError

    mock_fallback_resp = _make_mock_llm_response(model_id="fallback-model", provider="gemini")
    mock_fallback_config = MagicMock(model_id="fallback-model", quality_tier="high", provider="gemini")

    primary_config = MagicMock(model_id="cheap-model", quality_tier="low", provider="ollama")

    call_count = {"n": 0}

    async def side_effect(prompt, model_config, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ProviderError("ollama", ValueError("connection refused"))
        return mock_fallback_resp

    with (
        patch("api.routes.completions.classify_and_route",
              return_value=("simple", primary_config)),
        patch("api.routes.completions.list_models_by_tier", return_value=[mock_fallback_config]),
        patch("api.routes.completions.send_request", side_effect=side_effect),
        patch("api.routes.completions._background_logging_and_verification", new_callable=AsyncMock),
    ):
        resp = client.post("/v1/completions", json={"prompt": "Test prompt"})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["fallback_used"] is True
    assert body["model_used"] == "fallback-model"


def test_all_providers_fail_returns_502(client: TestClient) -> None:
    """When all providers fail, return 502 Bad Gateway."""
    from core.exceptions import ProviderError

    primary_config = MagicMock(model_id="cheap-model", quality_tier="low", provider="ollama")
    fallback_config = MagicMock(model_id="mid-model", quality_tier="medium", provider="groq")

    async def always_fail(prompt, model_config, **kwargs):
        raise ProviderError(model_config.provider, ConnectionError("timeout"))

    with (
        patch("api.routes.completions.classify_and_route",
              return_value=("simple", primary_config)),
        patch("api.routes.completions.list_models_by_tier", return_value=[fallback_config]),
        patch("api.routes.completions.send_request", side_effect=always_fail),
        patch("api.routes.completions._background_logging_and_verification", new_callable=AsyncMock),
    ):
        resp = client.post("/v1/completions", json={"prompt": "Will this fail?"})

    assert resp.status_code == 502


def test_routing_config_update_invalid_model(client: TestClient) -> None:
    """PUT /v1/routing-config with unknown model_id should return 422."""
    with patch("api.routes.routing_config.get_model", side_effect=KeyError("bad-model")):
        resp = client.put(
            "/v1/routing-config",
            json={"routing": {
                "simple": "bad-model",
                "moderate": "bad-model",
                "complex": "bad-model",
            }},
        )

    assert resp.status_code == 422
    body = resp.json()
    assert "not found in registry" in str(body["detail"])


def test_list_models(client: TestClient) -> None:
    """GET /v1/models should return a non-empty list."""
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] > 0
    assert isinstance(body["models"], list)
    first = body["models"][0]
    assert "model_id" in first
    assert "quality_tier" in first
