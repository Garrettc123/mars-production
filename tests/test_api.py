"""
Full pytest test suite for the MARS API.

All tests mock the Anthropic client so no real API calls are made.
Run with:  pytest --cov=mars_api --cov-report=term-missing
"""
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------

VALID_KEY = "test-mars-key"


def _make_mock_response(text: str = "mock reasoning output") -> MagicMock:
    """Build a minimal Anthropic Messages API response mock."""
    content_block = MagicMock()
    content_block.text = text
    mock_response = MagicMock()
    mock_response.content = [content_block]
    return mock_response


@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    """Ensure required env vars are set for every test."""
    monkeypatch.setenv("MARS_API_KEY", VALID_KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")


@pytest.fixture
def client():
    """Return a TestClient with a fresh import of the app."""
    # Import here so env vars are already set
    from mars_api import app
    return TestClient(app)


@pytest.fixture
def mock_anthropic():
    """Patch anthropic.Anthropic so no real HTTP calls are made."""
    with patch("mars_api.anthropic.Anthropic") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "MARS"


# ---------------------------------------------------------------------------
# /api/status
# ---------------------------------------------------------------------------

class TestStatus:
    def test_status_returns_200(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200

    def test_status_fields(self, client):
        resp = client.get("/api/status")
        data = resp.json()
        assert data["service"] == "MARS"
        assert data["version"] == "1.0.0"
        assert "uptime_seconds" in data
        assert "timestamp" in data
        assert "model" in data

    def test_status_uptime_is_non_negative(self, client):
        resp = client.get("/api/status")
        assert resp.json()["uptime_seconds"] >= 0


# ---------------------------------------------------------------------------
# /api/reason
# ---------------------------------------------------------------------------

class TestReason:
    def test_reason_success(self, client, mock_anthropic):
        mock_anthropic.messages.create.return_value = _make_mock_response("Deep thought.")
        resp = client.post(
            "/api/reason",
            json={"query": "What is consciousness?"},
            headers={"x-api-key": VALID_KEY},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["reasoning"] == "Deep thought."
        assert data["model"] == "claude-3-5-sonnet-20241022"

    def test_reason_missing_key_returns_401(self, client):
        resp = client.post("/api/reason", json={"query": "test"})
        assert resp.status_code == 401

    def test_reason_wrong_key_returns_401(self, client):
        resp = client.post(
            "/api/reason",
            json={"query": "test"},
            headers={"x-api-key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_reason_missing_query_returns_422(self, client):
        resp = client.post(
            "/api/reason",
            json={},
            headers={"x-api-key": VALID_KEY},
        )
        assert resp.status_code == 422

    def test_reason_calls_anthropic_with_correct_model(self, client, mock_anthropic):
        mock_anthropic.messages.create.return_value = _make_mock_response()
        client.post(
            "/api/reason",
            json={"query": "Hello"},
            headers={"x-api-key": VALID_KEY},
        )
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-3-5-sonnet-20241022"

    def test_reason_custom_max_tokens(self, client, mock_anthropic):
        mock_anthropic.messages.create.return_value = _make_mock_response()
        client.post(
            "/api/reason",
            json={"query": "Hello", "max_tokens": 512},
            headers={"x-api-key": VALID_KEY},
        )
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 512


# ---------------------------------------------------------------------------
# /api/metacognize
# ---------------------------------------------------------------------------

class TestMetacognize:
    def test_metacognize_success(self, client, mock_anthropic):
        mock_anthropic.messages.create.return_value = _make_mock_response("Better reasoning.")
        resp = client.post(
            "/api/metacognize",
            json={"reasoning": "The sky is blue because reasons."},
            headers={"x-api-key": VALID_KEY},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["reflection"] == "Better reasoning."

    def test_metacognize_with_context(self, client, mock_anthropic):
        mock_anthropic.messages.create.return_value = _make_mock_response("Contextual reflection.")
        resp = client.post(
            "/api/metacognize",
            json={"reasoning": "foo", "context": "additional context here"},
            headers={"x-api-key": VALID_KEY},
        )
        assert resp.status_code == 200
        # Verify the context was included in the prompt sent to the model
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        prompt = call_kwargs["messages"][0]["content"]
        assert "additional context here" in prompt

    def test_metacognize_missing_key_returns_401(self, client):
        resp = client.post("/api/metacognize", json={"reasoning": "test"})
        assert resp.status_code == 401

    def test_metacognize_missing_reasoning_returns_422(self, client):
        resp = client.post(
            "/api/metacognize",
            json={},
            headers={"x-api-key": VALID_KEY},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /api/optimize
# ---------------------------------------------------------------------------

class TestOptimize:
    def test_optimize_success_single_iteration(self, client, mock_anthropic):
        mock_anthropic.messages.create.return_value = _make_mock_response("Optimized output.")
        resp = client.post(
            "/api/optimize",
            json={"task": "Write a haiku about recursion", "iterations": 1},
            headers={"x-api-key": VALID_KEY},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["iterations_completed"] == 1
        assert data["final_output"] == "Optimized output."
        assert len(data["history"]) == 1

    def test_optimize_multiple_iterations(self, client, mock_anthropic):
        outputs = ["v1", "v2", "v3"]
        mock_anthropic.messages.create.side_effect = [
            _make_mock_response(t) for t in outputs
        ]
        resp = client.post(
            "/api/optimize",
            json={"task": "Explain entropy", "iterations": 3},
            headers={"x-api-key": VALID_KEY},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["iterations_completed"] == 3
        assert data["final_output"] == "v3"
        assert [h["output"] for h in data["history"]] == outputs

    def test_optimize_clamps_iterations_above_5(self, client, mock_anthropic):
        mock_anthropic.messages.create.return_value = _make_mock_response()
        resp = client.post(
            "/api/optimize",
            json={"task": "test", "iterations": 99},
            headers={"x-api-key": VALID_KEY},
        )
        assert resp.status_code == 200
        assert resp.json()["iterations_completed"] == 5

    def test_optimize_clamps_iterations_below_1(self, client, mock_anthropic):
        mock_anthropic.messages.create.return_value = _make_mock_response()
        resp = client.post(
            "/api/optimize",
            json={"task": "test", "iterations": 0},
            headers={"x-api-key": VALID_KEY},
        )
        assert resp.status_code == 200
        assert resp.json()["iterations_completed"] == 1

    def test_optimize_missing_key_returns_401(self, client):
        resp = client.post("/api/optimize", json={"task": "test"})
        assert resp.status_code == 401

    def test_optimize_missing_task_returns_422(self, client):
        resp = client.post(
            "/api/optimize",
            json={},
            headers={"x-api-key": VALID_KEY},
        )
        assert resp.status_code == 422

    def test_optimize_history_has_iteration_numbers(self, client, mock_anthropic):
        mock_anthropic.messages.create.side_effect = [
            _make_mock_response(f"iter{i}") for i in range(1, 4)
        ]
        resp = client.post(
            "/api/optimize",
            json={"task": "test", "iterations": 3},
            headers={"x-api-key": VALID_KEY},
        )
        history = resp.json()["history"]
        assert [h["iteration"] for h in history] == [1, 2, 3]
