"""
Full pytest test suite for the MARS API.

All tests mock the Anthropic client so no real API calls are made.
Run with:  pytest --cov=mars_api --cov-report=term-missing
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


VALID_KEY = "test-mars-key"
NWU_TOKEN = "test-nwu-token"


def _make_mock_response(text: str = "mock reasoning output", stop_reason: str = "end_turn") -> MagicMock:
    content_block = MagicMock()
    content_block.text = text
    mock_response = MagicMock()
    mock_response.content = [content_block]
    mock_response.stop_reason = stop_reason
    return mock_response


@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    monkeypatch.setenv("MARS_API_KEY", VALID_KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("INTERNAL_AGENT_TOKEN", NWU_TOKEN)
    monkeypatch.setenv("MARS_THRESHOLD", "0.50")
    monkeypatch.setenv("MARS_DEFAULT_ROUTE", "auto")


@pytest.fixture
def client():
    from mars_api import app
    return TestClient(app)


@pytest.fixture
def mock_anthropic():
    with patch("mars_api.anthropic.Anthropic") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["service"] == "MARS"


class TestStatus:
    def test_status_returns_200(self, client):
        assert client.get("/api/status").status_code == 200

    def test_status_fields(self, client):
        data = client.get("/api/status").json()
        assert data["service"] == "MARS"
        assert data["version"] == "1.1.0"
        assert "uptime_seconds" in data
        assert "timestamp" in data
        assert "model" in data
        assert "routing" in data
        assert data["routing"]["score_version"]


class TestReason:
    def test_reason_success(self, client, mock_anthropic):
        mock_anthropic.messages.create.return_value = _make_mock_response(
            "Deep thought.\nconfidence: 0.92"
        )
        resp = client.post(
            "/api/reason",
            json={"query": "What is consciousness?"},
            headers={"x-api-key": VALID_KEY},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["reasoning"] == "Deep thought."
        assert data["route"]["path"] == "standard"

    def test_reason_escalates_when_unscored(self, client, mock_anthropic):
        mock_anthropic.messages.create.side_effect = [
            _make_mock_response("cheap pass with no score"),
            _make_mock_response("deep answer\nconfidence: 0.88"),
        ]
        resp = client.post(
            "/api/reason",
            json={"query": "hard question", "route": "auto"},
            headers={"x-api-key": VALID_KEY},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["route"]["path"] == "deep"
        assert data["route"]["escalated"] is True
        assert data["reasoning"] == "deep answer"

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


class TestRoutingStats:
    def test_stats_requires_key(self, client):
        assert client.get("/api/routing/stats").status_code == 401

    def test_stats_ok(self, client):
        resp = client.get("/api/routing/stats", headers={"x-api-key": VALID_KEY})
        assert resp.status_code == 200
        assert "requests" in resp.json()


class TestMetacognize:
    def test_metacognize_success(self, client, mock_anthropic):
        mock_anthropic.messages.create.return_value = _make_mock_response("Better reasoning.")
        resp = client.post(
            "/api/metacognize",
            json={"reasoning": "The sky is blue because reasons."},
            headers={"x-api-key": VALID_KEY},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_metacognize_missing_key_returns_401(self, client):
        resp = client.post("/api/metacognize", json={"reasoning": "test"})
        assert resp.status_code == 401


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
        assert data["iterations_completed"] == 1
        assert data["final_output"] == "Optimized output."

    def test_optimize_clamps_iterations_above_5(self, client, mock_anthropic):
        mock_anthropic.messages.create.return_value = _make_mock_response()
        resp = client.post(
            "/api/optimize",
            json={"task": "test", "iterations": 99},
            headers={"x-api-key": VALID_KEY},
        )
        assert resp.json()["iterations_completed"] == 5

    def test_optimize_missing_key_returns_401(self, client):
        resp = client.post("/api/optimize", json={"task": "test"})
        assert resp.status_code == 401


class TestNWUListener:
    def test_nwu_opportunity_acknowledged(self, client):
        resp = client.post(
            "/nwu-listener",
            json={
                "nwu_version": "1.0",
                "event_type": "opportunity_detected",
                "source": "scout",
                "target": "mars",
                "auth_token": NWU_TOKEN,
                "payload": {"id": "opp-1", "description": "test deal"},
                "timestamp": "2026-09-16T00:00:00Z",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "acknowledged"
        assert data["opportunity_id"] == "opp-1"

    def test_nwu_bad_token_returns_401(self, client):
        resp = client.post(
            "/nwu-listener",
            json={
                "nwu_version": "1.0",
                "event_type": "opportunity_detected",
                "source": "scout",
                "target": "mars",
                "auth_token": "wrong",
                "payload": {},
                "timestamp": "2026-09-16T00:00:00Z",
            },
        )
        assert resp.status_code == 401
