"""Tests for production hardening middleware."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def _settings(**overrides) -> Settings:
    base = dict(
        app_env="test",
        embedding_provider="fake",
        vector_backend="memory",
        completer_backend="echo",
        validation_sample_rate=0.0,
    )
    base.update(overrides)
    return Settings(**base)


def test_correlation_id_is_generated_and_echoed(client):
    resp = client.get("/healthz")
    assert "X-Correlation-ID" in resp.headers
    assert len(resp.headers["X-Correlation-ID"]) > 0


def test_inbound_correlation_id_is_preserved(client):
    resp = client.get("/healthz", headers={"X-Request-ID": "trace-abc-123"})
    assert resp.headers["X-Correlation-ID"] == "trace-abc-123"


def test_rate_limit_returns_429_when_exhausted():
    app = create_app(_settings(rate_limit_enabled=True, rate_limit_rps=3))
    with TestClient(app) as client:
        payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
        statuses = [
            client.post("/v1/chat/completions", json=payload).status_code for _ in range(8)
        ]
    assert 429 in statuses
    assert statuses.count(200) >= 3


def test_health_endpoints_are_exempt_from_rate_limit():
    app = create_app(_settings(rate_limit_enabled=True, rate_limit_rps=1))
    with TestClient(app) as client:
        statuses = [client.get("/healthz").status_code for _ in range(10)]
    assert all(s == 200 for s in statuses)
