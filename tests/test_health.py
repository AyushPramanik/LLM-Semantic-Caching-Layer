"""Smoke tests for application bootstrap and configuration."""

from __future__ import annotations


def test_healthz_returns_ok(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readyz_reports_ready_when_store_reachable(client):
    resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_root_reports_service_metadata(client):
    resp = client.get("/")
    body = resp.json()
    assert resp.status_code == 200
    assert body["service"]
    assert "version" in body


def test_settings_reject_out_of_range_threshold():
    import pytest
    from pydantic import ValidationError

    from app.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(similarity_threshold=1.5)
