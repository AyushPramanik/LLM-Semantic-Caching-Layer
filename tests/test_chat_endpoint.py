"""Tests for the OpenAI-compatible chat completions endpoint."""

from __future__ import annotations


def _payload(content: str, model: str = "gpt-4o-mini") -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": content},
        ],
    }


def test_first_request_is_a_miss(client):
    resp = client.post("/v1/chat/completions", json=_payload("What is HTTP?"))
    assert resp.status_code == 200
    assert resp.headers["X-Cache-Status"] == "MISS"
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"


def test_repeated_request_is_a_hit(client):
    client.post("/v1/chat/completions", json=_payload("What is TCP?"))
    resp = client.post("/v1/chat/completions", json=_payload("What is TCP?"))
    assert resp.headers["X-Cache-Status"] == "HIT"
    assert float(resp.headers["X-Similarity-Score"]) >= 0.95
    assert resp.headers["X-Cache-Latency"].endswith("ms")


def test_response_schema_matches_openai(client):
    resp = client.post("/v1/chat/completions", json=_payload("ping"))
    body = resp.json()
    for field in ("id", "object", "created", "model", "choices", "usage"):
        assert field in body
    assert body["choices"][0]["finish_reason"] == "stop"


def test_different_model_is_isolated(client):
    client.post("/v1/chat/completions", json=_payload("same text", model="gpt-4o-mini"))
    resp = client.post("/v1/chat/completions", json=_payload("same text", model="gpt-4o"))
    # Different model -> different namespace -> still a miss.
    assert resp.headers["X-Cache-Status"] == "MISS"


def test_tenant_header_isolates_cache(client):
    body = _payload("tenant scoped prompt")
    client.post("/v1/chat/completions", json=body, headers={"X-Tenant-Id": "team-a"})
    resp = client.post("/v1/chat/completions", json=body, headers={"X-Tenant-Id": "team-b"})
    assert resp.headers["X-Cache-Status"] == "MISS"
