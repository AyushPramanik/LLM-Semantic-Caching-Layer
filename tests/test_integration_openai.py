"""End-to-end integration: full proxy stack with a mocked upstream provider.

Exercises the real router/provider path (not the echo stub) to confirm OpenAI
wire compatibility, that a MISS forwards upstream and a subsequent similar
request is served from cache without a second upstream call.
"""

from __future__ import annotations

import httpx
import respx
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def _routed_settings(**overrides) -> Settings:
    base = dict(
        app_env="test",
        embedding_provider="fake",
        vector_backend="memory",
        completer_backend="router",
        validation_sample_rate=0.0,
        rate_limit_enabled=False,
        openai_api_key="sk-test",
        openai_base_url="https://api.openai.com/v1",
        anthropic_base_url="https://api.anthropic.com/v1",
    )
    base.update(overrides)
    return Settings(**base)


def _openai_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-int",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": text},
                 "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        },
    )


def _payload(content: str, model: str = "gpt-4o-mini") -> dict:
    return {"model": model, "messages": [{"role": "user", "content": content}]}


@respx.mock
def test_miss_forwards_to_openai_then_hit_serves_from_cache():
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=_openai_response("HTTP is a protocol.")
    )
    app = create_app(_routed_settings())
    with TestClient(app) as client:
        first = client.post("/v1/chat/completions", json=_payload("What is HTTP?"))
        second = client.post("/v1/chat/completions", json=_payload("What is HTTP?"))

    assert first.headers["X-Cache-Status"] == "MISS"
    assert first.headers["X-Provider"] == "openai"
    assert first.json()["choices"][0]["message"]["content"] == "HTTP is a protocol."

    assert second.headers["X-Cache-Status"] == "HIT"
    # Upstream OpenAI was called exactly once despite two client requests.
    assert route.call_count == 1


@respx.mock
def test_anthropic_model_routes_to_anthropic_upstream():
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_int", "type": "message", "role": "assistant",
                "model": "claude-3-5-sonnet",
                "content": [{"type": "text", "text": "Bonjour."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 4, "output_tokens": 2},
            },
        )
    )
    app = create_app(_routed_settings(anthropic_api_key="sk-ant"))
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions", json=_payload("Say hi", model="claude-3-5-sonnet")
        )

    assert resp.headers["X-Provider"] == "anthropic"
    assert resp.json()["choices"][0]["message"]["content"] == "Bonjour."
    assert route.call_count == 1


@respx.mock
def test_upstream_error_surfaces_status():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(429, text="rate limited")
    )
    app = create_app(_routed_settings())
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json=_payload("boom"))
    # ProviderError(status_code=429) propagates as a 5xx/4xx, not a cached 200.
    assert resp.status_code >= 400
    assert resp.headers.get("X-Cache-Status") is None
