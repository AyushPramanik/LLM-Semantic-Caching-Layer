"""Unit tests for provider routing and adapter translation."""

from __future__ import annotations

import httpx
import pytest
import respx

from app.models.chat import ChatCompletionRequest, ChatMessage
from app.providers.anthropic import AnthropicProvider
from app.providers.base import ProviderError
from app.providers.openai import OpenAIProvider
from app.providers.router import ProviderRouter


def _request(model: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        messages=[ChatMessage(role="user", content="hello")],
    )


class _StubProvider:
    def __init__(self, name: str) -> None:
        self.name = name

    async def complete(self, request):  # pragma: no cover - not called here
        raise NotImplementedError


@pytest.fixture
def router():
    return ProviderRouter(
        {
            "openai": _StubProvider("openai"),
            "anthropic": _StubProvider("anthropic"),
            "ollama": _StubProvider("ollama"),
        }
    )


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-4o-mini", "openai"),
        ("gpt-3.5-turbo", "openai"),
        ("o1-preview", "openai"),
        ("claude-3-5-sonnet", "anthropic"),
        ("claude-opus-4", "anthropic"),
        ("llama3.1", "ollama"),
        ("mistral-7b", "ollama"),
        ("qwen2.5", "ollama"),
    ],
)
def test_router_selects_provider_by_model(router, model, expected):
    assert router.resolve_provider(_request(model)) == expected


def test_router_falls_back_to_default_for_unknown(router):
    assert router.resolve_provider(_request("some-unknown-model")) == "openai"


def test_router_raises_when_provider_missing():
    router = ProviderRouter({"openai": _StubProvider("openai")})
    with pytest.raises(ProviderError):
        router._provider_for(_request("claude-3-5-sonnet"))


@respx.mock
async def test_openai_provider_passthrough():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hi"},
                     "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )
    provider = OpenAIProvider(api_key="sk-test", base_url="https://api.openai.com/v1")
    resp = await provider.complete(_request("gpt-4o-mini"))
    await provider.aclose()
    assert resp.first_text() == "hi"


@respx.mock
async def test_openai_provider_maps_error():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(401, text="unauthorized")
    )
    provider = OpenAIProvider(api_key="bad", base_url="https://api.openai.com/v1")
    with pytest.raises(ProviderError) as exc:
        await provider.complete(_request("gpt-4o-mini"))
    await provider.aclose()
    assert exc.value.status_code == 401


@respx.mock
async def test_anthropic_translation_roundtrip():
    captured = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet",
                "content": [{"type": "text", "text": "Bonjour"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 3},
            },
        )

    respx.post("https://api.anthropic.com/v1/messages").mock(side_effect=_capture)

    provider = AnthropicProvider(api_key="sk-ant", base_url="https://api.anthropic.com/v1")
    req = ChatCompletionRequest(
        model="claude-3-5-sonnet",
        messages=[
            ChatMessage(role="system", content="You are French."),
            ChatMessage(role="user", content="Hello"),
        ],
        max_tokens=128,
    )
    resp = await provider.complete(req)
    await provider.aclose()

    # System lifted to top-level; only user/assistant turns in messages.
    assert captured["system"] == "You are French."
    assert captured["max_tokens"] == 128
    assert all(m["role"] in ("user", "assistant") for m in captured["messages"])
    # Anthropic response normalized to OpenAI schema.
    assert resp.first_text() == "Bonjour"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage.total_tokens == 13
