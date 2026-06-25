"""Tests for streaming responses and the never-cache-partial guarantee."""

from __future__ import annotations

import pytest

from app.cache.memory_store import InMemoryVectorStore
from app.cache.semantic_cache import CacheStatus, SemanticCache
from app.embeddings.fake import FakeEmbeddingService
from app.models.chat import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionUsage,
)
from app.proxy.service import ProxyService
from app.proxy.streaming import StreamAssembler, response_to_sse


def _full_response(text: str = "hello world") -> ChatCompletionResponse:
    return ChatCompletionResponse(
        model="gpt-4o-mini",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=2, completion_tokens=2, total_tokens=4),
    )


def _request(content: str, stream: bool = True) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="gpt-4o-mini",
        stream=stream,
        messages=[
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content=content),
        ],
    )


class _StreamingCompleter:
    """A completer that emits SSE chunks, optionally failing mid-stream."""

    name = "stub"

    def __init__(self, text: str, *, fail_after: int | None = None) -> None:
        self._text = text
        self._fail_after = fail_after
        self.calls = 0

    def resolve_provider(self, request):
        return self.name

    async def complete(self, request):  # pragma: no cover - unused
        raise NotImplementedError

    async def stream(self, request):
        self.calls += 1
        for chunk in response_to_sse(_full_response(self._text)):
            yield chunk
            if self._fail_after is not None:
                self._fail_after -= 1
                if self._fail_after < 0:
                    raise RuntimeError("upstream exploded mid-stream")


def _build_proxy(completer):
    cache = SemanticCache(
        embedding_service=FakeEmbeddingService(dimensions=64),
        store=InMemoryVectorStore(),
        threshold=0.95,
    )
    return ProxyService(cache=cache, completer=completer, default_ttl_seconds=3600), cache


# --- assembler ----------------------------------------------------------------

def test_assembler_reconstructs_full_response():
    assembler = StreamAssembler("gpt-4o-mini")
    for chunk in response_to_sse(_full_response("streamed text")):
        assembler.push(chunk)
    assert assembler.completed
    response = assembler.build_response()
    assert response.first_text() == "streamed text"


def test_assembler_handles_split_chunk_boundaries():
    raw = b"".join(response_to_sse(_full_response("abc")))
    assembler = StreamAssembler("gpt-4o-mini")
    # Feed one byte at a time to stress buffer reassembly.
    for i in range(len(raw)):
        assembler.push(raw[i : i + 1])
    assert assembler.completed
    assert assembler.build_response().first_text() == "abc"


# --- proxy streaming ----------------------------------------------------------

async def test_stream_miss_then_caches_full_response():
    completer = _StreamingCompleter("the ocean is vast")
    proxy, cache = _build_proxy(completer)

    result = await proxy.stream(_request("describe the ocean"))
    assert result.cache_status is CacheStatus.MISS
    chunks = [chunk async for chunk in result.body]
    assert b"data: [DONE]" in b"".join(chunks)

    # A subsequent identical request should now be a HIT served from cache.
    hit = await proxy.stream(_request("describe the ocean"))
    assert hit.cache_status is CacheStatus.HIT
    _ = [chunk async for chunk in hit.body]
    assert completer.calls == 1  # upstream only hit once


async def test_failed_stream_is_never_cached():
    completer = _StreamingCompleter("partial", fail_after=1)
    proxy, cache = _build_proxy(completer)

    result = await proxy.stream(_request("will fail"))
    with pytest.raises(RuntimeError):
        _ = [chunk async for chunk in result.body]

    # Nothing should have been written to the cache.
    assert await cache._store.count() == 0


async def test_stream_hit_does_not_call_upstream_again():
    completer = _StreamingCompleter("cached answer")
    proxy, _ = _build_proxy(completer)
    r1 = await proxy.stream(_request("repeat me"))
    _ = [c async for c in r1.body]
    r2 = await proxy.stream(_request("repeat me"))
    _ = [c async for c in r2.body]
    assert r2.cache_status is CacheStatus.HIT
    assert completer.calls == 1


def test_streaming_endpoint_emits_sse(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "stream": True,
            "messages": [{"role": "user", "content": "stream this"}],
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "data:" in resp.text
    assert "[DONE]" in resp.text
