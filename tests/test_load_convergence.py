"""Deterministic load-style tests: warmup, hit-rate convergence, cost savings.

These run in-process against the in-memory cache + echo provider so they are
fast and hermetic, while still exercising the same code paths a Locust run hits.
They model the headline claims (cache warms up, hit rate converges, money is
saved) as assertions rather than prose.
"""

from __future__ import annotations

import random

import pytest
from prometheus_client import CollectorRegistry

from app.cache.memory_store import InMemoryVectorStore
from app.cache.semantic_cache import CacheStatus, SemanticCache
from app.embeddings.fake import FakeEmbeddingService
from app.models.chat import ChatCompletionRequest, ChatMessage
from app.monitoring.metrics import CacheMetrics
from app.proxy.echo import EchoCompleter
from app.proxy.service import ProxyService

pytestmark = pytest.mark.load

BASE_PROMPTS = [f"Explain concept number {i} in detail." for i in range(15)]


def _build_proxy():
    metrics = CacheMetrics(registry=CollectorRegistry())
    cache = SemanticCache(
        embedding_service=FakeEmbeddingService(dimensions=128),
        store=InMemoryVectorStore(),
        threshold=0.95,
    )
    proxy = ProxyService(
        cache=cache, completer=EchoCompleter(), default_ttl_seconds=3600, metrics=metrics
    )
    return proxy, metrics


def _request(prompt: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="gpt-4o-mini",
        messages=[
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content=prompt),
        ],
    )


async def _run_workload(proxy, n: int, *, unique_ratio: float, seed: int = 7):
    rng = random.Random(seed)
    statuses: list[CacheStatus] = []
    for _ in range(n):
        if rng.random() < unique_ratio:
            prompt = f"Unique one-off question {rng.randint(1, 10_000_000)}"
        else:
            prompt = rng.choice(BASE_PROMPTS)
        result = await proxy.complete(_request(prompt))
        statuses.append(result.cache_status)
    return statuses


async def test_hit_rate_converges_after_warmup():
    proxy, _ = _build_proxy()
    statuses = await _run_workload(proxy, 2000, unique_ratio=0.25)

    def hit_rate(window):
        hits = sum(1 for s in window if s is CacheStatus.HIT)
        return hits / len(window)

    early = hit_rate(statuses[:200])
    late = hit_rate(statuses[-200:])

    # Cache warms up: late hit rate is meaningfully higher than the cold start.
    assert late > early
    # With ~75% repeated traffic over 15 prompts, steady-state should be high.
    assert late > 0.6


async def test_cost_and_tokens_saved_are_positive():
    proxy, metrics = _build_proxy()
    await _run_workload(proxy, 1000, unique_ratio=0.2)
    assert metrics.tokens_saved._value.get() > 0
    assert metrics.cost_saved._value.get() > 0


async def test_all_unique_traffic_never_converges():
    proxy, _ = _build_proxy()
    statuses = await _run_workload(proxy, 300, unique_ratio=1.0)
    assert all(s is CacheStatus.MISS for s in statuses)
