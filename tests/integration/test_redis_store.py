"""Integration tests for the Redis Stack vector store.

Marked ``integration`` and skipped automatically unless a Redis Stack instance
(with the RediSearch module) is reachable. Run with:

    REDIS_URL=redis://localhost:6379/15 pytest -m integration
"""

from __future__ import annotations

import os
import uuid

import numpy as np
import pytest

from app.cache.store import RedisVectorStore
from app.models.cache import CacheEntry

pytestmark = pytest.mark.integration

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/15")
DIM = 64


def _unit_vector(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _entry(namespace: str, vector: list[float], **overrides) -> CacheEntry:
    base = dict(
        namespace=namespace,
        prompt="q",
        embedding=vector,
        response={"id": "x", "object": "chat.completion", "model": "gpt-4o-mini",
                  "choices": [], "usage": {}},
        model="gpt-4o-mini",
        provider="openai",
    )
    base.update(overrides)
    return CacheEntry(**base)


@pytest.fixture
async def store():
    try:
        s = RedisVectorStore(
            redis_url=REDIS_URL,
            index_name=f"test_idx_{uuid.uuid4().hex[:8]}",
            dimensions=DIM,
        )
        await s.ensure_index()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Redis Stack not available: {exc}")
    yield s
    await s.clear()
    await s.aclose()


async def test_upsert_and_exact_search(store):
    vec = _unit_vector(1)
    entry = _entry("ns-a", vec)
    await store.upsert(entry)

    results = await store.search("ns-a", vec, top_k=1)
    assert results
    assert results[0].entry.id == entry.id
    assert results[0].score == pytest.approx(1.0, abs=1e-3)


async def test_namespace_isolation(store):
    vec = _unit_vector(2)
    await store.upsert(_entry("ns-a", vec))
    assert await store.search("ns-b", vec, top_k=1) == []


async def test_delete_by_model(store):
    await store.upsert(_entry("ns-a", _unit_vector(3), model="gpt-4o-mini"))
    await store.upsert(_entry("ns-a", _unit_vector(4), model="gpt-4o"))
    removed = await store.delete(model="gpt-4o-mini")
    assert removed == 1
    assert await store.count() == 1


async def test_clear(store):
    for i in range(3):
        await store.upsert(_entry("ns-a", _unit_vector(10 + i)))
    assert await store.count() == 3
    assert await store.clear() == 3
    assert await store.count() == 0
