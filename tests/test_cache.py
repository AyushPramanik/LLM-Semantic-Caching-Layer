"""Tests for namespacing, similarity search, and the cache lookup workflow."""

from __future__ import annotations

import numpy as np
import pytest

from app.cache.memory_store import InMemoryVectorStore
from app.cache.namespace import RequestSignature, hash_system_prompt
from app.cache.semantic_cache import CacheStatus, SemanticCache
from app.cache.similarity import cosine_similarity, top_k_indices
from app.embeddings.fake import FakeEmbeddingService


@pytest.fixture
def cache():
    return SemanticCache(
        embedding_service=FakeEmbeddingService(dimensions=128),
        store=InMemoryVectorStore(),
        threshold=0.95,
        near_miss_window=0.05,
    )


def sig(**overrides) -> RequestSignature:
    base = dict(model="gpt-4o-mini", provider="openai", system_prompt="You are helpful.")
    base.update(overrides)
    return RequestSignature(**base)


# --- namespace safety ---------------------------------------------------------

def test_namespace_is_deterministic():
    assert sig().namespace() == sig().namespace()


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": "gpt-4o"},
        {"provider": "anthropic"},
        {"system_prompt": "Different system."},
        {"temperature": 0.2},
        {"max_tokens": 256},
        {"tenant": "team-b"},
    ],
)
def test_namespace_changes_with_safety_dimensions(overrides):
    assert sig().namespace() != sig(**overrides).namespace()


def test_temperature_micro_differences_share_namespace():
    # 0.70 and 0.74 bucket to the same one-decimal temperature.
    assert sig(temperature=0.70).namespace() == sig(temperature=0.74).namespace()


def test_empty_system_prompt_hash_is_stable():
    assert hash_system_prompt("") == hash_system_prompt(None) == "no-system"


# --- similarity ---------------------------------------------------------------

def test_cosine_identity_and_orthogonal():
    assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_top_k_indices_orders_descending():
    scores = np.array([0.1, 0.9, 0.5, 0.99], dtype=np.float32)
    assert top_k_indices(scores, 2) == [3, 1]


# --- lookup workflow ----------------------------------------------------------

async def test_exact_repeat_is_a_hit(cache):
    s = sig()
    await cache.store_response(s, "What is the capital of France?", {"id": "x"}, ttl_seconds=3600)
    result = await cache.lookup(s, "What is the capital of France?")
    assert result.status is CacheStatus.HIT
    assert result.score == pytest.approx(1.0, abs=1e-4)
    assert result.match is not None


async def test_unrelated_prompt_is_a_miss(cache):
    s = sig()
    await cache.store_response(s, "What is the capital of France?", {"id": "x"}, ttl_seconds=3600)
    result = await cache.lookup(s, "Write a poem about the ocean in iambic pentameter")
    assert result.status is CacheStatus.MISS


async def test_hit_increments_hit_count(cache):
    s = sig()
    entry = await cache.store_response(s, "2 + 2 = ?", {"id": "x"}, ttl_seconds=3600)
    await cache.lookup(s, "2 + 2 = ?")
    stored = await cache._store.get(entry.id)
    assert stored.hit_count == 1


async def test_different_namespace_does_not_match(cache):
    await cache.store_response(sig(), "Shared prompt text", {"id": "a"}, ttl_seconds=3600)
    # Same prompt, different model -> different namespace -> miss.
    result = await cache.lookup(sig(model="gpt-4o"), "Shared prompt text")
    assert result.status is CacheStatus.MISS


async def test_lower_threshold_can_turn_miss_into_hit(cache):
    s = sig()
    await cache.store_response(s, "Explain recursion simply", {"id": "x"}, ttl_seconds=3600)
    # Force a hit on any non-empty match by dropping the threshold.
    result = await cache.lookup(s, "Explain recursion simply please", threshold=-1.0)
    assert result.status is CacheStatus.HIT


async def test_store_response_records_tokens_and_tags(cache):
    s = sig()
    entry = await cache.store_response(
        s, "tag me", {"id": "x"}, ttl_seconds=3600, tags=["factual"], completion_tokens=42
    )
    assert entry.tags == ["factual"]
    assert entry.completion_tokens == 42
