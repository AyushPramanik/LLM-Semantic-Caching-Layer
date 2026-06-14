"""Unit tests for embedding services."""

from __future__ import annotations

import httpx
import numpy as np
import pytest
import respx

from app.embeddings import build_embedding_service
from app.embeddings.fake import FakeEmbeddingService
from app.embeddings.openai import OpenAIEmbeddingService


@pytest.fixture
def fake_service():
    return FakeEmbeddingService(dimensions=64)


async def test_fake_embedding_is_deterministic(fake_service):
    a = await fake_service.embed("What is the capital of France?")
    b = await fake_service.embed("what is the capital of france?  ")
    # Same normalized text -> identical vector.
    assert a.vector == b.vector


async def test_fake_embedding_is_unit_norm(fake_service):
    result = await fake_service.embed("hello world")
    norm = float(np.linalg.norm(result.as_numpy()))
    assert result.dimensions == 64
    assert norm == pytest.approx(1.0, abs=1e-5)


async def test_distinct_prompts_have_distinct_vectors(fake_service):
    a = await fake_service.embed("How do I sort a list in Python?")
    b = await fake_service.embed("What is the boiling point of water?")
    assert a.vector != b.vector


async def test_embed_batch_matches_single(fake_service):
    texts = ["alpha", "beta", "gamma"]
    batch = await fake_service.embed_batch(texts)
    singles = [await fake_service.embed(t) for t in texts]
    assert [r.vector for r in batch] == [r.vector for r in singles]


def test_factory_selects_fake_for_test_settings(settings):
    service = build_embedding_service(settings)
    assert isinstance(service, FakeEmbeddingService)


@respx.mock
async def test_openai_embedding_parses_and_normalizes():
    route = respx.post("https://api.openai.com/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [{"index": 0, "embedding": [3.0, 4.0]}],
                "usage": {"total_tokens": 8},
            },
        )
    )
    service = OpenAIEmbeddingService(
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        model="text-embedding-3-small",
        dimensions=2,
    )
    result = await service.embed("hi")
    await service.aclose()

    assert route.called
    # [3,4] normalized -> [0.6, 0.8]
    assert result.vector == pytest.approx([0.6, 0.8])
    assert result.tokens == 8


@respx.mock
async def test_openai_embedding_retries_then_succeeds():
    responses = [httpx.Response(503), httpx.Response(
        200,
        json={"data": [{"index": 0, "embedding": [1.0, 0.0]}], "usage": {"total_tokens": 2}},
    )]
    respx.post("https://api.openai.com/v1/embeddings").mock(side_effect=responses)

    service = OpenAIEmbeddingService(
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        model="text-embedding-3-small",
        dimensions=2,
    )
    result = await service.embed("retry please")
    await service.aclose()
    assert result.vector == pytest.approx([1.0, 0.0])
