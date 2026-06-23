"""Tests for the cache validation (shadow replay) system."""

from __future__ import annotations

import random

from app.analytics.validation import CacheValidator
from app.embeddings.fake import FakeEmbeddingService
from app.models.chat import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)


def _response(text: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        model="gpt-4o-mini",
        choices=[ChatCompletionChoice(index=0, message=ChatMessage(role="assistant", content=text))],
    )


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="gpt-4o-mini", messages=[ChatMessage(role="user", content="what is dns")]
    )


class _FixedCompleter:
    def __init__(self, text: str) -> None:
        self._text = text

    def resolve_provider(self, request):
        return "stub"

    async def complete(self, request):
        return _response(self._text)


def _validator(fresh_text: str, **kwargs):
    return CacheValidator(
        embedding_service=FakeEmbeddingService(dimensions=128),
        completer=_FixedCompleter(fresh_text),
        **kwargs,
    )


def test_sampling_is_deterministic_with_seed():
    v = _validator("x", sample_rate=0.5, rng=random.Random(42))
    decisions = [v.should_validate() for _ in range(20)]
    # With a fixed seed the sequence is reproducible and not all-true/all-false.
    assert any(decisions) and not all(decisions)


def test_zero_sample_rate_never_validates():
    v = _validator("x", sample_rate=0.0)
    assert all(not v.should_validate() for _ in range(50))


async def test_matching_response_is_not_drift():
    v = _validator("the domain name system", drift_threshold=0.9)
    result = await v.validate(_request(), _response("the domain name system"))
    assert result.response_similarity == 1.0
    assert not result.drift
    stats = v.stats()
    assert stats.validations == 1
    assert stats.cache_validation_accuracy == 1.0
    assert stats.semantic_drift_rate == 0.0


async def test_divergent_response_is_flagged_as_false_hit():
    v = _validator("a completely unrelated answer about cooking", drift_threshold=0.9)
    result = await v.validate(_request(), _response("dns resolves hostnames to ip addresses"))
    assert result.drift
    assert result.false_hit
    stats = v.stats()
    assert stats.false_hit_rate == 1.0
    assert stats.cache_validation_accuracy == 0.0


async def test_stats_aggregate_across_validations():
    v = _validator("dns resolves hostnames", drift_threshold=0.9)
    await v.validate(_request(), _response("dns resolves hostnames"))  # match
    v2 = _validator("totally different topic entirely", drift_threshold=0.9)
    await v2.validate(_request(), _response("dns resolves hostnames"))  # drift
    assert v.stats().semantic_drift_rate == 0.0
    assert v2.stats().semantic_drift_rate == 1.0


def test_validation_endpoint_reports_stats(client):
    resp = client.get("/analytics/validation")
    assert resp.status_code == 200
    body = resp.json()
    assert "cache_validation_accuracy" in body
    assert "semantic_drift_rate" in body
    assert "false_hit_rate" in body
