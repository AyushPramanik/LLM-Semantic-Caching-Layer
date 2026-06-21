"""Tests for similarity-threshold analytics."""

from __future__ import annotations

import pytest

from app.analytics.threshold import ThresholdAnalyzer, ThresholdTestCase
from app.embeddings.fake import FakeEmbeddingService


@pytest.fixture
def analyzer():
    return ThresholdAnalyzer(
        embedding_service=FakeEmbeddingService(dimensions=128),
        thresholds=(0.90, 0.92, 0.95, 0.98),
    )


def _cases() -> list[ThresholdTestCase]:
    return [
        # Identical text -> similarity 1.0, should match.
        ThresholdTestCase(query="what is tcp", candidate="what is tcp", expected_match=True),
        ThresholdTestCase(query="define dns", candidate="define dns", expected_match=True),
        # Unrelated text -> low similarity, should not match.
        ThresholdTestCase(query="stock price today", candidate="poem about spring",
                          expected_match=False),
        ThresholdTestCase(query="weather in paris", candidate="how to sort a list",
                          expected_match=False),
    ]


async def test_report_covers_all_thresholds(analyzer):
    report = await analyzer.evaluate(_cases())
    assert report.sample_size == 4
    assert [m.threshold for m in report.metrics] == [0.90, 0.92, 0.95, 0.98]


async def test_metrics_are_consistent(analyzer):
    report = await analyzer.evaluate(_cases())
    for m in report.metrics:
        # With identical/unrelated pairs the fake embedder separates classes
        # cleanly, so precision and recall should be perfect at every threshold.
        assert m.true_positives == 2
        assert m.false_positives == 0
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)
        assert 0.0 <= m.false_positive_rate <= 1.0


async def test_recommended_threshold_is_a_candidate(analyzer):
    report = await analyzer.evaluate(_cases())
    assert report.recommended_threshold in {0.90, 0.92, 0.95, 0.98}


def test_threshold_test_endpoint(client):
    payload = {
        "cases": [
            {"query": "what is tcp", "candidate": "what is tcp", "expected_match": True},
            {"query": "weather", "candidate": "sorting algorithms", "expected_match": False},
        ],
        "thresholds": [0.90, 0.95],
    }
    resp = client.post("/analytics/threshold-test", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["sample_size"] == 2
    assert {m["threshold"] for m in body["metrics"]} == {0.90, 0.95}
