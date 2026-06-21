"""Tests for the adaptive threshold engine."""

from __future__ import annotations

import pytest

from app.policies.threshold_engine import (
    AdaptiveThresholdEngine,
    ThresholdCategory,
    build_threshold_policy,
)


@pytest.fixture
def engine():
    return AdaptiveThresholdEngine()


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Write a poem about autumn leaves", ThresholdCategory.CREATIVE),
        ("Imagine a world without gravity", ThresholdCategory.CREATIVE),
        ("Debug this Python function that throws an exception", ThresholdCategory.PROGRAMMING),
        ("Write a regex to match emails", ThresholdCategory.PROGRAMMING),
        ("Classify the sentiment of this review", ThresholdCategory.CLASSIFICATION),
        ("Is this true or false: the earth is flat", ThresholdCategory.CLASSIFICATION),
        ("Tell me about the Roman Empire", ThresholdCategory.GENERAL),
    ],
)
def test_categorization(engine, prompt, expected):
    assert engine.categorize(prompt) == expected


def test_category_thresholds_follow_expected_ordering(engine):
    classification = engine.threshold_for_prompt("classify this text")
    programming = engine.threshold_for_prompt("implement quicksort in python")
    creative = engine.threshold_for_prompt("write a story about a dragon")
    assert classification < programming < creative


def test_false_hit_feedback_tightens_threshold(engine):
    before = engine.snapshot()[ThresholdCategory.CLASSIFICATION.value]
    after = engine.record_feedback(ThresholdCategory.CLASSIFICATION, false_hit=True)
    assert after > before


def test_missed_feedback_loosens_threshold(engine):
    before = engine.snapshot()[ThresholdCategory.PROGRAMMING.value]
    after = engine.record_feedback(ThresholdCategory.PROGRAMMING, false_hit=False)
    assert after < before


def test_threshold_stays_within_bounds(engine):
    for _ in range(100):
        engine.record_feedback(ThresholdCategory.CREATIVE, false_hit=True)
    assert engine.snapshot()[ThresholdCategory.CREATIVE.value] <= 0.995
    for _ in range(100):
        engine.record_feedback(ThresholdCategory.CREATIVE, false_hit=False)
    assert engine.snapshot()[ThresholdCategory.CREATIVE.value] >= 0.97


def test_policy_callable_uses_prompt(engine):
    from app.models.chat import ChatCompletionRequest, ChatMessage

    policy = build_threshold_policy(engine)
    request = ChatCompletionRequest(
        model="gpt-4o-mini",
        messages=[ChatMessage(role="user", content="write a haiku about the sea")],
    )
    assert policy(request) == pytest.approx(0.985)


def test_threshold_endpoints(client):
    snap = client.get("/analytics/thresholds")
    assert snap.status_code == 200
    assert "programming" in snap.json()

    resp = client.post(
        "/analytics/threshold-feedback",
        json={"category": "classification", "false_hit": True},
    )
    assert resp.status_code == 200
    assert resp.json()["category"] == "classification"
