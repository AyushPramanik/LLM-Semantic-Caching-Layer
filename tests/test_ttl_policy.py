"""Tests for the heuristic TTL classifier."""

from __future__ import annotations

import pytest

from app.policies.ttl_classifier import TtlClass, TtlClassifier, build_ttl_policy


@pytest.fixture
def classifier():
    return TtlClassifier(long_ttl_seconds=86_400, short_ttl_seconds=3_600)


@pytest.mark.parametrize(
    "prompt",
    [
        "What is the time complexity of quicksort?",
        "Explain recursion to a beginner",
        "Define photosynthesis",
        "How do I reverse a linked list in Python?",
    ],
)
def test_stable_knowledge_is_long(classifier, prompt):
    decision = classifier.classify(prompt)
    assert decision.ttl_class is TtlClass.LONG
    assert decision.ttl_seconds == 86_400
    assert decision.cacheable


@pytest.mark.parametrize(
    "prompt",
    [
        "What is the current stock price of AAPL?",
        "Give me today's weather in Seattle",
        "What's the latest news on the election?",
        "What is the bitcoin price right this week",
    ],
)
def test_time_sensitive_is_short(classifier, prompt):
    decision = classifier.classify(prompt)
    assert decision.ttl_class is TtlClass.SHORT
    assert decision.ttl_seconds == 3_600
    assert decision.cacheable


@pytest.mark.parametrize(
    "prompt",
    [
        "What is the score of the game right now?",
        "Who is winning the match live?",
        "Give me breaking news as of now",
    ],
)
def test_volatile_is_no_cache(classifier, prompt):
    decision = classifier.classify(prompt)
    assert decision.ttl_class is TtlClass.NO_CACHE
    assert decision.ttl_seconds == 0
    assert not decision.cacheable


def test_tag_reflects_class(classifier):
    assert classifier.classify("Explain TCP").tag == "ttl:long"


def test_policy_merges_caller_tags(classifier):
    from app.models.chat import ChatCompletionRequest, ChatMessage

    policy = build_ttl_policy(classifier)
    request = ChatCompletionRequest(
        model="gpt-4o-mini",
        messages=[ChatMessage(role="user", content="Explain DNS")],
        cache_tags=["docs"],
    )
    ttl, tags = policy(request, None)
    assert ttl == 86_400
    assert "docs" in tags and "ttl:long" in tags
