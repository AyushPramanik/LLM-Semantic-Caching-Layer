"""Heuristic TTL classification.

Not every answer ages at the same rate. A definition of recursion is good for
days; today's stock price is stale in minutes. The classifier buckets a prompt
into one of three freshness classes and assigns a TTL:

* ``LONG``     — stable knowledge (facts, concepts, programming). 24h+.
* ``SHORT``    — time-sensitive (finance, weather, news, current events). ~1h.
* ``NO_CACHE`` — inherently live/volatile (live scores, "right now"). Not cached.

The heuristics are intentionally transparent and cheap (keyword/regex matching)
so behavior is predictable and easy to tune. They emit a tag for analytics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class TtlClass(str, Enum):
    LONG = "long"
    SHORT = "short"
    NO_CACHE = "no_cache"


@dataclass(frozen=True, slots=True)
class TtlDecision:
    ttl_class: TtlClass
    ttl_seconds: int
    reason: str

    @property
    def cacheable(self) -> bool:
        return self.ttl_class is not TtlClass.NO_CACHE

    @property
    def tag(self) -> str:
        return f"ttl:{self.ttl_class.value}"


# Volatile / live information — should never be cached.
_NO_CACHE_PATTERNS = [
    r"\bright now\b",
    r"\blive\b",
    r"\bcurrently\b",
    r"\bat this (moment|time)\b",
    r"\bscore\b.*\b(game|match)\b",
    r"\bbreaking news\b",
    r"\bas of (today|now)\b",
]

# Time-sensitive — short TTL.
_SHORT_KEYWORDS = [
    "stock price", "share price", "stock market", "exchange rate", "crypto",
    "bitcoin price", "weather", "forecast", "news", "headline", "today",
    "this week", "current events", "trending", "latest",
]

# Stable knowledge — long TTL.
_LONG_KEYWORDS = [
    "what is", "what are", "who was", "who is", "explain", "define",
    "definition", "how does", "how do i", "difference between", "history of",
    "example of", "concept", "algorithm", "complexity", "python", "javascript",
    "sql", "regex", "function", "recursion", "data structure",
]


class TtlClassifier:
    def __init__(
        self,
        *,
        long_ttl_seconds: int = 86_400,
        short_ttl_seconds: int = 3_600,
    ) -> None:
        self._long_ttl = long_ttl_seconds
        self._short_ttl = short_ttl_seconds
        self._no_cache_re = [re.compile(p, re.IGNORECASE) for p in _NO_CACHE_PATTERNS]

    def classify(self, prompt: str) -> TtlDecision:
        text = prompt.lower().strip()

        for pattern in self._no_cache_re:
            if pattern.search(text):
                reason = f"matched volatile pattern '{pattern.pattern}'"
                return TtlDecision(TtlClass.NO_CACHE, 0, reason)

        if any(keyword in text for keyword in _SHORT_KEYWORDS):
            return TtlDecision(TtlClass.SHORT, self._short_ttl, "time-sensitive keyword")

        if any(keyword in text for keyword in _LONG_KEYWORDS):
            return TtlDecision(TtlClass.LONG, self._long_ttl, "stable-knowledge keyword")

        # Default: treat as reasonably stable but not maximally so.
        return TtlDecision(TtlClass.LONG, self._long_ttl, "default stable")


def build_ttl_policy(classifier: TtlClassifier):
    """Adapt a classifier into the proxy's ``ttl_policy`` callable.

    Returns ``(ttl_seconds, tags)``; a non-positive TTL signals "do not cache".
    Caller-supplied ``cache_tags`` are merged with the classifier's tag.
    """

    def policy(request, response) -> tuple[int, list[str]]:  # noqa: ANN001
        decision = classifier.classify(request.latest_user_prompt())
        tags = list(request.cache_tags or [])
        tags.append(decision.tag)
        return decision.ttl_seconds, tags

    return policy
