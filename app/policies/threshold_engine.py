"""Adaptive, per-category similarity thresholds.

Different request types tolerate different amounts of fuzziness:

* **Classification** (sentiment, yes/no, labeling) — answers are coarse and
  robust to paraphrase, so a lower threshold (~0.90–0.93) is safe and lifts hit
  rate.
* **Programming** — small wording changes can change intent ("sort ascending" vs
  "sort descending"), so the default 0.95 is appropriate.
* **Creative writing** — outputs are bespoke; only near-identical prompts should
  share a cached generation, so 0.98+.

Thresholds start from a configurable base per category and are nudged by
feedback: a reported false hit raises the threshold (be stricter); a reported
missed-but-equivalent raises hit rate by lowering it. Movement is bounded so the
engine self-corrects without drifting to extremes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class ThresholdCategory(str, Enum):
    CLASSIFICATION = "classification"
    PROGRAMMING = "programming"
    CREATIVE = "creative"
    GENERAL = "general"


@dataclass(slots=True)
class CategoryPolicy:
    """A category's current threshold and the bounds it may adapt within."""

    base: float
    minimum: float
    maximum: float
    step: float = 0.005
    current: float = field(init=False)

    def __post_init__(self) -> None:
        self.current = self.base

    def adjust(self, delta: float) -> float:
        self.current = min(self.maximum, max(self.minimum, self.current + delta))
        return self.current


_DEFAULT_POLICIES: dict[ThresholdCategory, CategoryPolicy] = {
    ThresholdCategory.CLASSIFICATION: CategoryPolicy(0.91, 0.88, 0.95),
    ThresholdCategory.PROGRAMMING: CategoryPolicy(0.95, 0.93, 0.98),
    ThresholdCategory.CREATIVE: CategoryPolicy(0.985, 0.97, 0.995),
    ThresholdCategory.GENERAL: CategoryPolicy(0.95, 0.92, 0.98),
}

_CREATIVE_RE = re.compile(
    r"\b(write|compose|draft|brainstorm)\b.*\b(poem|story|song|essay|tagline|"
    r"fiction|lyrics|joke|haiku)\b|\bcreative\b|\bimagine\b",
    re.IGNORECASE,
)
_PROGRAMMING_KEYWORDS = (
    "code", "python", "javascript", "java", "c++", "rust", "golang", "function",
    "bug", "stack trace", "compile", "regex", "sql", "algorithm", "implement",
    "refactor", "unit test", "exception", "api endpoint",
)
_CLASSIFICATION_KEYWORDS = (
    "classify", "categorize", "sentiment", "is this", "label this", "yes or no",
    "true or false", "which category", "detect the", "is the following",
)


class AdaptiveThresholdEngine:
    def __init__(self, policies: dict[ThresholdCategory, CategoryPolicy] | None = None) -> None:
        self._policies = policies or {k: CategoryPolicy(p.base, p.minimum, p.maximum, p.step)
                                      for k, p in _DEFAULT_POLICIES.items()}

    def categorize(self, prompt: str) -> ThresholdCategory:
        text = prompt.lower().strip()
        if _CREATIVE_RE.search(text):
            return ThresholdCategory.CREATIVE
        if any(k in text for k in _PROGRAMMING_KEYWORDS):
            return ThresholdCategory.PROGRAMMING
        if any(k in text for k in _CLASSIFICATION_KEYWORDS):
            return ThresholdCategory.CLASSIFICATION
        return ThresholdCategory.GENERAL

    def threshold_for_prompt(self, prompt: str) -> float:
        return self._policies[self.categorize(prompt)].current

    def record_feedback(self, category: ThresholdCategory, *, false_hit: bool) -> float:
        """Adapt a category's threshold from a feedback signal.

        ``false_hit=True`` means the cache served a wrong answer -> tighten.
        ``false_hit=False`` means an equivalent prompt was needlessly missed ->
        loosen.
        """
        policy = self._policies[category]
        return policy.adjust(policy.step if false_hit else -policy.step)

    def snapshot(self) -> dict[str, float]:
        return {category.value: policy.current for category, policy in self._policies.items()}


def build_threshold_policy(engine: AdaptiveThresholdEngine):
    """Adapt the engine into the proxy's ``threshold_policy`` callable."""

    def policy(request) -> float:  # noqa: ANN001
        return engine.threshold_for_prompt(request.latest_user_prompt())

    return policy
