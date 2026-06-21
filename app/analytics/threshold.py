"""Similarity-threshold analytics.

Choosing a similarity threshold is a precision/recall tradeoff: too low and the
cache returns answers to *different* questions (false hits); too high and it
misses genuine paraphrases (wasted spend). This module sweeps a set of candidate
thresholds over a labeled dataset of prompt pairs and reports, per threshold:

    hit_rate, precision, recall, false_positive_rate, false_negative_rate

so the operating point can be chosen from data rather than guesswork.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from app.cache.similarity import cosine_similarity
from app.embeddings.base import EmbeddingService

DEFAULT_THRESHOLDS = (0.90, 0.92, 0.95, 0.98)


class ThresholdTestCase(BaseModel):
    """A labeled pair: should these two prompts be treated as the same query?"""

    query: str
    candidate: str
    expected_match: bool


class ThresholdMetrics(BaseModel):
    threshold: float
    hit_rate: float
    precision: float
    recall: float
    false_positive_rate: float
    false_negative_rate: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


class ThresholdReport(BaseModel):
    sample_size: int
    metrics: list[ThresholdMetrics]
    recommended_threshold: float


@dataclass
class _Confusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0


@dataclass
class ThresholdAnalyzer:
    embedding_service: EmbeddingService
    thresholds: tuple[float, ...] = field(default_factory=lambda: DEFAULT_THRESHOLDS)

    async def _similarities(self, cases: list[ThresholdTestCase]) -> list[float]:
        sims: list[float] = []
        for case in cases:
            a = await self.embedding_service.embed(case.query)
            b = await self.embedding_service.embed(case.candidate)
            sims.append(cosine_similarity(a.vector, b.vector))
        return sims

    @staticmethod
    def _metrics(threshold: float, cases, sims) -> ThresholdMetrics:
        c = _Confusion()
        for case, sim in zip(cases, sims, strict=True):
            predicted = sim >= threshold
            if case.expected_match and predicted:
                c.tp += 1
            elif case.expected_match and not predicted:
                c.fn += 1
            elif not case.expected_match and predicted:
                c.fp += 1
            else:
                c.tn += 1

        total = max(1, c.tp + c.fp + c.tn + c.fn)
        return ThresholdMetrics(
            threshold=threshold,
            hit_rate=(c.tp + c.fp) / total,
            precision=_safe_div(c.tp, c.tp + c.fp),
            recall=_safe_div(c.tp, c.tp + c.fn),
            false_positive_rate=_safe_div(c.fp, c.fp + c.tn),
            false_negative_rate=_safe_div(c.fn, c.fn + c.tp),
            true_positives=c.tp,
            false_positives=c.fp,
            true_negatives=c.tn,
            false_negatives=c.fn,
        )

    async def evaluate(self, cases: list[ThresholdTestCase]) -> ThresholdReport:
        sims = await self._similarities(cases)
        metrics = [self._metrics(t, cases, sims) for t in sorted(self.thresholds)]
        recommended = _recommend(metrics)
        return ThresholdReport(
            sample_size=len(cases),
            metrics=metrics,
            recommended_threshold=recommended,
        )


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _recommend(metrics: list[ThresholdMetrics]) -> float:
    """Pick the threshold maximizing F1, breaking ties toward higher precision.

    Cache false hits are costly (a wrong answer), so when scores tie we prefer
    the safer, higher-precision operating point.
    """
    if not metrics:
        return 0.95

    def f1(m: ThresholdMetrics) -> float:
        if m.precision + m.recall == 0:
            return 0.0
        return 2 * m.precision * m.recall / (m.precision + m.recall)

    return max(metrics, key=lambda m: (round(f1(m), 4), m.precision, m.threshold)).threshold
