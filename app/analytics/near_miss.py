"""Near-miss analyzer.

A "near miss" is a request that *just* failed to clear the similarity threshold —
for a 0.95 threshold, a best match in roughly [0.90, 0.95). These are the most
informative signals for tuning: a pile-up of near misses means the threshold may
be slightly too strict and is leaving cache hits (and savings) on the table.

The tracker keeps a bounded recent window and turns it into a recommendation.
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass

from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class NearMiss:
    score: float
    threshold: float
    namespace: str
    prompt_preview: str
    timestamp: float

    @property
    def gap(self) -> float:
        return self.threshold - self.score


class NearMissReport(BaseModel):
    count: int
    window_size: int
    mean_score: float | None = None
    mean_gap: float | None = None
    histogram: dict[str, int]
    recommended_threshold: float | None = None
    recommendation: str


class NearMissTracker:
    def __init__(self, *, window: float = 0.05, capacity: int = 1000, min_samples: int = 20) -> None:
        self._window = window
        self._min_samples = min_samples
        self._events: deque[NearMiss] = deque(maxlen=capacity)

    def record(self, *, score: float, threshold: float, namespace: str, prompt: str) -> None:
        self._events.append(
            NearMiss(
                score=score,
                threshold=threshold,
                namespace=namespace,
                prompt_preview=prompt[:120],
                timestamp=time.time(),
            )
        )

    def _histogram(self) -> dict[str, int]:
        bins = {"0.90-0.92": 0, "0.92-0.94": 0, "0.94-0.95": 0, "other": 0}
        for event in self._events:
            s = event.score
            if 0.90 <= s < 0.92:
                bins["0.90-0.92"] += 1
            elif 0.92 <= s < 0.94:
                bins["0.92-0.94"] += 1
            elif 0.94 <= s < 0.95:
                bins["0.94-0.95"] += 1
            else:
                bins["other"] += 1
        return bins

    def report(self) -> NearMissReport:
        events = list(self._events)
        count = len(events)
        if count == 0:
            return NearMissReport(
                count=0, window_size=self._events.maxlen or 0, histogram=self._histogram(),
                recommendation="No near misses observed yet.",
            )

        scores = [e.score for e in events]
        mean_score = statistics.fmean(scores)
        mean_gap = statistics.fmean(e.gap for e in events)

        recommended = None
        if count >= self._min_samples:
            # Lowering the threshold to just below the 25th percentile of recent
            # near misses would convert most of them into hits.
            ordered = sorted(scores)
            p25 = ordered[max(0, int(0.25 * len(ordered)) - 1)]
            recommended = round(p25 - 0.005, 3)
            recommendation = (
                f"{count} near misses observed (mean score {mean_score:.3f}). Lowering the "
                f"threshold to ~{recommended} would recover most of them; validate against "
                f"/analytics/threshold-test before rolling out."
            )
        else:
            recommendation = (
                f"Only {count} near misses so far (need {self._min_samples} for a "
                f"recommendation). Mean score {mean_score:.3f}."
            )

        return NearMissReport(
            count=count,
            window_size=self._events.maxlen or 0,
            mean_score=round(mean_score, 4),
            mean_gap=round(mean_gap, 4),
            histogram=self._histogram(),
            recommended_threshold=recommended,
            recommendation=recommendation,
        )
