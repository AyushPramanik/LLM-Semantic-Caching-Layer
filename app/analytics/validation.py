"""Cache validation via shadow replay.

A semantic cache trades correctness for cost: occasionally it may serve a stored
answer to a prompt that is *similar but not equivalent*, or the world may have
moved on since the answer was cached (semantic drift). To keep the cache honest
in production, we **shadow-replay** a small, configurable fraction of cache hits:

  1. Re-issue the request to the real provider (off the hot path).
  2. Embed both the cached answer and the fresh answer.
  3. Compare them by cosine similarity.

A low similarity means the cached answer no longer matches what the provider
would produce — a *false hit* / drift. We expose:

  * ``cache_validation_accuracy`` — share of validated hits that still match.
  * ``semantic_drift_rate``       — share showing material drift.
  * ``false_hit_rate``            — share that were effectively wrong.

This is deliberately prominent: it is the difference between a cache that *looks*
cheap and one you can actually trust in production.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from pydantic import BaseModel

from app.cache.similarity import cosine_similarity
from app.core.logging import get_logger
from app.embeddings.base import EmbeddingService
from app.models.chat import ChatCompletionRequest, ChatCompletionResponse

logger = get_logger(__name__)


@dataclass(slots=True)
class ValidationResult:
    response_similarity: float
    drift: bool
    false_hit: bool


class ValidationStats(BaseModel):
    validations: int
    drift_count: int
    false_hit_count: int
    cache_validation_accuracy: float
    semantic_drift_rate: float
    false_hit_rate: float
    sample_rate: float
    drift_threshold: float


class CacheValidator:
    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        completer,
        sample_rate: float = 0.02,
        drift_threshold: float = 0.90,
        metrics=None,
        rng: random.Random | None = None,
    ) -> None:
        self._embeddings = embedding_service
        self._completer = completer
        self._sample_rate = sample_rate
        self._drift_threshold = drift_threshold
        self._metrics = metrics
        self._rng = rng or random.Random()

        self._validations = 0
        self._drift = 0
        self._false_hits = 0

    def should_validate(self) -> bool:
        return self._sample_rate > 0 and self._rng.random() < self._sample_rate

    async def validate(
        self, request: ChatCompletionRequest, cached: ChatCompletionResponse
    ) -> ValidationResult:
        fresh = await self._completer.complete(request)
        cached_emb = await self._embeddings.embed(cached.first_text())
        fresh_emb = await self._embeddings.embed(fresh.first_text())
        similarity = cosine_similarity(cached_emb.vector, fresh_emb.vector)

        drift = similarity < self._drift_threshold
        self._validations += 1
        if drift:
            self._drift += 1
            self._false_hits += 1

        if self._metrics is not None:
            self._metrics.record_validation(
                accuracy=self.stats().cache_validation_accuracy,
                drift_rate=self.stats().semantic_drift_rate,
                false_hit_rate=self.stats().false_hit_rate,
            )

        logger.info(
            "cache.validation",
            response_similarity=round(similarity, 4),
            drift=drift,
            model=request.model,
        )
        return ValidationResult(response_similarity=similarity, drift=drift, false_hit=drift)

    def stats(self) -> ValidationStats:
        n = self._validations
        drift_rate = self._drift / n if n else 0.0
        false_hit_rate = self._false_hits / n if n else 0.0
        return ValidationStats(
            validations=n,
            drift_count=self._drift,
            false_hit_count=self._false_hits,
            cache_validation_accuracy=1.0 - false_hit_rate,
            semantic_drift_rate=drift_rate,
            false_hit_rate=false_hit_rate,
            sample_rate=self._sample_rate,
            drift_threshold=self._drift_threshold,
        )
