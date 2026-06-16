"""Semantic cache lookup and write workflow.

Ties together the embedding service, the vector store, and the similarity
threshold into the core request flow:

    embed prompt -> KNN search within namespace -> compare best score to
    threshold -> HIT (serve cached response) or MISS (caller forwards upstream
    and writes the result back).

The orchestrator is provider- and transport-agnostic; the HTTP layer adapts
OpenAI-shaped requests onto it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.cache.namespace import RequestSignature
from app.cache.store import VectorStore
from app.core.logging import get_logger
from app.embeddings.base import EmbeddingService
from app.models.cache import CacheEntry, ScoredEntry

logger = get_logger(__name__)


class CacheStatus(str, Enum):
    HIT = "HIT"
    MISS = "MISS"


@dataclass(slots=True)
class CacheLookup:
    """Outcome of a cache lookup, including observability metadata."""

    status: CacheStatus
    score: float
    latency_ms: float
    match: ScoredEntry | None = None
    near_miss: bool = False

    @property
    def is_hit(self) -> bool:
        return self.status is CacheStatus.HIT


class SemanticCache:
    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        store: VectorStore,
        threshold: float = 0.95,
        near_miss_window: float = 0.05,
        top_k: int = 5,
    ) -> None:
        self._embeddings = embedding_service
        self._store = store
        self._threshold = threshold
        self._near_miss_window = near_miss_window
        self._top_k = top_k

    async def lookup(
        self,
        signature: RequestSignature,
        prompt: str,
        *,
        threshold: float | None = None,
    ) -> CacheLookup:
        """Look up a semantically similar cached response."""
        effective_threshold = threshold if threshold is not None else self._threshold
        started = time.perf_counter()

        embedding = await self._embeddings.embed(prompt)
        matches = await self._store.search(
            signature.namespace(), embedding.vector, top_k=self._top_k
        )
        best = matches[0] if matches else None
        score = best.score if best else 0.0

        if best is not None and score >= effective_threshold:
            await self._store.increment_hit(best.id)
            status = CacheStatus.HIT
            near_miss = False
        else:
            status = CacheStatus.MISS
            near_miss = best is not None and (
                effective_threshold - self._near_miss_window <= score < effective_threshold
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "cache.lookup",
            status=status.value,
            score=round(score, 4),
            threshold=effective_threshold,
            near_miss=near_miss,
            namespace=signature.namespace(),
            latency_ms=round(latency_ms, 3),
        )
        return CacheLookup(
            status=status,
            score=score,
            latency_ms=latency_ms,
            match=best if status is CacheStatus.HIT else best,
            near_miss=near_miss,
        )

    async def store_response(
        self,
        signature: RequestSignature,
        prompt: str,
        response: dict[str, Any],
        *,
        ttl_seconds: int,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> CacheEntry:
        """Embed and persist a successful response for future reuse."""
        embedding = await self._embeddings.embed(prompt)
        entry = CacheEntry(
            namespace=signature.namespace(),
            prompt=prompt,
            embedding=embedding.vector,
            response=response,
            model=signature.model,
            provider=signature.provider,
            system_prompt_hash=signature.system_prompt_hash,
            temperature=signature.temperature,
            max_tokens=signature.max_tokens,
            ttl_seconds=ttl_seconds,
            tags=tags or [],
            metadata=metadata or {},
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        await self._store.upsert(entry)
        logger.info(
            "cache.store",
            namespace=entry.namespace,
            ttl_seconds=ttl_seconds,
            tags=entry.tags,
        )
        return entry
