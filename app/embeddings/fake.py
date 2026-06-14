"""Deterministic, network-free embedding service for tests and local dev.

The vector is derived from a hash of the text so identical inputs map to
identical vectors and lexically similar inputs map to *somewhat* similar
vectors. It is not semantically meaningful — it exists so the cache machinery
can be exercised without calling a paid API.
"""

from __future__ import annotations

import hashlib

import numpy as np

from app.embeddings.base import EmbeddingResult, EmbeddingService


class FakeEmbeddingService(EmbeddingService):
    def __init__(self, *, model: str = "fake-embed", dimensions: int = 1536) -> None:
        super().__init__(model=model, dimensions=dimensions)

    def _vector_for(self, text: str) -> np.ndarray:
        # Seed a PRNG from a stable hash of the normalized text.
        digest = hashlib.sha256(text.strip().lower().encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self.dimensions).astype(np.float32)
        return self.normalize(vector)

    async def embed(self, text: str) -> EmbeddingResult:
        vector = self._vector_for(text)
        return EmbeddingResult(
            vector=vector.tolist(),
            model=self.model,
            dimensions=self.dimensions,
            tokens=max(1, len(text) // 4),
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        return [await self.embed(t) for t in texts]
