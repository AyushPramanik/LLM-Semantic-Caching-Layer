"""In-memory brute-force vector store.

A faithful, dependency-free implementation of :class:`VectorStore` used by unit
tests and local development where standing up Redis Stack is overkill. It runs
exact cosine KNN rather than the approximate HNSW search Redis uses, which makes
test assertions deterministic.
"""

from __future__ import annotations

import time

import numpy as np

from app.cache.similarity import cosine_similarity_matrix, top_k_indices
from app.cache.store import VectorStore
from app.models.cache import CacheEntry, ScoredEntry


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._entries: dict[str, CacheEntry] = {}

    async def ensure_index(self) -> None:  # no-op
        return None

    async def upsert(self, entry: CacheEntry) -> None:
        self._entries[entry.id] = entry

    def _live_entries(self, namespace: str) -> list[CacheEntry]:
        now = time.time()
        live = []
        for entry in self._entries.values():
            if entry.namespace != namespace:
                continue
            if entry.is_expired(now):
                continue
            live.append(entry)
        return live

    async def search(
        self, namespace: str, vector: list[float], top_k: int = 5
    ) -> list[ScoredEntry]:
        entries = self._live_entries(namespace)
        if not entries:
            return []
        query = np.asarray(vector, dtype=np.float32)
        matrix = np.asarray([e.embedding for e in entries], dtype=np.float32)
        scores = cosine_similarity_matrix(query, matrix)
        return [
            ScoredEntry(entry=entries[i], score=float(scores[i]))
            for i in top_k_indices(scores, top_k)
        ]

    async def get(self, entry_id: str) -> CacheEntry | None:
        return self._entries.get(entry_id)

    async def increment_hit(self, entry_id: str) -> None:
        entry = self._entries.get(entry_id)
        if entry is not None:
            entry.hit_count += 1

    async def delete(
        self,
        *,
        model: str | None = None,
        system_prompt_hash: str | None = None,
        tag: str | None = None,
    ) -> int:
        if model is None and system_prompt_hash is None and tag is None:
            return 0

        def matches(entry) -> bool:
            if model is not None and entry.model != model:
                return False
            if system_prompt_hash is not None and entry.system_prompt_hash != system_prompt_hash:
                return False
            if tag is not None and tag not in entry.tags:
                return False
            return True

        to_delete = [eid for eid, entry in self._entries.items() if matches(entry)]
        for entry_id in to_delete:
            self._entries.pop(entry_id, None)
        return len(to_delete)

    async def clear(self) -> int:
        count = len(self._entries)
        self._entries.clear()
        return count

    async def count(self) -> int:
        return len(self._entries)
