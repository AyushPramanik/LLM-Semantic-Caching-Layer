"""Vector store abstraction backed by Redis Stack (RediSearch).

The store persists :class:`CacheEntry` records as Redis hashes and indexes their
embeddings with an HNSW vector field so we can run cosine KNN queries filtered
by cache namespace. The abstract :class:`VectorStore` interface lets the rest of
the system stay decoupled from Redis (see the in-memory store used in tests).
"""

from __future__ import annotations

import abc

import numpy as np
import orjson
import redis.asyncio as redis
from redis.commands.search.field import NumericField, TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from app.core.logging import get_logger
from app.models.cache import CacheEntry, ScoredEntry

logger = get_logger(__name__)


class VectorStore(abc.ABC):
    """Interface for storing and similarity-searching cache entries."""

    @abc.abstractmethod
    async def ensure_index(self) -> None:
        """Create the underlying index if it does not yet exist."""

    @abc.abstractmethod
    async def upsert(self, entry: CacheEntry) -> None:
        """Insert or replace a cache entry."""

    @abc.abstractmethod
    async def search(
        self, namespace: str, vector: list[float], top_k: int = 5
    ) -> list[ScoredEntry]:
        """Return the ``top_k`` nearest entries within ``namespace`` by cosine."""

    @abc.abstractmethod
    async def get(self, entry_id: str) -> CacheEntry | None:
        ...

    @abc.abstractmethod
    async def increment_hit(self, entry_id: str) -> None:
        ...

    @abc.abstractmethod
    async def delete(
        self,
        *,
        model: str | None = None,
        system_prompt_hash: str | None = None,
        tag: str | None = None,
    ) -> int:
        """Delete entries matching any provided filter; returns count removed."""

    @abc.abstractmethod
    async def clear(self) -> int:
        """Delete every entry in the index; returns count removed."""

    @abc.abstractmethod
    async def count(self) -> int:
        ...

    async def aclose(self) -> None:
        ...


class RedisVectorStore(VectorStore):
    """RediSearch-backed vector store for Redis Stack."""

    def __init__(
        self,
        *,
        redis_url: str,
        index_name: str = "semantic_cache",
        dimensions: int = 1536,
    ) -> None:
        self._client: redis.Redis = redis.from_url(redis_url, decode_responses=False)
        self._index = index_name
        self._prefix = f"{index_name}:entry:"
        self._dimensions = dimensions

    def _key(self, entry_id: str) -> str:
        return f"{self._prefix}{entry_id}"

    async def ensure_index(self) -> None:
        try:
            await self._client.ft(self._index).info()
            return  # already exists
        except Exception:  # noqa: BLE001 - redis raises a generic error here
            pass

        schema = (
            TagField("namespace"),
            TagField("model"),
            TagField("provider"),
            TagField("system_prompt_hash"),
            TagField("tags", separator=","),
            NumericField("created_at"),
            VectorField(
                "embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self._dimensions,
                    "DISTANCE_METRIC": "COSINE",
                    "M": 16,
                    "EF_CONSTRUCTION": 200,
                },
            ),
        )
        definition = IndexDefinition(prefix=[self._prefix], index_type=IndexType.HASH)
        await self._client.ft(self._index).create_index(schema, definition=definition)
        logger.info("vectorstore.index_created", index=self._index, dim=self._dimensions)

    async def upsert(self, entry: CacheEntry) -> None:
        vector = np.asarray(entry.embedding, dtype=np.float32).tobytes()
        mapping = {
            "namespace": entry.namespace,
            "model": entry.model,
            "provider": entry.provider,
            "system_prompt_hash": entry.system_prompt_hash or "none",
            "tags": ",".join(entry.tags) if entry.tags else "none",
            "created_at": entry.created_at,
            "embedding": vector,
            "data": orjson.dumps(entry.model_dump()),
        }
        key = self._key(entry.id)
        await self._client.hset(key, mapping=mapping)
        if entry.ttl_seconds > 0:
            await self._client.expire(key, entry.ttl_seconds)

    async def search(
        self, namespace: str, vector: list[float], top_k: int = 5
    ) -> list[ScoredEntry]:
        query_vec = np.asarray(vector, dtype=np.float32).tobytes()
        escaped = _escape_tag(namespace)
        q = (
            Query(f"(@namespace:{{{escaped}}})=>[KNN {top_k} @embedding $vec AS dist]")
            .sort_by("dist")
            .return_fields("data", "dist")
            .dialect(2)
        )
        res = await self._client.ft(self._index).search(q, query_params={"vec": query_vec})

        scored: list[ScoredEntry] = []
        for doc in res.docs:
            entry = CacheEntry.model_validate(orjson.loads(doc.data))
            # RediSearch COSINE returns a distance in [0, 2]; similarity = 1 - dist.
            similarity = 1.0 - float(doc.dist)
            scored.append(ScoredEntry(entry=entry, score=similarity))
        return scored

    async def get(self, entry_id: str) -> CacheEntry | None:
        raw = await self._client.hget(self._key(entry_id), "data")
        if raw is None:
            return None
        return CacheEntry.model_validate(orjson.loads(raw))

    async def increment_hit(self, entry_id: str) -> None:
        entry = await self.get(entry_id)
        if entry is None:
            return
        entry.hit_count += 1
        await self._client.hset(self._key(entry_id), "data", orjson.dumps(entry.model_dump()))

    async def delete(
        self,
        *,
        model: str | None = None,
        system_prompt_hash: str | None = None,
        tag: str | None = None,
    ) -> int:
        clauses = []
        if model:
            clauses.append(f"@model:{{{_escape_tag(model)}}}")
        if system_prompt_hash:
            clauses.append(f"@system_prompt_hash:{{{_escape_tag(system_prompt_hash)}}}")
        if tag:
            clauses.append(f"@tags:{{{_escape_tag(tag)}}}")
        if not clauses:
            return 0
        return await self._delete_matching(" ".join(clauses))

    async def clear(self) -> int:
        return await self._delete_matching("*")

    async def _delete_matching(self, query_str: str) -> int:
        q = Query(query_str).return_fields("id").paging(0, 10_000).dialect(2)
        res = await self._client.ft(self._index).search(q)
        keys = [doc.id for doc in res.docs]
        if keys:
            await self._client.delete(*keys)
        return len(keys)

    async def count(self) -> int:
        info = await self._client.ft(self._index).info()
        return int(info.get("num_docs", 0))

    async def aclose(self) -> None:
        await self._client.aclose()


def _escape_tag(value: str) -> str:
    """Escape RediSearch TAG special characters."""
    out = value
    for ch in "-.@{}|/ ":
        out = out.replace(ch, f"\\{ch}")
    return out
