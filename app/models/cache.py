"""Cache entry schema and search result types.

A ``CacheEntry`` is the unit stored in the vector index. It carries everything
needed to (a) safely match a request to a prior response and (b) reason about
cost, freshness, and observability.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


class CacheEntry(BaseModel):
    """A cached prompt/response pair plus its embedding and provenance."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    namespace: str

    # Matching inputs
    prompt: str
    embedding: list[float]

    # Stored response payload (OpenAI-compatible chat completion JSON)
    response: dict[str, Any]

    # Safety / routing dimensions — entries must never be shared across these.
    model: str
    provider: str
    system_prompt_hash: str = ""
    temperature: float = 1.0
    max_tokens: int | None = None

    # Lifecycle / observability
    created_at: float = Field(default_factory=time.time)
    ttl_seconds: int = 86_400
    hit_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def age_seconds(self, now: float | None = None) -> float:
        return (now or time.time()) - self.created_at

    def is_expired(self, now: float | None = None) -> bool:
        if self.ttl_seconds <= 0:
            return True
        return self.age_seconds(now) > self.ttl_seconds


class ScoredEntry(BaseModel):
    """A cache entry returned from a similarity search with its score."""

    entry: CacheEntry
    score: float

    @property
    def id(self) -> str:
        return self.entry.id
