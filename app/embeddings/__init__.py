"""Embedding services.

The cache turns prompts into dense vectors via an ``EmbeddingService``. The
concrete implementation is selected from configuration so the rest of the system
depends only on the abstract interface.
"""

from __future__ import annotations

from app.core.config import Settings
from app.embeddings.base import EmbeddingResult, EmbeddingService
from app.embeddings.fake import FakeEmbeddingService
from app.embeddings.openai import OpenAIEmbeddingService

__all__ = [
    "EmbeddingResult",
    "EmbeddingService",
    "FakeEmbeddingService",
    "OpenAIEmbeddingService",
    "build_embedding_service",
]


def build_embedding_service(settings: Settings) -> EmbeddingService:
    """Construct the configured embedding service."""
    if settings.embedding_provider == "fake":
        return FakeEmbeddingService(
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
    return OpenAIEmbeddingService(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        timeout=settings.upstream_timeout_seconds,
    )
