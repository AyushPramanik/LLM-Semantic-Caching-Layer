"""Abstract embedding service interface."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """A single embedding plus light provenance for observability."""

    vector: list[float]
    model: str
    dimensions: int
    tokens: int = 0

    def as_numpy(self) -> np.ndarray:
        return np.asarray(self.vector, dtype=np.float32)


class EmbeddingService(abc.ABC):
    """Turns text into dense vectors.

    Implementations must return L2-normalized vectors so that a dot product is
    equivalent to cosine similarity downstream — this lets the vector store use
    the cheaper inner-product distance metric.
    """

    def __init__(self, model: str, dimensions: int) -> None:
        self.model = model
        self.dimensions = dimensions

    @abc.abstractmethod
    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single piece of text."""

    @abc.abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed a batch of texts in a single round-trip when possible."""

    async def aclose(self) -> None:
        """Release any underlying network resources."""

    @staticmethod
    def normalize(vector: np.ndarray) -> np.ndarray:
        """Return the L2-normalized vector (zero-safe)."""
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm
