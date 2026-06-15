"""Cosine similarity primitives.

Embeddings produced by :class:`EmbeddingService` are L2-normalized, so cosine
similarity reduces to a dot product. These helpers are intentionally tiny and
dependency-light so they can be reused by the in-memory store, the validation
replay path, and analytics.
"""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Return cosine similarity in [-1, 1] (robust to non-normalized inputs)."""
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of ``query`` against every row of ``matrix``."""
    if matrix.size == 0:
        return np.empty(0, dtype=np.float32)
    q_norm = np.linalg.norm(query)
    row_norms = np.linalg.norm(matrix, axis=1)
    denom = row_norms * (q_norm or 1.0)
    denom[denom == 0.0] = 1.0
    return (matrix @ query) / denom


def top_k_indices(scores: np.ndarray, k: int) -> list[int]:
    """Return indices of the ``k`` highest scores, best first."""
    if scores.size == 0:
        return []
    k = min(k, scores.size)
    # argpartition for the top-k, then sort just those descending.
    part = np.argpartition(scores, -k)[-k:]
    return [int(i) for i in part[np.argsort(scores[part])[::-1]]]
