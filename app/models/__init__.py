"""Pydantic data models shared across the service."""

from app.models.cache import CacheEntry, ScoredEntry

__all__ = ["CacheEntry", "ScoredEntry"]
