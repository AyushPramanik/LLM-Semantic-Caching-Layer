"""FastAPI dependency providers.

Services are constructed once during application startup and stashed on
``app.state``; these helpers expose them to request handlers and keep the wiring
in one place for easy testing/overriding.
"""

from __future__ import annotations

from fastapi import Request

from app.cache.store import VectorStore
from app.proxy.service import ProxyService


def get_proxy(request: Request) -> ProxyService:
    return request.app.state.proxy


def get_store(request: Request) -> VectorStore:
    return request.app.state.store


def get_tenant(request: Request) -> str:
    """Resolve the calling application/tenant for cache isolation.

    Uses an explicit ``X-Tenant-Id`` header when present, otherwise falls back to
    a shared default. This is the seam where API-key-based tenancy would plug in.
    """
    return request.headers.get("X-Tenant-Id", "default")
