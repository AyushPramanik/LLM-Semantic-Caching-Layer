"""Prometheus scrape endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST

router = APIRouter(tags=["monitoring"])


@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    cache_metrics = request.app.state.metrics
    store = request.app.state.store
    # Refresh the cache-size gauge at scrape time so it reflects live state.
    try:
        cache_metrics.set_cache_size(await store.count())
    except Exception:  # noqa: BLE001 - never let a metrics scrape fail the endpoint
        pass
    return Response(content=cache_metrics.render(), media_type=CONTENT_TYPE_LATEST)
