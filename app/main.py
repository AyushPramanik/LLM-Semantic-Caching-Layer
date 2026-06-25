"""Application entrypoint and FastAPI factory.

This module wires together configuration, logging, the semantic cache, and the
OpenAI-compatible routing surface. Long-lived resources (embedding client, Redis
vector store) are created in the application lifespan and exposed on
``app.state`` for dependency injection.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app import __version__
from app.analytics.near_miss import NearMissTracker
from app.analytics.validation import CacheValidator
from app.api.analytics import router as analytics_router
from app.api.cache import router as cache_router
from app.api.chat import router as chat_router
from app.api.metrics import router as metrics_router
from app.cache.memory_store import InMemoryVectorStore
from app.cache.semantic_cache import SemanticCache
from app.cache.store import RedisVectorStore, VectorStore
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.core.middleware import CorrelationIdMiddleware, RateLimitMiddleware
from app.embeddings import build_embedding_service
from app.monitoring.metrics import CacheMetrics
from app.policies import (
    AdaptiveThresholdEngine,
    TtlClassifier,
    build_threshold_policy,
    build_ttl_policy,
)
from app.providers import build_router
from app.providers.base import ProviderError
from app.proxy.echo import EchoCompleter
from app.proxy.service import Completer, ProxyService

logger = get_logger(__name__)


def _build_store(settings: Settings) -> VectorStore:
    if settings.vector_backend == "memory":
        return InMemoryVectorStore()
    return RedisVectorStore(
        redis_url=str(settings.redis_url),
        index_name=settings.cache_index_name,
        dimensions=settings.embedding_dimensions,
    )


def _build_completer(settings: Settings) -> Completer:
    if settings.completer_backend == "echo":
        return EchoCompleter()
    return build_router(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings

    embeddings = build_embedding_service(settings)
    store = _build_store(settings)
    await store.ensure_index()
    cache = SemanticCache(
        embedding_service=embeddings,
        store=store,
        threshold=settings.similarity_threshold,
        near_miss_window=settings.near_miss_window,
    )
    ttl_classifier = TtlClassifier(
        long_ttl_seconds=settings.default_ttl_seconds,
        short_ttl_seconds=settings.short_ttl_seconds,
    )
    threshold_engine = AdaptiveThresholdEngine()
    threshold_policy = (
        build_threshold_policy(threshold_engine)
        if settings.adaptive_thresholds_enabled
        else None
    )
    # A per-app registry keeps metrics isolated (important for tests).
    metrics = CacheMetrics()
    near_miss_tracker = NearMissTracker(window=settings.near_miss_window)
    completer = _build_completer(settings)
    validator = (
        CacheValidator(
            embedding_service=embeddings,
            completer=completer,
            sample_rate=settings.validation_sample_rate,
            drift_threshold=settings.validation_drift_threshold,
            metrics=metrics,
        )
        if settings.validation_enabled
        else None
    )
    proxy = ProxyService(
        cache=cache,
        completer=completer,
        default_ttl_seconds=settings.default_ttl_seconds,
        ttl_policy=build_ttl_policy(ttl_classifier),
        threshold_policy=threshold_policy,
        metrics=metrics,
        near_miss_tracker=near_miss_tracker,
        validator=validator,
    )
    app.state.threshold_engine = threshold_engine
    app.state.metrics = metrics
    app.state.near_miss_tracker = near_miss_tracker
    app.state.validator = validator

    app.state.embeddings = embeddings
    app.state.store = store
    app.state.cache = cache
    app.state.proxy = proxy

    logger.info("service.startup", env=settings.app_env, version=__version__,
                backend=settings.vector_backend)
    try:
        yield
    finally:
        # Graceful shutdown: drain upstream/HTTP clients and the datastore.
        logger.info("service.shutdown.begin")
        if hasattr(completer, "aclose"):
            await completer.aclose()
        await embeddings.aclose()
        await store.aclose()
        logger.info("service.shutdown.complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Construct and configure the FastAPI application."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_logs=settings.is_production)

    app = FastAPI(
        title="LLM-Semantic-Caching-Layer",
        version=__version__,
        summary="OpenAI-compatible semantic caching proxy for LLM APIs.",
        lifespan=lifespan,
    )
    app.state.settings = settings

    # Middleware is applied bottom-up; add rate limiting first so correlation-id
    # logging wraps (and tags) even rejected requests.
    if settings.rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware,
            rate_per_sec=settings.rate_limit_rps,
            exempt_paths=("/healthz", "/readyz", "/metrics"),
        )
    app.add_middleware(CorrelationIdMiddleware)

    @app.exception_handler(ProviderError)
    async def _provider_error_handler(_request, exc: ProviderError) -> JSONResponse:
        # Map upstream failures to a clean, OpenAI-style error envelope. 5xx
        # provider errors are reported as 502 (bad gateway); 4xx pass through.
        status = exc.status_code if exc.status_code < 500 else 502
        return JSONResponse(
            {"error": {"message": str(exc), "type": "provider_error", "provider": exc.provider}},
            status_code=status,
        )

    app.include_router(chat_router)
    app.include_router(cache_router)
    app.include_router(analytics_router)
    app.include_router(metrics_router)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {"service": settings.service_name, "version": __version__}

    @app.get("/healthz", tags=["health"])
    async def healthz() -> dict[str, str]:
        """Liveness probe — the process is up and serving."""
        return {"status": "ok"}

    @app.get("/readyz", tags=["health"])
    async def readyz() -> JSONResponse:
        """Readiness probe — dependencies (vector store) are reachable."""
        store = getattr(app.state, "store", None)
        if store is None:
            return JSONResponse({"status": "starting"}, status_code=503)
        try:
            await store.count()
        except Exception as exc:  # noqa: BLE001
            logger.warning("readiness.failed", error=str(exc))
            return JSONResponse({"status": "unavailable"}, status_code=503)
        return JSONResponse({"status": "ready"})

    return app


app = create_app()
