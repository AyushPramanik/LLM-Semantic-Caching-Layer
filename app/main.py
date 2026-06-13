"""Application entrypoint and FastAPI factory.

This module wires together configuration, logging, and routing. Heavier
subsystems (Redis vector store, providers, metrics) are attached in later
phases via the application lifespan.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup/shutdown of long-lived resources."""
    settings: Settings = app.state.settings
    logger.info("service.startup", env=settings.app_env, version=__version__)
    yield
    logger.info("service.shutdown")


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

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {"service": settings.service_name, "version": __version__}

    @app.get("/healthz", tags=["health"])
    async def healthz() -> dict[str, str]:
        """Liveness probe — the process is up and serving."""
        return {"status": "ok"}

    return app


app = create_app()
