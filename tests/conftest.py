"""Shared pytest fixtures.

These fixtures keep tests hermetic: configuration is overridden to a ``test``
environment and external dependencies (Redis, providers) are faked so unit
tests run without network or a live datastore.
"""

from __future__ import annotations

import os

import pytest

# Ensure tests never accidentally read a developer's real .env / credentials.
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("EMBEDDING_PROVIDER", "fake")


@pytest.fixture
def settings():
    from app.core.config import Settings

    return Settings(
        app_env="test",
        embedding_provider="fake",
        vector_backend="memory",
        completer_backend="echo",
        validation_sample_rate=0.0,
        similarity_threshold=0.95,
        redis_url="redis://localhost:6379/15",
    )


@pytest.fixture
def app(settings):
    from app.main import create_app

    return create_app(settings)


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client
