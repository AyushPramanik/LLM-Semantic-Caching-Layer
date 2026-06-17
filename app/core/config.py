"""Centralized application configuration.

All tunables live here and are sourced from environment variables (or a local
`.env` file). Settings are cached so the object is constructed once per process
and can be cheaply injected into request handlers and services.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Values are validated at startup so misconfiguration fails fast rather than
    surfacing as obscure runtime errors deep in a request path.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Service ---
    app_env: Literal["development", "staging", "production", "test"] = "development"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000
    service_name: str = "semantic-cache-layer"

    # --- Redis ---
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")
    cache_index_name: str = "semantic_cache"
    vector_backend: Literal["redis", "memory"] = "redis"

    # --- Embeddings ---
    embedding_provider: Literal["openai", "fake"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # --- Semantic cache ---
    similarity_threshold: float = 0.95
    default_ttl_seconds: int = 86_400
    near_miss_window: float = 0.05

    # --- Cache validation (shadow replay) ---
    validation_sample_rate: float = 0.02

    # --- Providers ---
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    ollama_base_url: str = "http://localhost:11434/v1"

    # --- Rate limiting ---
    rate_limit_enabled: bool = True
    rate_limit_rps: int = 50

    # --- HTTP client ---
    upstream_timeout_seconds: float = 60.0
    upstream_max_retries: int = 2

    @field_validator("similarity_threshold", "validation_sample_rate", "near_miss_window")
    @classmethod
    def _check_unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"value must be within [0.0, 1.0], got {v}")
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    return Settings()
