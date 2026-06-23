"""Prometheus metrics for the semantic cache.

A :class:`CacheMetrics` instance owns all collectors against a given registry, so
the application can use the global registry while tests use a throwaway one
(avoiding duplicate-timeseries errors). The proxy records outcomes through the
:class:`MetricsSink` protocol; :class:`NoOpMetrics` is the inert default.

Cost/token savings are estimated from a small per-model pricing table: on a cache
HIT we credit the tokens (and dollars) we would otherwise have paid to
regenerate the cached completion.
"""

from __future__ import annotations

from typing import Protocol

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# USD per 1K tokens (input, output). Approximate list prices for estimation.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "claude-3-5-sonnet": (0.003, 0.015),
    "claude-3-opus": (0.015, 0.075),
    "claude-3-haiku": (0.00025, 0.00125),
}
_FALLBACK_PRICE = (0.0005, 0.0015)


class MetricsSink(Protocol):
    def record_lookup(self, status: str, provider: str, model: str,
                      similarity: float, latency_ms: float) -> None: ...
    def record_provider_latency(self, provider: str, latency_ms: float) -> None: ...
    def record_savings(self, model: str, prompt_tokens: int, completion_tokens: int) -> None: ...
    def record_eviction(self, count: int = 1) -> None: ...
    def set_cache_size(self, size: int) -> None: ...
    def record_validation(self, accuracy: float, drift_rate: float,
                          false_hit_rate: float) -> None: ...


class NoOpMetrics:
    """Inert sink used when metrics are disabled or in unit tests."""

    def record_lookup(self, *args, **kwargs) -> None: ...
    def record_provider_latency(self, *args, **kwargs) -> None: ...
    def record_savings(self, *args, **kwargs) -> None: ...
    def record_eviction(self, *args, **kwargs) -> None: ...
    def set_cache_size(self, *args, **kwargs) -> None: ...
    def record_validation(self, *args, **kwargs) -> None: ...


_LATENCY_BUCKETS = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500)
_SIMILARITY_BUCKETS = (0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.98, 0.99, 1.0)


class CacheMetrics:
    def __init__(
        self,
        registry: CollectorRegistry | None = None,
        pricing: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.registry = registry or CollectorRegistry()
        self._pricing = pricing or DEFAULT_PRICING

        self.cache_hits = Counter(
            "cache_hits_total", "Total cache hits", ["provider", "model"], registry=self.registry)
        self.cache_misses = Counter(
            "cache_misses_total", "Total cache misses", ["provider", "model"], registry=self.registry)
        self.hit_rate = Gauge(
            "cache_hit_rate", "Rolling cache hit rate", registry=self.registry)
        self.lookup_latency = Histogram(
            "cache_lookup_latency_ms", "Cache lookup latency (ms)",
            buckets=_LATENCY_BUCKETS, registry=self.registry)
        self.provider_latency = Histogram(
            "provider_latency_ms", "Upstream provider latency (ms)", ["provider"],
            buckets=_LATENCY_BUCKETS, registry=self.registry)
        self.cache_size = Gauge(
            "cache_size", "Number of entries in the cache", registry=self.registry)
        self.evictions = Counter(
            "evictions_total", "Total cache evictions", registry=self.registry)
        self.similarity = Histogram(
            "similarity_distribution", "Distribution of best similarity scores",
            buckets=_SIMILARITY_BUCKETS, registry=self.registry)
        self.cost_saved = Counter(
            "estimated_cost_saved_usd", "Estimated USD saved by cache hits", registry=self.registry)
        self.tokens_saved = Counter(
            "tokens_saved", "Tokens not regenerated thanks to cache hits", registry=self.registry)
        self.requests_by_provider = Counter(
            "requests_by_provider", "Requests grouped by provider", ["provider"],
            registry=self.registry)
        self.requests_by_model = Counter(
            "requests_by_model", "Requests grouped by model", ["model"], registry=self.registry)
        self.validation_accuracy = Gauge(
            "cache_validation_accuracy", "Share of validated hits that still match",
            registry=self.registry)
        self.semantic_drift_rate = Gauge(
            "semantic_drift_rate", "Share of validated hits showing semantic drift",
            registry=self.registry)
        self.false_hit_rate = Gauge(
            "false_hit_rate", "Share of validated hits that were effectively wrong",
            registry=self.registry)

        self._hits = 0
        self._total = 0

    def record_lookup(
        self, status: str, provider: str, model: str, similarity: float, latency_ms: float
    ) -> None:
        self.requests_by_provider.labels(provider=provider).inc()
        self.requests_by_model.labels(model=model).inc()
        self.lookup_latency.observe(latency_ms)
        self.similarity.observe(max(0.0, similarity))
        self._total += 1
        if status == "HIT":
            self.cache_hits.labels(provider=provider, model=model).inc()
            self._hits += 1
        else:
            self.cache_misses.labels(provider=provider, model=model).inc()
        self.hit_rate.set(self._hits / self._total if self._total else 0.0)

    def record_provider_latency(self, provider: str, latency_ms: float) -> None:
        self.provider_latency.labels(provider=provider).observe(latency_ms)

    def record_savings(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        input_price, output_price = self._pricing.get(model, _FALLBACK_PRICE)
        dollars = (prompt_tokens / 1000) * input_price + (completion_tokens / 1000) * output_price
        self.cost_saved.inc(dollars)
        self.tokens_saved.inc(prompt_tokens + completion_tokens)

    def record_eviction(self, count: int = 1) -> None:
        self.evictions.inc(count)

    def set_cache_size(self, size: int) -> None:
        self.cache_size.set(size)

    def record_validation(
        self, accuracy: float, drift_rate: float, false_hit_rate: float
    ) -> None:
        self.validation_accuracy.set(accuracy)
        self.semantic_drift_rate.set(drift_rate)
        self.false_hit_rate.set(false_hit_rate)

    def render(self) -> bytes:
        return generate_latest(self.registry)
