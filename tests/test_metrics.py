"""Tests for Prometheus instrumentation."""

from __future__ import annotations

from prometheus_client import CollectorRegistry

from app.monitoring.metrics import CacheMetrics


def test_lookup_updates_hit_rate():
    m = CacheMetrics(registry=CollectorRegistry())
    m.record_lookup("HIT", "openai", "gpt-4o-mini", 0.99, 3.1)
    m.record_lookup("MISS", "openai", "gpt-4o-mini", 0.40, 5.0)
    assert m.hit_rate._value.get() == 0.5


def test_savings_estimate_is_priced_per_model():
    m = CacheMetrics(registry=CollectorRegistry())
    m.record_savings("gpt-4o", prompt_tokens=1000, completion_tokens=1000)
    # 1k input @ 0.0025 + 1k output @ 0.01 = 0.0125
    assert round(m.cost_saved._value.get(), 6) == 0.0125
    assert m.tokens_saved._value.get() == 2000


def test_unknown_model_uses_fallback_pricing():
    m = CacheMetrics(registry=CollectorRegistry())
    m.record_savings("mystery-model", prompt_tokens=1000, completion_tokens=0)
    assert m.cost_saved._value.get() > 0


def test_render_exposes_expected_metric_names():
    m = CacheMetrics(registry=CollectorRegistry())
    m.record_lookup("HIT", "openai", "gpt-4o-mini", 0.99, 3.1)
    output = m.render().decode()
    for name in (
        "cache_hits_total",
        "cache_misses_total",
        "cache_hit_rate",
        "cache_lookup_latency_ms",
        "similarity_distribution",
        "estimated_cost_saved_usd",
        "tokens_saved",
        "requests_by_provider",
    ):
        assert name in output


def test_metrics_endpoint_exposes_counters(client):
    client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
    )
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "cache_misses_total" in resp.text
    assert "requests_by_model" in resp.text
