# Roadmap

The service is built in phases. Each phase is independently shippable.

- **Phase 1 — Semantic Cache Engine**: embedding pipeline, Redis vector storage,
  cosine similarity search, cache-safe namespacing.
- **Phase 2 — OpenAI-Compatible Proxy**: `/v1/chat/completions`, provider
  routing (OpenAI / Anthropic / Ollama), streaming.
- **Phase 3 — Cache Policies**: TTL classification, invalidation APIs,
  threshold analytics, adaptive thresholds.
- **Phase 4 — Observability**: Prometheus metrics, Grafana dashboards,
  near-miss analyzer, cache validation (shadow replay).
- **Phase 5 — Containerization & Load Testing**: Docker Compose stack, Locust
  benchmarks.
- **Phase 6 — Production Hardening**: structured logging, correlation IDs,
  retries, graceful shutdown, rate limiting, request tracing.
