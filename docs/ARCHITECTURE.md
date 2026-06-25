# Architecture

`LLM-Semantic-Caching-Layer` is an OpenAI-compatible reverse proxy. It sits
between an application and one or more LLM providers and serves cached responses
for *semantically similar* prompts.

## Request flow

```
                         ┌───────────────────────────────────────────────┐
   client (OpenAI SDK)   │            Semantic Caching Layer              │
   OPENAI_BASE_URL  ───► │                                               │
                         │  ┌─────────────┐   1. embed prompt            │
                         │  │ FastAPI /v1 │──────────────► ┌───────────┐ │
                         │  │  proxy      │                │ Embeddings│ │
                         │  └─────┬───────┘ ◄───vector──── │ (OpenAI)  │ │
                         │        │                        └───────────┘ │
                         │        │ 2. KNN search (namespace-scoped)      │
                         │        ▼                        ┌───────────┐ │
                         │   ┌─────────┐  cosine ≥ thresh? │Redis Stack│ │
                         │   │ Cache   │◄─────────────────►│  (HNSW)   │ │
                         │   └────┬────┘                   └───────────┘ │
                         │   HIT  │  MISS                                 │
                         │   ◄────┘    │ 3. route by model               │
                         │             ▼            ┌──────────────────┐ │
                         │      ┌────────────┐      │ OpenAI / Anthropic│ │
                         │      │  Provider  │─────►│ / Ollama          │ │
                         │      │  Router    │◄─────│ (upstream)        │ │
                         │      └─────┬──────┘      └──────────────────┘ │
                         │            │ 4. cache successful response      │
                         │            ▼  (TTL by freshness policy)        │
                         │        back to Redis                          │
                         └───────────────────────────────────────────────┘
                                    │  metrics              ▲ shadow replay
                                    ▼                       │ (validation)
                            Prometheus ──► Grafana
```

1. **Embed** the latest user prompt (`text-embedding-3-small`, L2-normalized).
2. **Search** the vector index, scoped to a cache-safe **namespace**.
3. Compare the best cosine score to the (possibly adaptive) **threshold**.
   - **HIT** → return the cached response immediately (`X-Cache-Status: HIT`).
   - **MISS** → route to the provider, then cache the successful response with a
     freshness-based TTL.

## Components (clean architecture)

| Layer | Package | Responsibility |
|-------|---------|----------------|
| API | `app/api` | HTTP surface: chat, cache invalidation, analytics, metrics, health |
| Proxy | `app/proxy` | Orchestration, streaming, SSE assembly |
| Cache | `app/cache` | Namespacing, vector store, similarity, lookup workflow |
| Embeddings | `app/embeddings` | Embedding service abstraction (OpenAI / fake) |
| Providers | `app/providers` | Provider adapters + model-based router (strategy) |
| Policies | `app/policies` | TTL classifier, adaptive threshold engine |
| Analytics | `app/analytics` | Threshold tuning, near-miss analyzer, validation |
| Monitoring | `app/monitoring` | Prometheus metrics |
| Core | `app/core` | Config, structured logging, middleware |

Dependencies point inward: API → proxy → cache/providers → embeddings/store.
Every boundary (vector store, embedding service, provider, completer, metrics)
is an interface, so Redis, OpenAI, and the metrics backend can be swapped without
touching business logic. The in-memory store and fake embedder are drop-in test
doubles built on the same interfaces.

## Cache safety & isolation

A cache entry is scoped by a deterministic **namespace** derived from `tenant`,
`provider`, `model`, `system-prompt hash`, bucketed `temperature`, and
`max_tokens`. Two requests can share an entry only if all of these match, which
prevents cross-application contamination and unsafe reuse across decoding
parameters. See [`app/cache/namespace.py`](../app/cache/namespace.py).

## Streaming

On a HIT the stored response is re-emitted as SSE. On a MISS upstream chunks are
forwarded to the client while a `StreamAssembler` reconstructs the full
response; it is cached **only** after the stream completes successfully — partial
or failed generations are never cached.

## Observability

Prometheus metrics cover hit rate, latency histograms, similarity distribution,
cost/tokens saved, and validation accuracy. A shadow-replay **validation** loop
re-issues a sampled fraction of hits to the provider and compares answers to
detect false hits and semantic drift. See [`docs/BENCHMARKS.md`](BENCHMARKS.md).

## Scaling

The service is stateless; all shared state lives in Redis. Scale horizontally
behind a load balancer and point every replica at the same Redis Stack cluster.
The token-bucket rate limiter is per-replica today; for global limits it moves to
a shared Redis counter behind the same middleware interface.
