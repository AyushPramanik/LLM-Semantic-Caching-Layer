# LLM-Semantic-Caching-Layer

**A drop-in, OpenAI-compatible semantic caching proxy that cut simulated LLM API
costs by 73.8% and P50 latency by 99.3% (P95 by 15.6%) across a 2,000-request
workload — by serving cached responses for *semantically similar* prompts.**

Point your client at the proxy by changing a single setting:

```diff
- OPENAI_BASE_URL=https://api.openai.com/v1
+ OPENAI_BASE_URL=http://localhost:8000/v1
```

No application code changes are required. Same request schema, same response
schema, plus cache metadata in response headers.

---

## Why

LLM calls are slow and expensive, and a large share of production traffic is
**near-duplicate**: the same questions, lightly reworded. An exact-match cache
misses all of it. This service embeds each prompt, runs a cosine **KNN** search
over a Redis Stack vector index, and returns a cached answer when similarity
clears a (configurable, adaptive) threshold — turning that near-duplicate traffic
into sub-10ms, $0 responses.

| | |
|---|---|
| **Cost reduction** | **73.8%** (2,000-request mixed workload) |
| **Cache hit rate** | **73.75%** |
| **P50 latency** | 649 ms → **4.5 ms** (−99.3%) |
| **P95 latency** | 979 ms → **827 ms** (−15.6%) |
| **Tokens saved** | ~1.03M |

See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for methodology and reproduction.

## Features

- **OpenAI-compatible** `POST /v1/chat/completions` (buffered **and** streaming).
- **Provider-agnostic routing** (strategy pattern): `gpt-*` → OpenAI,
  `claude-*` → Anthropic, `llama-*`/`mistral-*` → Ollama.
- **Cache-safe namespacing**: entries never shared across tenant, model, system
  prompt, temperature, or `max_tokens`.
- **Freshness-aware TTLs**: long for stable knowledge, short for time-sensitive
  topics, no-cache for live/volatile prompts.
- **Adaptive thresholds** per request category (classification / programming /
  creative), tunable from feedback.
- **Streaming with a never-cache-partial guarantee.**
- **Cache validation (shadow replay)** — re-issues a sampled fraction of hits to
  the provider to measure accuracy, semantic drift, and false-hit rate. *(The
  feature I'm proudest of: it's how you trust a cache in production.)*
- **Near-miss analyzer** with concrete threshold-tuning recommendations.
- **Full observability**: Prometheus metrics + provisioned Grafana dashboards.
- **Production hardening**: structured logging, correlation IDs, retries,
  rate limiting, graceful shutdown, health/readiness probes.

## Architecture

```
client ──► FastAPI proxy ──► embed ──► Redis Stack (HNSW cosine KNN)
                │                          │
              HIT ◄── cached response ─────┘
                │
              MISS ──► provider router ──► OpenAI / Anthropic / Ollama
                          └──► cache response (TTL by freshness)
   metrics ──► Prometheus ──► Grafana        shadow replay ──► validation
```

Full diagram and component breakdown: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Quick start

```bash
cp .env.example .env          # add OPENAI_API_KEY (and others as needed)
docker compose up -d          # app + Redis Stack + Prometheus + Grafana
curl localhost:8000/readyz
```

Send a request exactly as you would to OpenAI:

```bash
curl localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"What is HTTP?"}]}' -i
# Response headers include:
#   X-Cache-Status: MISS         (HIT on the next, similar request)
#   X-Similarity-Score: 0.0000
#   X-Cache-Latency: 6.231ms
```

- App: `http://localhost:8000` (`/docs` for OpenAPI)
- Grafana: `http://localhost:3000` (admin/admin) → **Semantic Cache → Overview**
- Prometheus: `http://localhost:9090`

### Local dev (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
EMBEDDING_PROVIDER=fake VECTOR_BACKEND=memory COMPLETER_BACKEND=echo \
  uvicorn app.main:app --reload
pytest                          # 100+ tests
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/chat/completions` | OpenAI-compatible completion (cached) |
| DELETE | `/cache/model/{model}` | Evict entries for a model |
| DELETE | `/cache/system-prompt/{hash}` | Evict by system-prompt hash |
| DELETE | `/cache/tag/{tag}` | Evict by tag |
| DELETE | `/cache/all` | Flush the cache |
| POST | `/analytics/threshold-test` | Sweep thresholds over labeled pairs |
| GET | `/analytics/near-misses` | Near-miss histogram + recommendation |
| GET/POST | `/analytics/thresholds`, `/threshold-feedback` | Inspect/adapt thresholds |
| GET | `/analytics/validation` | Shadow-replay accuracy / drift / false-hit rate |
| GET | `/metrics` | Prometheus exposition |
| GET | `/healthz`, `/readyz` | Liveness / readiness |

## Configuration

All settings are environment-driven and validated at startup
([`app/core/config.py`](app/core/config.py)); see [`.env.example`](.env.example).
Key knobs: `SIMILARITY_THRESHOLD` (0.95), `DEFAULT_TTL_SECONDS`,
`VALIDATION_SAMPLE_RATE`, `RATE_LIMIT_RPS`, `EMBEDDING_PROVIDER`,
`VECTOR_BACKEND`, `COMPLETER_BACKEND`.

## Testing

```bash
pytest --cov=app                # unit + integration + load (in-process)
pytest -m integration           # requires a live Redis Stack
```

Coverage target is 80%+. Unit tests cover embeddings, similarity, namespacing,
routing, TTL/threshold policies, validation, and metrics; integration tests
cover the HTTP surface, streaming, and (against a live Redis) the vector store.

## Scaling & cost analysis

The service is stateless — scale replicas horizontally behind a load balancer,
all sharing one Redis Stack cluster. Cost scales with **misses**, not requests:
at a 74% hit rate you pay for ~26% of calls. The break-even vs. embedding cost is
immediate, since an embedding (`text-embedding-3-small`) is ~100× cheaper than a
chat completion. See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Lessons learned

- **Cache safety is the hard part, not the search.** Getting KNN working is easy;
  guaranteeing two prompts never share an entry across system prompt, model, and
  decoding params (the namespace strategy) is what makes it safe to ship.
- **A cache you don't validate is a liability.** Shadow replay turned "looks
  cheap" into "measurably correct" — false-hit and drift rates are first-class.
- **One threshold doesn't fit all traffic.** Classification tolerates 0.90;
  creative writing needs 0.98+. Per-category adaptive thresholds materially lift
  hit rate without raising false hits.
- **Never cache partial streams.** Buffer-then-cache on stream completion is the
  difference between a fast cache and a corrupt one.

## License

MIT — see [`LICENSE`](LICENSE).
