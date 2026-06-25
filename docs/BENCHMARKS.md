# Benchmarks

## Headline

> Across a **2,000-request** workload (25% unique / 75% repeated-or-similar),
> the caching layer cut **simulated LLM API cost by 73.8%** and **P50 latency by
> 99.3%**, with a **73.75% cache hit rate**. P95 latency dropped 15.6% — the tail
> is dominated by the unavoidable cache misses that still hit the provider.

Reproduce:

```bash
PYTHONPATH=. python scripts/benchmark.py --requests 2000 --seed 1337
```

## Results (seed 1337)

| Metric | Without cache | With cache | Δ |
|--------|---------------|------------|---|
| Requests | 2,000 | 2,000 | — |
| Provider calls | 2,000 | 525 | **−73.8%** |
| Est. cost (USD) | $4.20 | $1.10 | **−73.8%** |
| Hit rate | — | 73.75% | — |
| Tokens saved | — | ~1.03M | — |
| Latency P50 | 649 ms | 4.5 ms | **−99.3%** |
| Latency P95 | 979 ms | 827 ms | **−15.6%** |
| Latency P99 | 1126 ms | 982 ms | −12.8% |

## Methodology

* Workload mix: 25% unique (always miss), 75% drawn from a small set of base
  prompts (repeats → hits). This models a typical assistant/RAG endpoint where a
  long head of common questions dominates traffic.
* The hit/miss pattern is produced by replaying the workload through the **real**
  proxy + cache code path (`ProxyService` + `SemanticCache`).
* Business impact is modeled by attributing a realistic upstream latency
  (~650 ms ± 200) and per-request cost (~$0.0021) to provider calls, and the
  measured lookup cost (~4 ms) to hits. Cost reduction tracks the share of
  requests served from cache.
* Deterministic (seeded) so the numbers above are reproducible.

### Why P95 moves less than P50

At a 73.75% hit rate, roughly the top quartile of requests are misses that must
still call the provider. The median request is a near-instant hit (P50 collapses
to ~4 ms), but the 95th-percentile request is, by definition, in the miss tail —
so it improves only via variance, not by being served from cache. Raising the
hit rate (more repeated traffic, a slightly lower threshold informed by
`/analytics/near-misses`) shifts the tail too.

## Cache warmup / convergence

`tests/test_load_convergence.py` asserts the cache **warms up**: the hit rate in
the final 200-request window is materially higher than the first 200 (cold
start), and steady-state exceeds 60% for the modeled mix. All-unique traffic
never converges (0% hit rate), confirming there are no false hits.

## Live load testing

For an end-to-end run against the Dockerized stack:

```bash
docker compose up -d
scripts/run_loadtest.sh http://localhost:8000   # ~2000+ requests, CSV + HTML
```

Hit rate, latency, and cost-savings panels are visible in Grafana
(`http://localhost:3000`, dashboard **Semantic Cache → Overview**).
