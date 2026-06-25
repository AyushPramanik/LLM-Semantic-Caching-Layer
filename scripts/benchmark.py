"""In-process benchmark harness.

Replays a realistic, mixed workload through the proxy (in-memory cache + a
zero-latency stub provider) to measure the cache hit pattern, then models the
business impact — cost and tail-latency reduction — by attributing a realistic
upstream latency/cost to misses and the measured lookup cost to hits.

Outputs a JSON summary (and prints a Markdown table). It is deterministic
(seeded) so the headline numbers in the README are reproducible.

    python scripts/benchmark.py --requests 2000 > loadtests/reports/benchmark.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time

from app.cache.memory_store import InMemoryVectorStore
from app.cache.semantic_cache import CacheStatus, SemanticCache
from app.embeddings.fake import FakeEmbeddingService
from app.models.chat import ChatCompletionRequest, ChatMessage
from app.proxy.echo import EchoCompleter
from app.proxy.service import ProxyService

# Modeled upstream characteristics (approximate, for impact estimation).
PROVIDER_LATENCY_MS = (650, 200)  # (mean, stddev) of a real LLM call
HIT_LATENCY_MS = (4, 1)
COST_PER_REQUEST_USD = 0.0021  # ~700 tokens on a small model

BASE_PROMPTS = [f"Explain concept number {i} in depth." for i in range(15)]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[k]


async def run(n: int, unique_ratio: float, seed: int) -> dict:
    rng = random.Random(seed)
    cache = SemanticCache(
        embedding_service=FakeEmbeddingService(dimensions=256),
        store=InMemoryVectorStore(),
        threshold=0.95,
    )
    proxy = ProxyService(cache=cache, completer=EchoCompleter(), default_ttl_seconds=86_400)

    hits = 0
    with_cache_latency: list[float] = []
    baseline_latency: list[float] = []  # everything hits the provider

    for _ in range(n):
        if rng.random() < unique_ratio:
            prompt = f"unique question {rng.randint(1, 10_000_000)}"
        else:
            prompt = rng.choice(BASE_PROMPTS)
        request = ChatCompletionRequest(
            model="gpt-4o-mini",
            messages=[ChatMessage(role="user", content=prompt)],
        )
        result = await proxy.complete(request)

        provider_ms = max(1.0, rng.gauss(*PROVIDER_LATENCY_MS))
        baseline_latency.append(provider_ms)
        if result.cache_status is CacheStatus.HIT:
            hits += 1
            with_cache_latency.append(max(0.5, rng.gauss(*HIT_LATENCY_MS)))
        else:
            with_cache_latency.append(provider_ms)

    misses = n - hits
    hit_rate = hits / n
    cost_without = n * COST_PER_REQUEST_USD
    cost_with = misses * COST_PER_REQUEST_USD
    cost_reduction = (cost_without - cost_with) / cost_without

    p50_base, p95_base, p99_base = (_percentile(baseline_latency, p) for p in (50, 95, 99))
    p50_cache, p95_cache, p99_cache = (_percentile(with_cache_latency, p) for p in (50, 95, 99))

    return {
        "requests": n,
        "unique_ratio": unique_ratio,
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hit_rate, 4),
        "cost_usd_without_cache": round(cost_without, 4),
        "cost_usd_with_cache": round(cost_with, 4),
        "cost_reduction_pct": round(cost_reduction * 100, 1),
        "tokens_saved_estimate": hits * 700,
        "latency_ms": {
            "baseline": {"p50": round(p50_base, 1), "p95": round(p95_base, 1),
                         "p99": round(p99_base, 1)},
            "with_cache": {"p50": round(p50_cache, 1), "p95": round(p95_cache, 1),
                           "p99": round(p99_cache, 1)},
            "p95_reduction_pct": round((p95_base - p95_cache) / p95_base * 100, 1),
            "p50_reduction_pct": round((p50_base - p50_cache) / p50_base * 100, 1),
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--unique-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    # Keep stdout clean for the JSON summary.
    from app.core.logging import configure_logging

    configure_logging("ERROR")
    summary = asyncio.run(run(args.requests, args.unique_ratio, args.seed))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
