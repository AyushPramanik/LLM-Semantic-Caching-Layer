"""Locust load test for the semantic caching layer.

Drives a realistic mix of repeated / semantically-similar / unique prompts at
``POST /v1/chat/completions`` and reports cache effectiveness from the
``X-Cache-Status`` header, including hit-rate convergence as the cache warms.

Usage:
    locust -f loadtests/locustfile.py --host http://localhost:8000
    # headless, ~2000 requests:
    locust -f loadtests/locustfile.py --host http://localhost:8000 \
        --headless -u 50 -r 10 -t 1m --only-summary
"""

from __future__ import annotations

import random

from locust import HttpUser, between, events, task

from loadtests.prompts import repeated_prompt, similar_prompt, unique_prompt

_rng = random.Random(1337)


class _Stats:
    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


STATS = _Stats()


def _post(client, prompt: str, name: str) -> None:
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    }
    with client.post("/v1/chat/completions", json=payload, name=name, catch_response=True) as resp:
        status = resp.headers.get("X-Cache-Status", "MISS")
        if status == "HIT":
            STATS.hits += 1
        else:
            STATS.misses += 1
        if resp.status_code != 200:
            resp.failure(f"status {resp.status_code}")


class CacheUser(HttpUser):
    wait_time = between(0.05, 0.25)

    @task(5)
    def repeated(self) -> None:
        _post(self.client, repeated_prompt(_rng), "repeated")

    @task(3)
    def similar(self) -> None:
        _post(self.client, similar_prompt(_rng), "similar")

    @task(2)
    def unique(self) -> None:
        _post(self.client, unique_prompt(_rng), "unique")


@events.quitting.add_listener
def _report(environment, **_kwargs) -> None:
    print("\n=== Semantic Cache Load Summary ===")
    print(f"requests:  {STATS.total}")
    print(f"hits:      {STATS.hits}")
    print(f"misses:    {STATS.misses}")
    print(f"hit_rate:  {STATS.hit_rate:.1%}")
