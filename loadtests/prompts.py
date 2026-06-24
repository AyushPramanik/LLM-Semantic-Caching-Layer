"""Prompt corpus for load testing.

Provides three classes of traffic so we can observe realistic cache behavior:

* **repeated**  — exact repeats of a base prompt (should become HITs fast).
* **similar**   — paraphrases of a base prompt (exercise semantic matching).
* **unique**    — never-seen prompts (always MISS; model the long tail).
"""

from __future__ import annotations

import random

BASE_PROMPTS: list[str] = [
    "What is the time complexity of binary search?",
    "Explain how TCP differs from UDP.",
    "What is a database index and why does it help?",
    "How does HTTPS keep traffic secure?",
    "What is the difference between a process and a thread?",
    "Explain the CAP theorem in simple terms.",
    "How does garbage collection work in Python?",
    "What is a deadlock and how do you avoid it?",
    "Explain what a hash map is and its average complexity.",
    "What is the difference between SQL and NoSQL databases?",
    "How does DNS resolution work?",
    "What is eventual consistency?",
    "Explain the difference between latency and throughput.",
    "What is a load balancer and what algorithms does it use?",
    "How does a bloom filter work?",
]

PARAPHRASES: dict[str, list[str]] = {
    "What is the time complexity of binary search?": [
        "Can you tell me the big-O complexity of binary search?",
        "How fast is binary search in terms of time complexity?",
    ],
    "Explain how TCP differs from UDP.": [
        "What are the differences between TCP and UDP?",
        "Compare TCP versus UDP for me.",
    ],
    "How does DNS resolution work?": [
        "Walk me through how DNS lookups resolve a hostname.",
        "Explain the DNS resolution process.",
    ],
    "What is a database index and why does it help?": [
        "Why do database indexes speed up queries?",
        "Explain the purpose of an index in a database.",
    ],
}

_UNIQUE_TOPICS = [
    "quantum computing", "the Roman aqueducts", "octopus cognition",
    "the history of jazz", "supernova formation", "medieval trade routes",
    "coral reef ecosystems", "the printing press", "volcanic soil fertility",
]


def repeated_prompt(rng: random.Random) -> str:
    return rng.choice(BASE_PROMPTS)


def similar_prompt(rng: random.Random) -> str:
    base = rng.choice(list(PARAPHRASES.keys()))
    return rng.choice(PARAPHRASES[base])


def unique_prompt(rng: random.Random) -> str:
    topic = rng.choice(_UNIQUE_TOPICS)
    nonce = rng.randint(1, 10_000_000)
    return f"Give me an obscure fact about {topic} (ref {nonce})."
