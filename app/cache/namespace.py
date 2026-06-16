"""Deterministic cache namespacing for safety and isolation.

Two requests must **never** share a cache entry if they differ in any dimension
that can change the model's output: system prompt, model, temperature,
max_tokens, or provider. We also fold in an optional tenant/application id so
different applications pointed at the same proxy cannot read each other's data.

The namespace is a stable hash of those dimensions, so it is reproducible across
processes and safe to use as a Redis TAG.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


def hash_system_prompt(system_prompt: str | None) -> str:
    """Stable short hash of a system prompt (empty prompt -> sentinel)."""
    normalized = (system_prompt or "").strip()
    if not normalized:
        return "no-system"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class RequestSignature:
    """The set of dimensions that scope a cache entry."""

    model: str
    provider: str
    system_prompt: str | None = None
    temperature: float = 1.0
    max_tokens: int | None = None
    tenant: str = "default"

    @property
    def system_prompt_hash(self) -> str:
        return hash_system_prompt(self.system_prompt)

    def namespace(self) -> str:
        """Deterministic namespace key for this signature.

        Temperature is bucketed to one decimal place: tiny floating-point
        differences shouldn't fragment the cache, but materially different
        sampling temperatures should.
        """
        temp_bucket = f"{round(self.temperature, 1):.1f}"
        parts = [
            self.tenant,
            self.provider,
            self.model,
            self.system_prompt_hash,
            temp_bucket,
            str(self.max_tokens if self.max_tokens is not None else "default"),
        ]
        raw = "|".join(parts)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
        # Human-readable prefix + digest aids debugging in redis-cli.
        return f"{self.tenant}:{self.provider}:{self.model}:{digest}"
