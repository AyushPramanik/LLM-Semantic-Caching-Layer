"""Proxy service: the request flow behind ``/v1/chat/completions``.

Responsibilities:
  1. Build a cache-safe :class:`RequestSignature` from the request.
  2. Look up a semantically similar cached response.
  3. On HIT, return the cached payload immediately.
  4. On MISS, forward upstream via a :class:`Completer`, then cache the
     successful response with a policy-derived TTL.

The TTL policy and adaptive threshold are injected as optional callables so this
service stays focused and the policy layers (Phase 3) can evolve independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.cache.namespace import RequestSignature
from app.cache.semantic_cache import CacheStatus, SemanticCache
from app.core.logging import get_logger
from app.models.chat import ChatCompletionRequest, ChatCompletionResponse

logger = get_logger(__name__)


class Completer(Protocol):
    """Anything that can turn a chat request into a completion (a provider).

    ``resolve_provider`` lets a router report which concrete provider will serve
    a given request, so the cache namespace reflects the real upstream.
    """

    def resolve_provider(self, request: ChatCompletionRequest) -> str: ...

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse: ...


# Optional policy hooks. Defaults keep behavior simple until Phase 3 wires real
# implementations in.
TtlPolicy = "Callable[[ChatCompletionRequest, ChatCompletionResponse], tuple[int, list[str]]]"
ThresholdPolicy = "Callable[[ChatCompletionRequest], float | None]"


@dataclass(slots=True)
class ProxyResult:
    """The completion plus cache metadata surfaced as response headers."""

    response: ChatCompletionResponse
    cache_status: CacheStatus
    similarity_score: float
    cache_latency_ms: float
    provider: str


class ProxyService:
    def __init__(
        self,
        *,
        cache: SemanticCache,
        completer: Completer,
        default_ttl_seconds: int = 86_400,
        ttl_policy=None,
        threshold_policy=None,
    ) -> None:
        self._cache = cache
        self._completer = completer
        self._default_ttl = default_ttl_seconds
        self._ttl_policy = ttl_policy
        self._threshold_policy = threshold_policy

    def _signature(
        self, request: ChatCompletionRequest, tenant: str, provider: str
    ) -> RequestSignature:
        return RequestSignature(
            model=request.model,
            provider=provider,
            system_prompt=request.system_prompt(),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tenant=tenant,
        )

    async def complete(
        self, request: ChatCompletionRequest, *, tenant: str = "default"
    ) -> ProxyResult:
        provider = self._completer.resolve_provider(request)
        signature = self._signature(request, tenant, provider)
        prompt = request.latest_user_prompt()
        threshold = self._threshold_policy(request) if self._threshold_policy else None

        lookup = await self._cache.lookup(signature, prompt, threshold=threshold)

        if lookup.is_hit and lookup.match is not None:
            cached = ChatCompletionResponse.model_validate(lookup.match.entry.response)
            return ProxyResult(
                response=cached,
                cache_status=CacheStatus.HIT,
                similarity_score=lookup.score,
                cache_latency_ms=lookup.latency_ms,
                provider=provider,
            )

        # MISS — forward upstream and cache the successful response.
        response = await self._completer.complete(request)
        ttl_seconds, tags = self._resolve_policy(request, response)
        await self._cache.store_response(
            signature,
            prompt,
            response.model_dump(),
            ttl_seconds=ttl_seconds,
            tags=tags,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )
        return ProxyResult(
            response=response,
            cache_status=CacheStatus.MISS,
            similarity_score=lookup.score,
            cache_latency_ms=lookup.latency_ms,
            provider=provider,
        )

    def _resolve_policy(
        self, request: ChatCompletionRequest, response: ChatCompletionResponse
    ) -> tuple[int, list[str]]:
        if self._ttl_policy is not None:
            return self._ttl_policy(request, response)
        tags = list(request.cache_tags or [])
        return self._default_ttl, tags
