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

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from app.cache.namespace import RequestSignature
from app.cache.semantic_cache import CacheStatus, SemanticCache
from app.core.logging import get_logger
from app.models.chat import ChatCompletionRequest, ChatCompletionResponse
from app.monitoring.metrics import MetricsSink, NoOpMetrics
from app.proxy.streaming import StreamAssembler, response_to_sse

logger = get_logger(__name__)


class Completer(Protocol):
    """Anything that can turn a chat request into a completion (a provider).

    ``resolve_provider`` lets a router report which concrete provider will serve
    a given request, so the cache namespace reflects the real upstream.
    """

    def resolve_provider(self, request: ChatCompletionRequest) -> str: ...

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse: ...

    def stream(self, request: ChatCompletionRequest) -> AsyncIterator[bytes]: ...


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


@dataclass(slots=True)
class StreamResult:
    """A streaming response: cache headers plus the SSE byte generator."""

    headers: dict[str, str]
    body: AsyncIterator[bytes]
    cache_status: CacheStatus


class ProxyService:
    def __init__(
        self,
        *,
        cache: SemanticCache,
        completer: Completer,
        default_ttl_seconds: int = 86_400,
        ttl_policy=None,
        threshold_policy=None,
        metrics: MetricsSink | None = None,
        near_miss_tracker=None,
    ) -> None:
        self._cache = cache
        self._completer = completer
        self._default_ttl = default_ttl_seconds
        self._ttl_policy = ttl_policy
        self._threshold_policy = threshold_policy
        self._metrics: MetricsSink = metrics or NoOpMetrics()
        self._near_miss = near_miss_tracker

    def _record_near_miss(self, lookup, signature, prompt: str, threshold: float | None) -> None:
        if self._near_miss is not None and lookup.near_miss:
            self._near_miss.record(
                score=lookup.score,
                threshold=threshold if threshold is not None else self._cache.default_threshold,
                namespace=signature.namespace(),
                prompt=prompt,
            )

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
        self._metrics.record_lookup(
            lookup.status.value, provider, request.model, lookup.score, lookup.latency_ms
        )
        self._record_near_miss(lookup, signature, prompt, threshold)

        if lookup.is_hit and lookup.match is not None:
            cached = ChatCompletionResponse.model_validate(lookup.match.entry.response)
            self._metrics.record_savings(
                request.model, cached.usage.prompt_tokens, cached.usage.completion_tokens
            )
            return ProxyResult(
                response=cached,
                cache_status=CacheStatus.HIT,
                similarity_score=lookup.score,
                cache_latency_ms=lookup.latency_ms,
                provider=provider,
            )

        # MISS — forward upstream and cache the successful response.
        started = time.perf_counter()
        response = await self._completer.complete(request)
        self._metrics.record_provider_latency(provider, (time.perf_counter() - started) * 1000.0)
        ttl_seconds, tags = self._resolve_policy(request, response)
        if ttl_seconds > 0:
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

    async def stream(
        self, request: ChatCompletionRequest, *, tenant: str = "default"
    ) -> StreamResult:
        """Streaming variant of :meth:`complete`.

        On HIT, the cached response is re-emitted as SSE immediately. On MISS the
        upstream stream is forwarded chunk-by-chunk while being reassembled; the
        full response is cached only if the stream completes successfully. Partial
        or failed generations are never cached.
        """
        provider = self._completer.resolve_provider(request)
        signature = self._signature(request, tenant, provider)
        prompt = request.latest_user_prompt()
        threshold = self._threshold_policy(request) if self._threshold_policy else None
        lookup = await self._cache.lookup(signature, prompt, threshold=threshold)
        self._metrics.record_lookup(
            lookup.status.value, provider, request.model, lookup.score, lookup.latency_ms
        )
        self._record_near_miss(lookup, signature, prompt, threshold)

        if lookup.is_hit and lookup.match is not None:
            cached = ChatCompletionResponse.model_validate(lookup.match.entry.response)
            self._metrics.record_savings(
                request.model, cached.usage.prompt_tokens, cached.usage.completion_tokens
            )

            async def hit_body() -> AsyncIterator[bytes]:
                for chunk in response_to_sse(cached):
                    yield chunk

            return StreamResult(
                headers=self._headers(CacheStatus.HIT, lookup.score, lookup.latency_ms, provider),
                body=hit_body(),
                cache_status=CacheStatus.HIT,
            )

        async def miss_body() -> AsyncIterator[bytes]:
            assembler = StreamAssembler(request.model)
            try:
                async for chunk in self._completer.stream(request):
                    assembler.push(chunk)
                    yield chunk
            except Exception:
                logger.warning("stream.failed", provider=provider, model=request.model)
                raise  # never cache a failed/partial generation
            if not assembler.completed:
                logger.warning("stream.incomplete", provider=provider)
                return
            response = assembler.build_response()
            ttl_seconds, tags = self._resolve_policy(request, response)
            if ttl_seconds > 0:
                await self._cache.store_response(
                    signature,
                    prompt,
                    response.model_dump(),
                    ttl_seconds=ttl_seconds,
                    tags=tags,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                )

        return StreamResult(
            headers=self._headers(CacheStatus.MISS, lookup.score, lookup.latency_ms, provider),
            body=miss_body(),
            cache_status=CacheStatus.MISS,
        )

    @staticmethod
    def _headers(
        status: CacheStatus, score: float, latency_ms: float, provider: str
    ) -> dict[str, str]:
        return {
            "X-Cache-Status": status.value,
            "X-Similarity-Score": f"{score:.4f}",
            "X-Cache-Latency": f"{latency_ms:.3f}ms",
            "X-Provider": provider,
        }

    def _resolve_policy(
        self, request: ChatCompletionRequest, response: ChatCompletionResponse
    ) -> tuple[int, list[str]]:
        if self._ttl_policy is not None:
            return self._ttl_policy(request, response)
        tags = list(request.cache_tags or [])
        return self._default_ttl, tags
