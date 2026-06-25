"""Cross-cutting HTTP middleware: correlation IDs, request tracing, rate limiting.

* **CorrelationIdMiddleware** assigns every request a correlation id (honoring an
  inbound ``X-Request-ID``/``X-Correlation-ID``), binds it to the logging context
  so all logs for the request are linked, echoes it back as a response header,
  and emits a structured access log with server-side latency.

* **RateLimitMiddleware** applies a per-client token bucket. It protects upstream
  providers and the cache from abusive bursts and returns ``429`` with a
  ``Retry-After`` hint when exhausted.
"""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.logging import get_logger, set_correlation_id

logger = get_logger("http")

_CORRELATION_HEADERS = ("x-correlation-id", "x-request-id")


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        correlation_id = None
        for header in _CORRELATION_HEADERS:
            if header in request.headers:
                correlation_id = request.headers[header]
                break
        correlation_id = correlation_id or uuid.uuid4().hex
        set_correlation_id(correlation_id)
        request.state.correlation_id = correlation_id

        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            set_correlation_id(None)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        response.headers["X-Correlation-ID"] = correlation_id
        logger.info(
            "http.request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(elapsed_ms, 2),
            correlation_id=correlation_id,
        )
        return response


class _TokenBucket:
    __slots__ = ("capacity", "tokens", "refill_per_sec", "updated")

    def __init__(self, rate: int) -> None:
        self.capacity = float(rate)
        self.tokens = float(rate)
        self.refill_per_sec = float(rate)
        self.updated = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.refill_per_sec)
        self.updated = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-process per-client token bucket.

    Keyed by tenant header when present, else client IP. For multi-replica
    deployments this would move to a shared Redis counter; the interface stays
    the same.
    """

    def __init__(self, app, *, rate_per_sec: int = 50, exempt_paths: tuple[str, ...] = ()) -> None:
        super().__init__(app)
        self._rate = rate_per_sec
        self._exempt = exempt_paths
        self._buckets: dict[str, _TokenBucket] = {}

    def _client_key(self, request: Request) -> str:
        tenant = request.headers.get("x-tenant-id")
        if tenant:
            return f"tenant:{tenant}"
        client = request.client.host if request.client else "unknown"
        return f"ip:{client}"

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in self._exempt:
            return await call_next(request)

        key = self._client_key(request)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = self._buckets[key] = _TokenBucket(self._rate)

        if not bucket.allow():
            logger.warning("http.rate_limited", client=key, path=request.url.path)
            return JSONResponse(
                {"error": {"message": "rate limit exceeded", "type": "rate_limit_error"}},
                status_code=429,
                headers={"Retry-After": "1"},
            )
        return await call_next(request)
