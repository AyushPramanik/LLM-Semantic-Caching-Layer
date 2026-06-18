"""Abstract chat provider interface.

A provider knows how to (a) take an OpenAI-shaped request, (b) call its upstream
vendor, and (c) return an OpenAI-shaped response. Streaming support is layered on
in Phase 2 via :meth:`stream`.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator

import httpx

from app.models.chat import ChatCompletionRequest, ChatCompletionResponse


class ProviderError(RuntimeError):
    """Raised when an upstream provider call fails non-transiently."""

    def __init__(self, message: str, *, status_code: int = 502, provider: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider


class ChatProvider(abc.ABC):
    """Base class for upstream LLM providers."""

    name: str = "base"

    def __init__(self, *, base_url: str, api_key: str = "", timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)

    @abc.abstractmethod
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Return a full (non-streamed) completion."""

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[bytes]:
        """Yield raw SSE chunks from the provider. Overridden in Phase 2."""
        raise NotImplementedError(f"{self.name} does not implement streaming")
        yield b""  # pragma: no cover

    async def aclose(self) -> None:
        await self._client.aclose()
