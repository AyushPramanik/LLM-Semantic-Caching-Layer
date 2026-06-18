"""Ollama chat provider.

Ollama exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint for local
models, so this adapter is a credential-free pass-through pointed at the local
Ollama server.
"""

from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.logging import get_logger
from app.models.chat import ChatCompletionRequest, ChatCompletionResponse
from app.providers.base import ChatProvider, ProviderError

logger = get_logger(__name__)


class OllamaProvider(ChatProvider):
    name = "ollama"

    def __init__(self, *, base_url: str, timeout: float = 120.0) -> None:
        # Local generation can be slow, so the default timeout is generous.
        super().__init__(base_url=base_url, api_key="", timeout=timeout)

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.2, max=2.0),
        reraise=True,
    )
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        payload = request.upstream_payload()
        payload["stream"] = False
        resp = await self._client.post("/chat/completions", json=payload)
        if resp.status_code >= 400:
            raise ProviderError(
                f"ollama returned {resp.status_code}: {resp.text[:200]}",
                status_code=resp.status_code,
                provider=self.name,
            )
        return ChatCompletionResponse.model_validate(resp.json())
