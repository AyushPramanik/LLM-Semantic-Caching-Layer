"""OpenAI chat provider.

The proxy speaks the OpenAI wire format natively, so this adapter is a thin,
authenticated pass-through with retry/backoff on transient failures.
"""

from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.logging import get_logger
from app.models.chat import ChatCompletionRequest, ChatCompletionResponse
from app.providers.base import ChatProvider, ProviderError

logger = get_logger(__name__)


class OpenAIProvider(ChatProvider):
    name = "openai"

    def __init__(self, *, api_key: str, base_url: str, timeout: float = 60.0) -> None:
        super().__init__(base_url=base_url, api_key=api_key, timeout=timeout)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.2, max=2.0),
        reraise=True,
    )
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        payload = request.upstream_payload()
        payload["stream"] = False
        resp = await self._client.post("/chat/completions", json=payload, headers=self._headers)
        if resp.status_code >= 400:
            raise ProviderError(
                f"openai returned {resp.status_code}: {resp.text[:200]}",
                status_code=resp.status_code,
                provider=self.name,
            )
        return ChatCompletionResponse.model_validate(resp.json())
