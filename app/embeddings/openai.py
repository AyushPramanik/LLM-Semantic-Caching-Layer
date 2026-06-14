"""OpenAI embedding service (``text-embedding-3-small`` by default).

Uses the REST embeddings endpoint directly via httpx so we avoid a hard SDK
dependency and keep the proxy provider-agnostic. Requests are retried on
transient failures with exponential backoff.
"""

from __future__ import annotations

import httpx
import numpy as np
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.logging import get_logger
from app.embeddings.base import EmbeddingResult, EmbeddingService

logger = get_logger(__name__)

_RETRYABLE = (httpx.TransportError, httpx.HTTPStatusError)


class OpenAIEmbeddingService(EmbeddingService):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        dimensions: int,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(model=model, dimensions=dimensions)
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    async def embed(self, text: str) -> EmbeddingResult:
        results = await self.embed_batch([text])
        return results[0]

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.2, max=2.0),
        reraise=True,
    )
    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        if not texts:
            return []
        payload = {"model": self.model, "input": texts, "dimensions": self.dimensions}
        resp = await self._client.post("/embeddings", json=payload)
        resp.raise_for_status()
        body = resp.json()

        total_tokens = body.get("usage", {}).get("total_tokens", 0)
        per_item_tokens = total_tokens // max(len(texts), 1)

        results: list[EmbeddingResult] = []
        for item in sorted(body["data"], key=lambda d: d["index"]):
            vector = self.normalize(np.asarray(item["embedding"], dtype=np.float32))
            results.append(
                EmbeddingResult(
                    vector=vector.tolist(),
                    model=self.model,
                    dimensions=self.dimensions,
                    tokens=per_item_tokens,
                )
            )
        return results

    async def aclose(self) -> None:
        await self._client.aclose()
