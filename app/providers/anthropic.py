"""Anthropic chat provider.

Anthropic's Messages API differs from OpenAI's: the system prompt is a top-level
field, ``max_tokens`` is required, and the response is a list of content blocks.
This adapter translates in both directions so callers keep using the OpenAI
schema end-to-end.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.logging import get_logger
from app.models.chat import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionUsage,
)
from app.providers.base import ChatProvider, ProviderError

logger = get_logger(__name__)

_DEFAULT_MAX_TOKENS = 1024
_FINISH_REASON = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}


class AnthropicProvider(ChatProvider):
    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float = 60.0,
        anthropic_version: str = "2023-06-01",
    ) -> None:
        super().__init__(base_url=base_url, api_key=api_key, timeout=timeout)
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": anthropic_version,
            "content-type": "application/json",
        }

    def _to_anthropic_payload(self, request: ChatCompletionRequest) -> dict[str, Any]:
        messages = [
            {"role": m.role, "content": m.content or ""}
            for m in request.messages
            if m.role in ("user", "assistant")
        ]
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or _DEFAULT_MAX_TOKENS,
            "temperature": request.temperature,
        }
        system = request.system_prompt()
        if system:
            payload["system"] = system
        return payload

    def _to_openai_response(self, body: dict[str, Any], model: str) -> ChatCompletionResponse:
        text = "".join(
            block.get("text", "")
            for block in body.get("content", [])
            if block.get("type") == "text"
        )
        usage = body.get("usage", {})
        return ChatCompletionResponse(
            id=body.get("id", f"chatcmpl-{uuid.uuid4().hex[:24]}"),
            created=int(time.time()),
            model=body.get("model", model),
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason=_FINISH_REASON.get(body.get("stop_reason", ""), "stop"),
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            ),
        )

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.2, max=2.0),
        reraise=True,
    )
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        payload = self._to_anthropic_payload(request)
        resp = await self._client.post("/messages", json=payload, headers=self._headers)
        if resp.status_code >= 400:
            raise ProviderError(
                f"anthropic returned {resp.status_code}: {resp.text[:200]}",
                status_code=resp.status_code,
                provider=self.name,
            )
        return self._to_openai_response(resp.json(), request.model)
