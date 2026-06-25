"""Server-Sent Events helpers for streaming chat completions.

Two directions are supported:

* **Cache HIT** — a stored full response is re-emitted as a short SSE stream so a
  streaming client sees the same wire format it expects.
* **Cache MISS** — raw SSE bytes from the upstream provider are forwarded to the
  client unchanged while a :class:`StreamAssembler` reconstructs the full
  response in the background so it can be cached *only after* the stream
  completes successfully.
"""

from __future__ import annotations

import time
import uuid

import orjson

from app.models.chat import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatMessage,
    CompletionUsage,
)

_DONE = b"data: [DONE]\n\n"


def _sse(data: dict) -> bytes:
    return b"data: " + orjson.dumps(data) + b"\n\n"


def response_to_sse(response: ChatCompletionResponse) -> list[bytes]:
    """Render a complete response as OpenAI-style streaming chunks."""
    chunk_id = response.id
    base = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": response.created,
        "model": response.model,
    }
    role_chunk = {
        **base,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    content = response.first_text()
    content_chunk = {
        **base,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    finish = response.choices[0].finish_reason if response.choices else "stop"
    final_chunk = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
    return [_sse(role_chunk), _sse(content_chunk), _sse(final_chunk), _DONE]


class StreamAssembler:
    """Reconstructs a full :class:`ChatCompletionResponse` from SSE chunks.

    Bytes are pushed in as they arrive (arbitrary boundaries); complete
    ``data:`` events are parsed out and their deltas accumulated.
    """

    def __init__(self, model: str) -> None:
        self._model = model
        self._buffer = b""
        self._content_parts: list[str] = []
        self._finish_reason: str | None = None
        self._id: str | None = None
        self._created: int | None = None
        self._usage: dict | None = None
        self.completed = False

    def push(self, chunk: bytes) -> None:
        self._buffer += chunk
        while b"\n\n" in self._buffer:
            event, self._buffer = self._buffer.split(b"\n\n", 1)
            self._consume_event(event)

    def _consume_event(self, event: bytes) -> None:
        for line in event.split(b"\n"):
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:"):].strip()
            if payload == b"[DONE]":
                self.completed = True
                continue
            try:
                data = orjson.loads(payload)
            except orjson.JSONDecodeError:
                continue
            self._merge(data)

    def _merge(self, data: dict) -> None:
        self._id = data.get("id", self._id)
        self._created = data.get("created", self._created)
        if data.get("usage"):
            self._usage = data["usage"]
        for choice in data.get("choices", []):
            delta = choice.get("delta", {})
            if delta.get("content"):
                self._content_parts.append(delta["content"])
            if choice.get("finish_reason"):
                self._finish_reason = choice["finish_reason"]

    def build_response(self) -> ChatCompletionResponse:
        content = "".join(self._content_parts)
        if self._usage:
            usage = CompletionUsage(**{k: v for k, v in self._usage.items()
                                       if k in CompletionUsage.model_fields})
        else:
            # Streaming responses frequently omit usage; estimate for accounting.
            completion_tokens = max(1, len(content) // 4)
            usage = CompletionUsage(completion_tokens=completion_tokens,
                                    total_tokens=completion_tokens)
        return ChatCompletionResponse(
            id=self._id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
            created=self._created or int(time.time()),
            model=self._model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason=self._finish_reason or "stop",
                )
            ],
            usage=usage,
        )
