"""Echo completer — a development stand-in for a real provider.

Returns a deterministic canned completion without any network call. It lets the
proxy run end-to-end (and the cache be exercised) before provider credentials
are configured. It is replaced by the real provider router in production wiring.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.models.chat import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionUsage,
)


class EchoCompleter:
    name = "echo"

    def resolve_provider(self, request: ChatCompletionRequest) -> str:
        return self.name

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        prompt = request.latest_user_prompt()
        completion = f"[echo:{request.model}] {prompt}"
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(completion) // 4)
        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=completion),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[bytes]:
        from app.proxy.streaming import response_to_sse

        response = await self.complete(request)
        for chunk in response_to_sse(response):
            yield chunk
