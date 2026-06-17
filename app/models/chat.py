"""OpenAI-compatible chat completion schemas.

These mirror the public OpenAI Chat Completions contract closely enough that a
client only has to change its base URL. Unknown fields are preserved (``extra =
allow``) so newer parameters pass through to the upstream provider untouched.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool", "function"]
    content: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int | None = None
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None
    user: str | None = None
    # Proxy-specific, stripped before forwarding upstream.
    cache_tags: list[str] | None = Field(default=None, exclude=True)

    def system_prompt(self) -> str | None:
        for message in self.messages:
            if message.role == "system":
                return message.content
        return None

    def latest_user_prompt(self) -> str:
        """The text used as the semantic cache key.

        We key on the most recent user turn. Full conversational context could be
        folded in later, but the last user message is the dominant driver of the
        response and keeps the cache key stable across paraphrases.
        """
        for message in reversed(self.messages):
            if message.role == "user" and message.content:
                return message.content
        # Fall back to a concatenation of all message content.
        return "\n".join(m.content or "" for m in self.messages)

    def upstream_payload(self) -> dict[str, Any]:
        """Serialize for the upstream provider, dropping proxy-only fields."""
        return self.model_dump(exclude_none=True, exclude={"cache_tags"})


class ChatCompletionChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int = 0
    message: ChatMessage
    finish_reason: str | None = "stop"


class CompletionUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage = Field(default_factory=CompletionUsage)

    def first_text(self) -> str:
        if not self.choices:
            return ""
        return self.choices[0].message.content or ""
