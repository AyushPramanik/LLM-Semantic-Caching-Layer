"""Provider abstraction and routing.

Each upstream LLM vendor is wrapped in a :class:`ChatProvider`. A provider router
selects the right provider per request using the model name (strategy pattern),
so the cache and proxy stay vendor-agnostic.
"""

from __future__ import annotations

from app.providers.anthropic import AnthropicProvider
from app.providers.base import ChatProvider, ProviderError
from app.providers.openai import OpenAIProvider

__all__ = ["AnthropicProvider", "ChatProvider", "OpenAIProvider", "ProviderError"]
