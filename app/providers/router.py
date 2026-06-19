"""Provider router — selects a provider per request by model name.

This is the strategy pattern: a set of ordered ``(predicate -> provider)`` rules.
The first matching rule wins; an explicit default handles unknown models. The
router itself satisfies the proxy's ``Completer`` protocol so it can be dropped
in wherever a single provider would go.
"""

from __future__ import annotations

from collections.abc import Callable

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.chat import ChatCompletionRequest, ChatCompletionResponse
from app.providers.anthropic import AnthropicProvider
from app.providers.base import ChatProvider, ProviderError
from app.providers.ollama import OllamaProvider
from app.providers.openai import OpenAIProvider

logger = get_logger(__name__)

Rule = tuple[Callable[[str], bool], str]

# Ordered model-name routing rules. Prefix checks are cheap and explicit.
_DEFAULT_RULES: list[Rule] = [
    (lambda m: m.startswith(("gpt-", "o1", "o3", "chatgpt", "text-davinci")), "openai"),
    (lambda m: m.startswith("claude"), "anthropic"),
    (lambda m: m.startswith(("llama", "mistral", "qwen", "gemma", "phi", "deepseek")), "ollama"),
]


class ProviderRouter:
    def __init__(
        self,
        providers: dict[str, ChatProvider],
        *,
        rules: list[Rule] | None = None,
        default: str = "openai",
    ) -> None:
        self._providers = providers
        self._rules = rules if rules is not None else _DEFAULT_RULES
        self._default = default

    def resolve_provider(self, request: ChatCompletionRequest) -> str:
        """Return the provider name that will serve this request."""
        model = request.model.lower()
        for predicate, provider in self._rules:
            if predicate(model):
                return provider
        return self._default

    def _provider_for(self, request: ChatCompletionRequest) -> ChatProvider:
        name = self.resolve_provider(request)
        provider = self._providers.get(name)
        if provider is None:
            raise ProviderError(
                f"no provider configured for '{name}' (model={request.model})",
                status_code=400,
                provider=name,
            )
        return provider

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        provider = self._provider_for(request)
        logger.info("router.dispatch", model=request.model, provider=provider.name)
        return await provider.complete(request)

    async def stream(self, request: ChatCompletionRequest):
        provider = self._provider_for(request)
        logger.info("router.dispatch.stream", model=request.model, provider=provider.name)
        async for chunk in provider.stream(request):
            yield chunk

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()


def build_router(settings: Settings) -> ProviderRouter:
    """Construct a router with all configured providers."""
    providers: dict[str, ChatProvider] = {
        "openai": OpenAIProvider(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.upstream_timeout_seconds,
        ),
        "anthropic": AnthropicProvider(
            api_key=settings.anthropic_api_key,
            base_url=settings.anthropic_base_url,
            timeout=settings.upstream_timeout_seconds,
        ),
        "ollama": OllamaProvider(
            base_url=settings.ollama_base_url,
            timeout=settings.upstream_timeout_seconds,
        ),
    }
    return ProviderRouter(providers)
