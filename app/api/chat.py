"""OpenAI-compatible chat completions endpoint.

``POST /v1/chat/completions`` accepts and returns the OpenAI schema verbatim, so
clients switch to the proxy by changing only ``OPENAI_BASE_URL``. Both buffered
and streaming (``stream: true``) responses are supported. Cache outcome is
surfaced via response headers without altering the body:

    X-Cache-Status     HIT | MISS
    X-Similarity-Score the best similarity score considered
    X-Cache-Latency    cache lookup time in milliseconds
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.dependencies import get_proxy, get_tenant
from app.models.chat import ChatCompletionRequest
from app.proxy.service import ProxyResult, ProxyService

router = APIRouter(prefix="/v1", tags=["chat"])


def _cache_headers(result: ProxyResult) -> dict[str, str]:
    return {
        "X-Cache-Status": result.cache_status.value,
        "X-Similarity-Score": f"{result.similarity_score:.4f}",
        "X-Cache-Latency": f"{result.cache_latency_ms:.3f}ms",
        "X-Provider": result.provider,
    }


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    proxy: ProxyService = Depends(get_proxy),
    tenant: str = Depends(get_tenant),
):
    if request.stream:
        stream_result = await proxy.stream(request, tenant=tenant)
        return StreamingResponse(
            stream_result.body,
            media_type="text/event-stream",
            headers=stream_result.headers,
        )

    result = await proxy.complete(request, tenant=tenant)
    return JSONResponse(content=result.response.model_dump(), headers=_cache_headers(result))
