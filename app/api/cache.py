"""Cache invalidation endpoints.

Operational levers for evicting stale or incorrect cached responses without a
full flush. Useful when a model is updated, a system prompt changes, or a tagged
class of answers needs to be purged.

    DELETE /cache/model/{model}
    DELETE /cache/system-prompt/{hash}
    DELETE /cache/tag/{tag}
    DELETE /cache/all
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_store
from app.cache.store import VectorStore

router = APIRouter(prefix="/cache", tags=["cache"])


@router.delete("/model/{model}")
async def invalidate_model(model: str, store: VectorStore = Depends(get_store)):
    removed = await store.delete(model=model)
    return {"invalidated": removed, "scope": "model", "model": model}


@router.delete("/system-prompt/{prompt_hash}")
async def invalidate_system_prompt(
    prompt_hash: str, store: VectorStore = Depends(get_store)
):
    removed = await store.delete(system_prompt_hash=prompt_hash)
    return {"invalidated": removed, "scope": "system_prompt", "hash": prompt_hash}


@router.delete("/tag/{tag}")
async def invalidate_tag(tag: str, store: VectorStore = Depends(get_store)):
    removed = await store.delete(tag=tag)
    return {"invalidated": removed, "scope": "tag", "tag": tag}


@router.delete("/all")
async def invalidate_all(store: VectorStore = Depends(get_store)):
    removed = await store.clear()
    return {"invalidated": removed, "scope": "all"}
