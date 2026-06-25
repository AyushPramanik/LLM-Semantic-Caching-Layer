"""Tests for cache invalidation endpoints."""

from __future__ import annotations


def _post(client, content, model="gpt-4o-mini", tags=None):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": content},
        ],
    }
    if tags is not None:
        body["cache_tags"] = tags
    return client.post("/v1/chat/completions", json=body)


def test_invalidate_by_model(client):
    _post(client, "explain http", model="gpt-4o-mini")
    _post(client, "explain tcp", model="gpt-4o")

    resp = client.delete("/cache/model/gpt-4o-mini")
    assert resp.status_code == 200
    assert resp.json()["invalidated"] == 1

    # The gpt-4o-mini entry is gone -> miss again; gpt-4o remains -> hit.
    assert _post(client, "explain http", model="gpt-4o-mini").headers["X-Cache-Status"] == "MISS"
    assert _post(client, "explain tcp", model="gpt-4o").headers["X-Cache-Status"] == "HIT"


def test_invalidate_by_tag(client):
    _post(client, "what is a database index", tags=["docs"])
    # ttl:long is auto-tagged by the policy; "docs" is caller-supplied.
    resp = client.delete("/cache/tag/docs")
    assert resp.json()["invalidated"] == 1
    again = _post(client, "what is a database index", tags=["docs"])
    assert again.headers["X-Cache-Status"] == "MISS"


def test_invalidate_all(client):
    _post(client, "prompt one")
    _post(client, "prompt two")
    resp = client.delete("/cache/all")
    assert resp.json()["invalidated"] >= 2
    assert _post(client, "prompt one").headers["X-Cache-Status"] == "MISS"


def test_invalidate_by_system_prompt_hash(client):
    from app.cache.namespace import hash_system_prompt

    _post(client, "scoped prompt")
    digest = hash_system_prompt("You are helpful.")
    resp = client.delete(f"/cache/system-prompt/{digest}")
    assert resp.json()["invalidated"] == 1
