"""
Tests for the DocStore router (MogDB-backed document store for the browser chat DB).

Covers the CRUD contract the frontend ``apps/web/lib/db.ts`` client depends on:
upsert semantics, merge-only PATCH, meta stripping, collection whitelisting,
bulk import, and sort/limit listing.
"""

import pytest
from mogdb import MogDB
from routers import docstore
from test_support import get_test_client

client = get_test_client()


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path):
    """Point the router's MogDB instance at a fresh temp directory per test."""
    docstore._db = MogDB(str(tmp_path / "docstore"))
    yield
    docstore._db = None


def _data(resp):
    """Unwrap the success_response() envelope."""
    body = resp.json()
    return body.get("data", body)


# ── upsert (PUT) + read (GET) ──────────────────────────────────────────────


def test_put_then_get_round_trip():
    session = {
        "id": "s1",
        "name": "Hello",
        "messages": [
            {"id": "m1", "role": "user", "content": "hi", "timestamp": "2024-01-01T00:00:00.000Z"}
        ],
        "createdAt": "2024-01-01T00:00:00.000Z",
        "updatedAt": "2024-01-01T00:00:00.000Z",
        "synced": False,
        "starred": False,
        "pinned": False,
    }
    put = client.put("/docstore/sessions/s1", json=session)
    assert put.status_code == 200
    assert _data(put) == {"id": "s1", "created": True}

    got = client.get("/docstore/sessions/s1")
    assert got.status_code == 200
    data = _data(got)
    assert data["id"] == "s1"
    assert data["name"] == "Hello"
    assert data["messages"] == session["messages"]
    assert "_id" not in data
    assert "_created" not in data
    assert "_updated" not in data


def test_get_missing_doc_returns_none():
    resp = client.get("/docstore/sessions/nope")
    assert resp.status_code == 200
    assert _data(resp) is None


def test_put_replaces_existing_document():
    client.put("/docstore/kv/theme", json={"id": "theme", "value": "dark"})
    second = client.put("/docstore/kv/theme", json={"id": "theme", "value": "light"})
    assert _data(second) == {"id": "theme", "created": False}

    got = client.get("/docstore/kv/theme")
    assert _data(got) == {"id": "theme", "value": "light"}


def test_put_overwrites_removed_fields():
    client.put(
        "/docstore/bookmarks/b1", json={"id": "b1", "content": "a", "role": "user", "timestamp": 1}
    )
    client.put(
        "/docstore/bookmarks/b1",
        json={"id": "b1", "content": "b", "role": "assistant", "timestamp": 2},
    )
    data = _data(client.get("/docstore/bookmarks/b1"))
    assert data == {"id": "b1", "content": "b", "role": "assistant", "timestamp": 2}


# ── merge-only PATCH ────────────────────────────────────────────────────────


def test_patch_merges_into_existing_doc():
    client.put(
        "/docstore/sessions/s1", json={"id": "s1", "name": "Old", "synced": False, "pinned": False}
    )
    patch = client.patch("/docstore/sessions/s1", json={"name": "New", "synced": True})
    assert patch.status_code == 200
    assert _data(patch) == {"modified": 1}

    data = _data(client.get("/docstore/sessions/s1"))
    assert data["name"] == "New"
    assert data["synced"] is True
    assert data["pinned"] is False  # untouched field preserved


def test_patch_missing_doc_is_noop():
    resp = client.patch("/docstore/sessions/ghost", json={"starred": True})
    assert _data(resp) == {"modified": 0}
    assert _data(client.get("/docstore/sessions/ghost")) is None


def test_patch_empty_body_is_noop():
    client.put("/docstore/kv/k", json={"id": "k", "value": 1})
    resp = client.patch("/docstore/kv/k", json={})
    assert _data(resp) == {"modified": 0}
    assert _data(client.get("/docstore/kv/k")) == {"id": "k", "value": 1}


# ── DELETE ─────────────────────────────────────────────────────────────────


def test_delete_single_doc():
    client.put("/docstore/drafts/d1", json={"id": "d1", "text": "hi"})
    assert _data(client.delete("/docstore/drafts/d1")) == {"deleted": True}
    assert _data(client.delete("/docstore/drafts/d1")) == {"deleted": False}
    assert _data(client.get("/docstore/drafts/d1")) is None


def test_delete_collection_clears_all():
    client.put("/docstore/errors/e1", json={"id": "e1", "message": "boom"})
    client.put("/docstore/errors/e2", json={"id": "e2", "message": "bang"})
    assert _data(client.delete("/docstore/errors")) == {"cleared": True}
    assert _data(client.get("/docstore/errors")) == []


# ── bulk import (POST /{collection}/bulk) ──────────────────────────────────


def test_bulk_import():
    docs = [
        {
            "id": "p1",
            "name": "Summarize",
            "prompt": "sum",
            "icon": "",
            "category": "a",
            "createdAt": 1,
            "updatedAt": 1,
            "description": "",
        },
        {
            "id": "p2",
            "name": "Translate",
            "prompt": "tr",
            "icon": "",
            "category": "b",
            "createdAt": 2,
            "updatedAt": 2,
            "description": "",
        },
    ]
    resp = client.post("/docstore/prompts/bulk", json={"docs": docs})
    assert resp.status_code == 200
    assert _data(resp) == {"imported": 2}

    listed = _data(client.get("/docstore/prompts"))
    assert {d["id"] for d in listed} == {"p1", "p2"}


def test_bulk_import_overwrites_existing():
    client.put("/docstore/knowledge/k1", json={"id": "k1", "content": "old", "timestamp": 0})
    resp = client.post(
        "/docstore/knowledge/bulk", json={"docs": [{"id": "k1", "content": "new", "timestamp": 1}]}
    )
    assert _data(resp) == {"imported": 1}
    data = _data(client.get("/docstore/knowledge/k1"))
    assert data["content"] == "new"


def test_bulk_requires_docs_array():
    resp = client.post("/docstore/knowledge/bulk", json={"docs": "nope"})
    assert resp.status_code == 400
    body = resp.json()
    assert "docs" in body.get("error", "").lower() or "docs" in str(body).lower()


def test_bulk_skips_docs_without_id():
    resp = client.post("/docstore/knowledge/bulk", json={"docs": [{"content": "no id"}]})
    assert _data(resp) == {"imported": 0}


# ── list + sort/limit ──────────────────────────────────────────────────────


def test_list_returns_empty_array():
    assert _data(client.get("/docstore/sessions")) == []


def test_list_sort_and_limit():
    for i in range(3):
        client.put(
            "/docstore/pendingMessages/p%d" % i,
            json={"id": "p%d" % i, "content": str(i), "createdAt": "2024-01-0%d" % (i + 1)},
        )
    listed = _data(client.get("/docstore/pendingMessages?sort=createdAt&dir=-1&limit=2"))
    assert [d["id"] for d in listed] == ["p2", "p1"]


# ── collection whitelist ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/docstore/unknown"),
        ("put", "/docstore/unknown/x"),
        ("patch", "/docstore/unknown/x"),
        ("delete", "/docstore/unknown/x"),
        ("post", "/docstore/unknown/bulk"),
    ],
)
def test_unknown_collection_rejected(method, path):
    resp = client.request(method, path, json={} if method in ("put", "patch", "post") else None)
    body = resp.json()
    assert body["code"] == "E_UNKNOWN_COLLECTION"
    assert "error" in body
