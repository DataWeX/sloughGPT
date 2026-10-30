"""Branching + lineage journeys — session forking with prefix-copy semantics.

Backend legs of slice 3 (conversation branching): `POST /chat/sessions/{id}/branch`
copies `messages[:from_index + 1]` into a new session (drawer file + context
store, so `loadSession` reads it), records lineage on the new session, and
`GET /chat/sessions/{id}/lineage` walks the parent chain. The branch point is
an **index** — server-side messages carry no ids (`MessageData` is
role+content), so identity is the position in the ordered list.
"""

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.exception_handlers import register_all_handlers

from apps.api.server.routers.inference import _instance as _inference_router
from apps.api.server.routers.session import router as session_router

MESSAGES = [
    {"role": "user", "content": "question zero"},
    {"role": "assistant", "content": "answer one"},
    {"role": "user", "content": "question two"},
]


@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    register_all_handlers(app)
    app.include_router(_inference_router.router)
    app.include_router(session_router)
    return TestClient(app, raise_server_exceptions=False)


def _seed(client: TestClient, with_history: bool = True, name: str | None = None) -> str:
    """Create a session the way production does: file + context store."""
    body = {"name": name} if name else {}
    r = client.post("/chat/sessions", json=body)
    assert r.status_code == 200, r.text
    sid = r.json()["data"]["session_id"]
    if with_history:
        r = client.post(f"/session/{sid}/context", json={"messages": MESSAGES})
        assert r.status_code == 200, r.text
    return sid


def _cleanup(client: TestClient, *sids: str) -> None:
    for sid in sids:
        if sid:
            client.delete(f"/chat/sessions/{sid}")
    _inference_router._session_metadata_cache = None


class TestBranchEndpoint:
    def test_branch_copies_prefix_and_is_independently_readable(self, client):
        src = _seed(client)
        new = None
        try:
            r = client.post(f"/chat/sessions/{src}/branch", json={"from_index": 1})
            assert r.status_code == 200, r.text
            data = r.json()["data"]
            new = data["session_id"]
            assert new and new != src
            assert data["branched_from"] == src
            assert data["from_index"] == 1

            # new session opens with exactly the prefix (inclusive boundary)
            r = client.get(f"/session/{new}/messages")
            assert r.status_code == 200, r.text
            assert r.json()["data"]["messages"] == MESSAGES[:2]

            # source untouched
            r = client.get(f"/session/{src}/messages")
            assert r.json()["data"]["messages"] == MESSAGES
        finally:
            _cleanup(client, src, new or "")

    def test_branch_boundaries_first_and_last_message(self, client):
        src = _seed(client)
        first = last = None
        try:
            r = client.post(f"/chat/sessions/{src}/branch", json={"from_index": 0})
            assert r.status_code == 200, r.text
            first = r.json()["data"]["session_id"]
            r = client.get(f"/session/{first}/messages")
            assert r.json()["data"]["messages"] == MESSAGES[:1]

            r = client.post(f"/chat/sessions/{src}/branch", json={"from_index": 2})
            assert r.status_code == 200, r.text
            last = r.json()["data"]["session_id"]
            r = client.get(f"/session/{last}/messages")
            assert r.json()["data"]["messages"] == MESSAGES
        finally:
            _cleanup(client, src, first, last)

    def test_branch_rejects_out_of_range_and_empty_sources(self, client):
        src = _seed(client)
        empty = _seed(client, with_history=False)
        try:
            r = client.post(f"/chat/sessions/{src}/branch", json={"from_index": 99})
            assert r.status_code == 400, r.text
            r = client.post(f"/chat/sessions/{src}/branch", json={"from_index": -1})
            assert r.status_code == 400, r.text
            # no history → nothing to branch from
            r = client.post(f"/chat/sessions/{empty}/branch", json={"from_index": 0})
            assert r.status_code == 400, r.text
            r = client.post(
                f"/chat/sessions/nope-{uuid.uuid4().hex[:8]}/branch",
                json={"from_index": 0},
            )
            assert r.status_code == 404, r.text
        finally:
            _cleanup(client, src, empty)

    def test_branch_records_lineage_on_new_session(self, client):
        src = _seed(client, name="Root chat")
        new = None
        try:
            r = client.post(f"/chat/sessions/{src}/branch", json={"from_index": 1})
            new = r.json()["data"]["session_id"]

            r = client.get(f"/chat/sessions/{new}/lineage")
            assert r.status_code == 200, r.text
            data = r.json()["data"]
            assert data["session_id"] == new
            assert data["parent_session_id"] == src
            assert data["branch_from_index"] == 1

            # lineage fields ride the session payload too (drawer-visible)
            r = client.get(f"/chat/sessions/{new}")
            assert r.status_code == 200, r.text
            assert r.json()["data"]["parent_session_id"] == src
        finally:
            _cleanup(client, src, new or "")


class TestLineageEndpoint:
    def test_chain_walks_nearest_parent_first(self, client):
        a = _seed(client, name="A")
        b = c = None
        try:
            # root: no parent, empty ancestors
            r = client.get(f"/chat/sessions/{a}/lineage")
            assert r.status_code == 200, r.text
            data = r.json()["data"]
            assert data["session_id"] == a
            assert data["parent_session_id"] is None
            assert data["ancestors"] == []

            r = client.post(f"/chat/sessions/{a}/branch", json={"from_index": 1})
            b = r.json()["data"]["session_id"]
            r = client.post(f"/chat/sessions/{b}/branch", json={"from_index": 0})
            c = r.json()["data"]["session_id"]

            r = client.get(f"/chat/sessions/{c}/lineage")
            assert r.status_code == 200, r.text
            data = r.json()["data"]
            assert data["parent_session_id"] == b
            assert [anc["session_id"] for anc in data["ancestors"]] == [b, a]
            assert data["ancestors"][1]["name"] == "A"
        finally:
            _cleanup(client, a, b or "", c or "")

    def test_lineage_unknown_session_404(self, client):
        r = client.get(f"/chat/sessions/nope-{uuid.uuid4().hex[:8]}/lineage")
        assert r.status_code == 404, r.text
