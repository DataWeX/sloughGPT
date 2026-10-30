"""Stable message ids through the context-store seam (slice 4 foundation).

Mobile messages carry client-generated ids, but ``SessionCore.store_context``
converted through ``MessageData(role, content)`` — stripping them — so every
reloaded session came back with ``role/content`` only and ``Message.id`` was
undefined at runtime (breaking id-keyed consumers: delete, regenerate, pins).
This suite pins the contract: ids written to ``POST /session/{id}/context``
round-trip through ``GET /session/{id}/messages``; legacy writes without ids
keep their exact legacy shape (no ``id: null`` leak); branch prefixes carry
ids forward.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.exception_handlers import register_all_handlers

from apps.api.server.routers.inference import _instance as _inference_router
from apps.api.server.routers.session import router as session_router

MESSAGES_WITH_IDS = [
    {"id": "s1:0:u", "role": "user", "content": "question zero"},
    {"id": "s1:1:a", "role": "assistant", "content": "answer one"},
    {"id": "s1:2:u", "role": "user", "content": "question two"},
]

MESSAGES_LEGACY = [
    {"role": "user", "content": "question zero"},
    {"role": "assistant", "content": "answer one"},
]


@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    register_all_handlers(app)
    app.include_router(_inference_router.router)
    app.include_router(session_router)
    return TestClient(app, raise_server_exceptions=False)


def _seed(client: TestClient, messages: list[dict]) -> str:
    r = client.post("/chat/sessions", json={})
    assert r.status_code == 200, r.text
    sid = r.json()["data"]["session_id"]
    r = client.post(f"/session/{sid}/context", json={"messages": messages})
    assert r.status_code == 200, r.text
    return sid


def _cleanup(client: TestClient, *sids: str) -> None:
    for sid in sids:
        if sid:
            client.delete(f"/chat/sessions/{sid}")
    _inference_router._session_metadata_cache = None


class TestIdRoundtrip:
    def test_ids_written_to_context_come_back_from_messages_route(self, client):
        sid = _seed(client, MESSAGES_WITH_IDS)
        try:
            r = client.get(f"/session/{sid}/messages")
            assert r.status_code == 200, r.text
            got = r.json()["data"]["messages"]
            assert [m["id"] for m in got] == ["s1:0:u", "s1:1:a", "s1:2:u"]
            assert [(m["role"], m["content"]) for m in got] == [
                (m["role"], m["content"]) for m in MESSAGES_WITH_IDS
            ]
        finally:
            _cleanup(client, sid)

    def test_legacy_writes_keep_legacy_shape_no_null_ids(self, client):
        sid = _seed(client, MESSAGES_LEGACY)
        try:
            r = client.get(f"/session/{sid}/messages")
            assert r.status_code == 200, r.text
            got = r.json()["data"]["messages"]
            assert got == MESSAGES_LEGACY
            for m in got:
                assert "id" not in m
        finally:
            _cleanup(client, sid)


class TestBranchCarriesIds:
    def test_branch_prefix_preserves_message_ids(self, client):
        src = _seed(client, MESSAGES_WITH_IDS)
        new = None
        try:
            r = client.post(f"/chat/sessions/{src}/branch", json={"from_index": 1})
            assert r.status_code == 200, r.text
            new = r.json()["data"]["session_id"]

            r = client.get(f"/session/{new}/messages")
            assert r.status_code == 200, r.text
            got = r.json()["data"]["messages"]
            assert [m["id"] for m in got] == ["s1:0:u", "s1:1:a"]
        finally:
            _cleanup(client, src, new)
