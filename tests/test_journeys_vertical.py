"""Vertical journey tests — real routers in-process (no stack, no model).

API legs of the mobile in-code journeys (`m1-organize`, `m2-context` in
``domain/journeys/flows.py``), walked in the call order the mobile app
executes them through FastAPI ``TestClient`` → real handlers. The jest
journeys (``apps/mobile/src/components/__tests__/journeys/``) assert client
behavior against mocked transports; these assert the server contracts those
mocks stand in for:

- m1: session lifecycle (create → history seed → load → clone → list →
  delete) across both storage projections — the file drawer store
  (``/chat/sessions``) and the in-memory context store
  (``/session/{id}/messages``, fed by ``SessionCore.store_context``).
- m2: context injection (``POST /chat/control`` action=context) lands on the
  stream pickup seam, and ``GET /context/inspect`` returns the payload the
  mobile viewer mapper consumes.
"""

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.exception_handlers import register_all_handlers

from apps.api.server.routers.inference import (
    _instance as _inference_router,
)
from apps.api.server.routers.inference import (
    get_chat_control,
)
from apps.api.server.routers.session import router as session_router


@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    register_all_handlers(app)
    app.include_router(_inference_router.router)
    app.include_router(session_router)
    return TestClient(app, raise_server_exceptions=False)


def _refresh_session_list():
    # list_sessions serves a 5s metadata cache (_build_session_metadata_index);
    # the journey asserts post-write drawer state deterministically.
    _inference_router._session_metadata_cache = None


class TestM1OrganizeJourney:
    """m1-organize — create, read history, clone, list, delete."""

    def test_session_lifecycle_with_clone(self, client):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        # 1. new session (drawer create)
        r = client.post("/chat/sessions", json={})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "success"
        src = body["data"]["session_id"]
        assert src

        try:
            # 2. conversation history lands in the context store — the same
            #    seam ChatManager._store and mobile saveSessionContext write
            r = client.post(f"/session/{src}/context", json={"messages": history})
            assert r.status_code == 200, r.text
            assert r.json()["data"]["message_count"] == len(history)

            # 3. loadSession leg — history roundtrip
            r = client.get(f"/session/{src}/messages")
            assert r.status_code == 200, r.text
            assert r.json()["data"]["messages"] == history

            # 4. clone: create target, copy name + history
            #    (mobile cloneSession = POST + GET source + PUT + seed context)
            r = client.post("/chat/sessions", json={})
            assert r.status_code == 200, r.text
            clone = r.json()["data"]["session_id"]

            r = client.put(
                f"/chat/sessions/{clone}",
                json={"name": "Hello (Copy)", "messages": history},
            )
            assert r.status_code == 200, r.text
            assert r.json()["data"]["session_id"] == clone

            r = client.get(f"/session/{src}/messages")
            assert r.status_code == 200, r.text
            source_messages = r.json()["data"]["messages"]

            r = client.post(f"/session/{clone}/context", json={"messages": source_messages})
            assert r.status_code == 200, r.text

            # 5. opening the clone yields identical history
            r = client.get(f"/session/{clone}/messages")
            assert r.status_code == 200, r.text
            assert r.json()["data"]["messages"] == history

            # 6. drawer lists both sessions; clone carries its new name
            _refresh_session_list()
            r = client.get("/chat/sessions")
            assert r.status_code == 200, r.text
            sessions = {s["id"]: s for s in r.json()["data"]}
            assert src in sessions, "created session missing from drawer list"
            assert clone in sessions, "cloned session missing from drawer list"
            assert sessions[clone]["name"] == "Hello (Copy)"

            # 7. delete source → gone from drawer; clone survives
            r = client.delete(f"/chat/sessions/{src}")
            assert r.status_code == 200, r.text
            _refresh_session_list()
            r = client.get("/chat/sessions")
            ids = [s["id"] for s in r.json()["data"]]
            assert src not in ids, "deleted session still listed"
            assert clone in ids, "clone was collateral-deleted"
        finally:
            client.delete(f"/chat/sessions/{src}")
            client.delete(f"/chat/sessions/{clone}")
            _refresh_session_list()


class TestM2ContextJourney:
    """m2-context — inject what the AI should see, then inspect it."""

    def test_inject_context_then_inspect(self, client):
        session_id = f"journey-ctx-{uuid.uuid4().hex[:8]}"

        # 1. ContextInjectionBar: POST /chat/control action=context
        r = client.post(
            "/chat/control",
            json={
                "session_id": session_id,
                "action": "context",
                "context": "The lease requires 30 days notice to move out.",
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["data"] == {"stored": True, "session_id": session_id}

        # 2. the stream pickup seam holds exactly what was injected
        #    (get_chat_control is consume-once, as the streaming loop reads it)
        pending = get_chat_control(session_id)
        assert pending is not None, "injected context never reached the control store"
        assert pending["action"] == "context"
        assert "30 days notice" in pending["context"]

        # 3. nothing else is left queued for this session
        assert get_chat_control(session_id) is None

        # 4. ContextWindowViewer: GET /context/inspect returns the raw payload
        #    the mobile mapper (services/context-items.ts) consumes
        r = client.get("/context/inspect")
        assert r.status_code == 200, r.text
        payload = r.json()
        assert isinstance(payload.get("system_prompt"), str)
        assert isinstance(payload.get("session_messages"), list)
        for key in (
            "working_memory",
            "semantic_keys",
            "episodic_count",
            "sensory_buffer_size",
        ):
            assert key in payload, f"inspector payload missing mapper field {key!r}"

    def test_control_rejects_unknown_action(self, client):
        r = client.post(
            "/chat/control",
            json={"session_id": "x", "action": "warp-drive"},
        )
        assert r.status_code in (400, 422), r.text
