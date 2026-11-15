"""Contract tests for the ``agents.routines_stream`` crossing (agent phase 2).

The routine status stream is descriptor-projected — one ``ToolSpec``, one
``create_router`` projection in its own module (``routers/agent_routines.py``,
because the hand-built declaration ratchet forbids a new route site inside the
grandfathered agents router). These tests hold the same guarantees every
projected route carries: one declaration, one auth story, and the registry
claim must match what FastAPI actually serves.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.auth import require_auth_if_enabled
from infrastructure.contract import REGISTRY
from infrastructure.exception_handlers import register_all_handlers
from routers.agent_routines import SPEC
from routers.agent_routines import router as routines_router

# The contract name this module exists to pin (also read by
# scripts/check_contract.py --static: a contract without a test is a wish).
CONTRACT_NAME = "agents.routines_stream"


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(routines_router)
    register_all_handlers(app)
    return app


def _claimed() -> list[dict]:
    return [e for e in REGISTRY.routes() if e["module"] == "routers.agent_routines"]


class _NullStore:
    """A store that never touches disk — the snapshot frames need no data."""

    def list(self):
        return []

    def get(self, routine_id):
        return None

    def save(self, routine):
        return True

    def delete(self, routine_id):
        return False


class _StubRunner:
    running = False
    store = _NullStore()

    def events_since(self, cursor):
        return []


def test_agents_routines_stream_descriptor_is_registered():
    rows = [e for e in _claimed() if e["path"] == "/agents/routines/stream"]
    assert rows, f"{CONTRACT_NAME} missing from the contract registry"
    row = rows[0]
    assert row["name"] == CONTRACT_NAME
    assert row["method"] == "GET"
    assert row["auth_scope"] == "authenticated"
    assert row["idempotent"] is True  # a read — GET is derived, not hand-picked
    assert SPEC.name == CONTRACT_NAME
    assert SPEC.parameters == []  # no inputs: a pure status stream


def test_agents_routines_stream_registry_claim_is_served():
    """Registry ↔ served parity: the projection must land on the real path."""
    served = {
        (route.path, method)
        for route in routines_router.routes
        for method in (route.methods or set())
        if method != "HEAD"
    }
    assert ("/agents/routines/stream", "GET") in served
    claimed = {(e["path"], e["method"]) for e in _claimed()}
    assert claimed <= served, f"registry claims routes no router serves: {claimed - served}"


def test_agents_routines_stream_serves_sse_snapshot(monkeypatch):
    """Raw-wire facet (envelope=False): an SSE stream, not the JSON envelope."""
    monkeypatch.setattr("routers.agent_routines.get_routine_runner", lambda: _StubRunner())
    with patch("fastapi.Request.is_disconnected", new=AsyncMock(side_effect=[False, True])):
        with TestClient(_app()).stream("GET", "/agents/routines/stream") as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            body = resp.read().decode()
    assert '"stream": "agent-routines"' in body
    assert '"phase": "SNAPSHOT"' in body
    assert '"phase": "COMPLETE"' in body


def test_agents_routines_stream_honors_auth_scope(monkeypatch):
    """One auth story: authenticated scope → 401 without a token."""
    monkeypatch.setenv("SLO_AUTH_REQUIRED", "true")
    app = _app()
    app.dependency_overrides[require_auth_if_enabled] = lambda: None
    try:
        with patch("fastapi.Request.is_disconnected", new=AsyncMock(side_effect=[True])):
            resp = TestClient(app).get("/agents/routines/stream")
        assert resp.status_code == 200  # override proves the dependency IS wired
    finally:
        app.dependency_overrides.clear()

    # Without the override: anonymous traffic is rejected before the handler.
    with patch("fastapi.Request.is_disconnected", new=AsyncMock(side_effect=[True])):
        resp = TestClient(_app()).get("/agents/routines/stream")
    assert resp.status_code == 401
    assert resp.json()["code"] == "E_AUTH_MISSING"
