"""GET /search — transport-level tests: auth, params, visibility gate.

Query semantics live in the core (test_search_service /
test_search_contract); this file only proves the endpoint wires them
up without leaking anything.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import routers.search as search_router
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.auth import require_auth_if_enabled
from infrastructure.exception_handlers import register_app_error_handler

from domain.search.registry import SearchRegistry
from domain.search.service import SearchService
from domain.search.types import SearchHit

_AUTH_USER = {"sub": "user1", "tenant_id": "t1"}


class _FakeStore:
    store = "members"
    capability = "live"
    workspace_scoped = False

    async def search(self, q, limit, ctx):
        return [
            SearchHit(
                id="m1",
                store="members",
                title="alice",
                detail="Role: admin",
                score=1.0,
                locator="route:/workspace/members",
            )
        ]


def _build_app(monkeypatch, *, user=None, member=True):
    user = user or SimpleNamespace(id="user1", is_admin=False)
    user_repo = SimpleNamespace(get=lambda uid: user)
    ws_repo = SimpleNamespace(
        get=lambda ws_id: SimpleNamespace(id=ws_id),
        get_member=lambda ws_id, uid: SimpleNamespace(id="m1") if member else None,
    )
    monkeypatch.setattr(search_router, "UserRepository", lambda: user_repo)
    monkeypatch.setattr(search_router, "WorkspaceRepository", lambda: ws_repo)

    reg = SearchRegistry()
    reg.register(_FakeStore())
    monkeypatch.setattr(search_router, "_service", SearchService(reg))

    app = FastAPI()
    register_app_error_handler(app)
    app.include_router(search_router.router)
    app.dependency_overrides[require_auth_if_enabled] = lambda: _AUTH_USER
    return app


def test_requires_q(monkeypatch):
    client = TestClient(_build_app(monkeypatch))
    assert client.get("/search").status_code == 422


def test_anonymous_when_auth_disabled_serves(monkeypatch):
    # SLO_AUTH_REQUIRED unset => dependency returns None (its documented
    # anonymous mode). Search must serve identity-less callers, not 401 —
    # this exact path 401'd in the live journey walk.
    app = _build_app(monkeypatch, member=False)
    app.dependency_overrides[require_auth_if_enabled] = lambda: None
    resp = TestClient(app).get("/search", params={"q": "alice"})
    assert resp.status_code == 200
    assert resp.json()["data"]["hits"]


def test_anonymous_with_workspace_skips_membership_gate(monkeypatch):
    app = _build_app(monkeypatch, member=False)
    app.dependency_overrides[require_auth_if_enabled] = lambda: None
    resp = TestClient(app).get("/search", params={"q": "alice", "workspace_id": "ws1"})
    assert resp.status_code == 200  # open deployment: existence only


def test_returns_normalized_hits_envelope(monkeypatch):
    client = TestClient(_build_app(monkeypatch))
    resp = client.get("/search", params={"q": "alice"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["query"] == "alice"
    assert data["partial"] == []
    assert data["skipped"] == []
    hit = data["hits"][0]
    assert hit["store"] == "members"
    assert hit["title"] == "alice"
    assert hit["locator"].startswith("route:")


def test_workspace_scope_gates_membership(monkeypatch):
    # Non-member, non-admin: 403 before any store is queried.
    app = _build_app(monkeypatch, member=False)
    resp = TestClient(app).get("/search", params={"q": "alice", "workspace_id": "ws1"})
    assert resp.status_code == 403


def test_workspace_scope_allows_member(monkeypatch):
    app = _build_app(monkeypatch, member=True)
    resp = TestClient(app).get("/search", params={"q": "alice", "workspace_id": "ws1"})
    assert resp.status_code == 200


def test_workspace_scope_allows_admin(monkeypatch):
    app = _build_app(
        monkeypatch,
        user=SimpleNamespace(id="admin1", is_admin=True),
        member=False,
    )
    resp = TestClient(app).get("/search", params={"q": "alice", "workspace_id": "ws1"})
    assert resp.status_code == 200


def test_unknown_workspace_404(monkeypatch):
    app = _build_app(monkeypatch)
    search_router_ws = SimpleNamespace(
        get=lambda ws_id: None,
        get_member=lambda ws_id, uid: None,
    )
    monkeypatch.setattr(search_router, "WorkspaceRepository", lambda: search_router_ws)
    resp = TestClient(app).get("/search", params={"q": "alice", "workspace_id": "nope"})
    assert resp.status_code == 404


@pytest.mark.parametrize("bad_q", ["", "   "])
def test_blank_q_rejected(monkeypatch, bad_q):
    client = TestClient(_build_app(monkeypatch))
    assert client.get("/search", params={"q": bad_q}).status_code == 422
