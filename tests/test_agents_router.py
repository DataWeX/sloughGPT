"""
Tests for the agents router CRUD endpoints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SERVER_DIR = _REPO_ROOT / "apps" / "api" / "server"
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))


TestClient = pytest.importorskip("fastapi.testclient").TestClient


@pytest.fixture(scope="module")
def client():
    from main import app
    return app, TestClient(app)


def test_list_agents(client):
    app, tc = client
    resp = tc.get("/agents")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    # Default agents are pre-seeded
    assert len(data) >= 2
    names = [a["name"] for a in data]
    assert "Researcher" in names
    assert "Writer" in names


def test_create_agent(client):
    app, tc = client
    resp = tc.post("/agents", json={"name": "TestBot", "description": "A test agent"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "TestBot"
    assert data["id"] == "testbot"


def test_create_duplicate_returns_409(client):
    app, tc = client
    resp = tc.post("/agents", json={"name": "TestBot"})
    assert resp.status_code == 409


def test_get_agent(client):
    app, tc = client
    resp = tc.get("/agents/testbot")
    assert resp.status_code == 200
    assert resp.json()["name"] == "TestBot"


def test_get_agent_not_found_returns_404(client):
    app, tc = client
    resp = tc.get("/agents/nonexistent")
    assert resp.status_code == 404


def test_update_agent(client):
    app, tc = client
    resp = tc.put("/agents/testbot", json={"description": "Updated description"})
    assert resp.status_code == 200
    assert resp.json()["description"] == "Updated description"


def test_update_agent_not_found_returns_404(client):
    app, tc = client
    resp = tc.put("/agents/nonexistent", json={"name": "Ghost"})
    assert resp.status_code == 404


def test_delete_agent(client):
    app, tc = client
    resp = tc.delete("/agents/testbot")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"


def test_delete_agent_not_found_returns_404(client):
    app, tc = client
    resp = tc.delete("/agents/nonexistent")
    assert resp.status_code == 404
