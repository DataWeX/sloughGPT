"""
Integration tests for labs knowledge endpoints and health-status.
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_knowledge():
    """Clear knowledge singleton between tests for isolation."""
    from domains.learner.knowledge import get_knowledge_memory
    km = get_knowledge_memory()
    km.clear_all()
    yield


class TestKnowledgeCRUD:
    BASE = "/labs/knowledge"

    def test_list_empty(self, client):
        r = client.get(self.BASE)
        assert r.status_code == 200
        data = r.json()
        assert "items" in data
        assert data["count"] == 0

    def test_add_and_list(self, client):
        r = client.post(self.BASE, json={"content": "The sky is blue", "category": "science"})
        assert r.status_code == 200
        assert r.json()["stored"] is True

        r = client.get(self.BASE)
        assert r.status_code == 200
        data = r.json()
        assert data["count"] >= 1
        items = data["items"]
        assert any("the sky is blue" in i.get("content", "").lower() for i in items)

    def test_duplicate_not_stored(self, client):
        r = client.post(self.BASE, json={"content": "Unique fact here", "category": "test"})
        assert r.json()["stored"] is True
        r = client.post(self.BASE, json={"content": "Unique fact here", "category": "test"})
        assert r.json()["stored"] is False

    def test_delete_all(self, client):
        client.post(self.BASE, json={"content": "A", "category": "x"})
        client.post(self.BASE, json={"content": "B", "category": "x"})
        r = client.delete(self.BASE)
        assert r.status_code == 200
        assert r.json()["cleared"] is True

        r = client.get(self.BASE)
        assert len(r.json()["items"]) == 0

    def test_search(self, client):
        client.post(self.BASE, json={"content": "Python is a programming language", "category": "tech"})
        r = client.get(f"{self.BASE}/search", params={"query": "python", "category": "tech"})
        assert r.status_code == 200
        data = r.json()
        assert "items" in data


class TestHealthStatus:
    def test_health_status_returns_all_sections(self, client):
        r = client.get("/labs/health-status")
        assert r.status_code == 200
        data = r.json()
        assert "available" in data
        assert "model" in data
        assert "workflow" in data
        assert "adapters" in data

    def test_health_status_available_flag(self, client):
        r = client.get("/labs/health-status")
        assert r.json()["available"] is True


class TestAdapterToggle:
    def test_user_id_enabled_parameter(self, client):
        r = client.post("/labs/chat", json={
            "prompt": "hello",
            "user_id": "test_user",
            "user_id_enabled": False,
        })
        assert r.status_code in (200, 400)
