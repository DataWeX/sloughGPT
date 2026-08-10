"""Tests for vector router — init, stats, upsert, search, ingest status."""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

pytest.importorskip("fastapi")

# Ensure apps/api/server is on the path for schemas.common import
_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.vector import (
    VectorRouter, VectorStoreConfig, UpsertRequest, SearchRequest,
)


class FakeVectorStore:
    """Minimal in-memory vector store for testing."""

    def __init__(self):
        self._entries = []

    async def upsert(self, entries):
        self._entries.extend(entries)
        return len(entries)

    async def query(self, embedding, top_k=5):
        from domains.inference.vector_store import QueryResult
        return [
            QueryResult(text=e.text, score=0.9, id=e.id or str(i))
            for i, e in enumerate(self._entries[:top_k])
        ]

    async def count(self):
        return len(self._entries)


@pytest.fixture
def app():
    """Create FastAPI app with vector router."""
    router_instance = VectorRouter()
    app = FastAPI()
    app.include_router(router_instance.router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestIngestStatus:
    def test_returns_ready(self, client):
        resp = client.get("/vector/ingest/status")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "ready"


class TestGetStats:
    def test_no_store(self, client):
        resp = client.get("/vector/stats")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["count"] == 0

    def test_with_store(self, client):
        router_instance = None
        # Get the router instance to inject a mock store
        for route in client.app.routes:
            if hasattr(route, "endpoint") and hasattr(route, "methods"):
                from apps.api.server.routers.vector import VectorRouter
                # Access via the app's dependency
                break

        # Patch get_vector_store to return a fake store
        fake_store = FakeVectorStore()
        with patch("apps.api.server.routers.vector.VectorRouter.get_vector_store",
                    new_callable=AsyncMock, return_value=fake_store):
            resp = client.get("/vector/stats")
            assert resp.status_code == 200
            assert resp.json()["data"]["count"] == 0


class TestInitVectorStore:
    def test_init_in_memory(self, client):
        fake_store = FakeVectorStore()
        with patch("domains.inference.vector_store.create_vector_store",
                    new_callable=AsyncMock, return_value=fake_store):
            resp = client.post("/vector/init", json={"provider": "in_memory", "dimension": 384})
            assert resp.status_code == 200
            assert resp.json()["data"]["status"] == "connected"

    def test_init_fallback_on_import_error(self, client):
        """When requested provider fails, falls back to in_memory."""
        call_count = 0

        async def fake_create(provider, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ImportError("chromadb not installed")
            return FakeVectorStore()

        with patch("domains.inference.vector_store.create_vector_store", side_effect=fake_create):
            resp = client.post("/vector/init", json={"provider": "chromadb"})
            assert resp.status_code == 200
            assert resp.json()["data"]["provider"] == "in_memory"


class TestUpsertVectors:
    def test_upsert(self, client):
        fake_store = FakeVectorStore()
        with patch("apps.api.server.routers.vector.VectorRouter.get_vector_store",
                    new_callable=AsyncMock, return_value=fake_store):
            resp = client.post("/vector/upsert", json={
                "texts": ["hello world", "test embedding"],
                "ids": ["id1", "id2"],
            })
            assert resp.status_code == 200
            assert resp.json()["data"]["count"] == 2

    def test_upsert_no_store(self, client):
        with patch("apps.api.server.routers.vector.VectorRouter.get_vector_store",
                    new_callable=AsyncMock, return_value=None):
            resp = client.post("/vector/upsert", json={"texts": ["hello"]})
            assert resp.status_code == 500


class TestSearchVectors:
    def test_search(self, client):
        from domains.inference.vector_store import QueryResult
        fake_store = FakeVectorStore()
        fake_store._entries = [
            QueryResult(text="hello", score=0.9, id="1"),
            QueryResult(text="world", score=0.8, id="2"),
        ]
        with patch("apps.api.server.routers.vector.VectorRouter.get_vector_store",
                    new_callable=AsyncMock, return_value=fake_store):
            resp = client.post("/vector/search", json={"query": "hello", "top_k": 2})
            assert resp.status_code == 200
            results = resp.json()["data"]["results"]
            assert len(results) == 2

    def test_search_no_store(self, client):
        with patch("apps.api.server.routers.vector.VectorRouter.get_vector_store",
                    new_callable=AsyncMock, return_value=None):
            resp = client.post("/vector/search", json={"query": "hello"})
            assert resp.status_code == 200
            assert resp.json()["data"]["results"] == []
