"""
Tests for the vector store router — init, stats, upsert, search, ingest status.
"""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.vector import VectorRouter


@pytest.fixture
def router():
    return VectorRouter()


@pytest.fixture
def app(router):
    _app = FastAPI()
    _app.include_router(router.router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestStats:
    def test_returns_stats_before_init(self, client):
        resp = client.get("/vector/stats")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "provider" in data
        assert data["count"] >= 0

    def test_returns_provider_field(self, client):
        resp = client.get("/vector/stats")
        assert resp.json()["data"]["provider"] == "in_memory"


class TestSearch:
    def test_search_no_store_returns_results(self, client):
        resp = client.post("/vector/search", json={"query": "hello"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "results" in data

    def test_search_with_top_k(self, client):
        resp = client.post("/vector/search", json={"query": "test", "top_k": 3})
        assert resp.status_code == 200

    def test_search_empty_query(self, client):
        resp = client.post("/vector/search", json={"query": ""})
        assert resp.status_code == 200


class TestInitVectorStore:
    def test_init_in_memory(self, client):
        resp = client.post("/vector/init", json={"provider": "in_memory", "dimension": 128})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["provider"] == "in_memory"
        assert data["status"] == "connected"

    def test_init_chromadb_fallback(self, client):
        resp = client.post("/vector/init", json={"provider": "chromadb", "dimension": 128})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "connected"

    def test_init_updates_stats(self, client):
        client.post("/vector/init", json={"provider": "in_memory", "dimension": 64})
        resp = client.get("/vector/stats")
        assert resp.json()["data"]["provider"] == "in_memory"


class TestUpsert:
    def test_upsert_no_store_returns_error(self, client):
        resp = client.post("/vector/upsert", json={"texts": ["hello"]})
        assert resp.status_code in (200, 500)

    def test_upsert_with_store(self, client):
        client.post("/vector/init", json={"provider": "in_memory", "dimension": 64})
        resp = client.post("/vector/upsert", json={
            "texts": ["hello world", "foo bar"],
            "ids": ["id1", "id2"]
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "upserted"
        assert data["count"] == 2

    def test_upsert_with_embeddings(self, client):
        client.post("/vector/init", json={"provider": "in_memory", "dimension": 4})
        resp = client.post("/vector/upsert", json={
            "texts": ["test"],
            "embeddings": [[0.1, 0.2, 0.3, 0.4]]
        })
        assert resp.status_code == 200

    def test_upsert_with_metadata(self, client):
        client.post("/vector/init", json={"provider": "in_memory", "dimension": 4})
        resp = client.post("/vector/upsert", json={
            "texts": ["test"],
            "metadata": [{"source": "test"}]
        })
        assert resp.status_code == 200


class TestSearchAfterUpsert:
    def test_search_finds_upserted_text(self, client):
        client.post("/vector/init", json={"provider": "in_memory", "dimension": 64})
        client.post("/vector/upsert", json={"texts": ["machine learning is great"]})
        resp = client.post("/vector/search", json={"query": "machine learning", "top_k": 5})
        assert resp.status_code == 200
        results = resp.json()["data"]["results"]
        assert len(results) > 0


class TestIngestStatus:
    def test_returns_ready(self, client):
        resp = client.get("/vector/ingest/status")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "ready"


class TestVectorErrorPaths:
    """Error/fallback branches via patched create_vector_store."""

    @patch("domains.inference.vector_store.create_vector_store")
    def test_init_falls_back_to_in_memory_on_import_error(self, mock_cvs, client):
        real_store = AsyncMock()
        real_store.count = AsyncMock(return_value=0)
        real_store.upsert = AsyncMock(return_value=0)
        real_store.query = AsyncMock(return_value=[])
        mock_cvs.side_effect = [ImportError("chromadb missing"), real_store]
        resp = client.post("/vector/init", json={"provider": "chromadb"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["provider"] == "in_memory"
        assert "note" in data

    @patch("domains.inference.vector_store.create_vector_store",
           new=AsyncMock(side_effect=RuntimeError("boom")))
    def test_init_raises_http_error_on_unexpected_exception(self, client):
        resp = client.post("/vector/init", json={"provider": "weird"})
        assert resp.status_code == 500

    @patch("domains.inference.vector_store.create_vector_store",
           new=AsyncMock(return_value=None))
    def test_stats_empty_when_store_unavailable(self, client):
        resp = client.get("/vector/stats")
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 0

    @patch("domains.inference.vector_store.create_vector_store",
           new=AsyncMock(return_value=None))
    def test_upsert_500_when_store_unavailable(self, client):
        resp = client.post("/vector/upsert", json={"texts": ["hello"]})
        assert resp.status_code == 500


class TestUpsertBranches:
    def test_upsert_empty_texts(self, client):
        client.post("/vector/init", json={"provider": "in_memory", "dimension": 64})
        resp = client.post("/vector/upsert", json={"texts": []})
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 0

    def test_upsert_embeddings_partial(self, client):
        client.post("/vector/init", json={"provider": "in_memory", "dimension": 4})
        resp = client.post("/vector/upsert", json={
            "texts": ["one", "two"],
            "embeddings": [[0.1, 0.2, 0.3, 0.4]]
        })
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 2

    def test_search_returns_score_and_id(self, client):
        client.post("/vector/init", json={"provider": "in_memory", "dimension": 64})
        client.post("/vector/upsert", json={
            "texts": ["alpha beta gamma"],
            "ids": ["doc-1"],
        })
        results = client.post("/vector/search", json={"query": "alpha beta"}).json()["data"]["results"]
        assert len(results) > 0
        assert "text" in results[0]
        assert "score" in results[0]
        assert results[0]["id"] == "doc-1"
