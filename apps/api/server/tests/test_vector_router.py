"""Tests for the /vector router.

Covers all 5 endpoints:
  POST /vector/init          — init_vector_store
  GET  /vector/stats         — get_stats
  POST /vector/upsert        — upsert_vectors
  POST /vector/search        — search_vectors
  GET  /vector/ingest/status — ingest_status
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from test_support import get_test_client


def _d(resp):
    """Unwrap success_response envelope."""
    j = resp.json()
    return j.get("data", j)


def _mock_vector_store():
    """Create a mock vector store."""
    m = AsyncMock()
    m.count.return_value = 0
    m.upsert.return_value = 2
    m.query.return_value = []
    return m


@pytest.fixture(autouse=True)
def _fresh_vector_router():
    """Reset the vector router state before each test."""
    import routers.vector as vec_mod

    # Find the VectorRouter instance through the module
    for attr_name in dir(vec_mod):
        attr = getattr(vec_mod, attr_name, None)
        if isinstance(attr, vec_mod.VectorRouter):
            attr._vector_store = None
            attr._vector_store_type = "in_memory"
            break
    yield


class TestInitVectorStore:
    def setup_method(self):
        self.client = get_test_client()

    @patch("domains.inference.vector_store.create_vector_store", new_callable=AsyncMock)
    def test_init_in_memory(self, mock_create):
        mock_create.return_value = _mock_vector_store()
        resp = self.client.post("/vector/init", json={"provider": "in_memory", "dimension": 384})
        assert resp.status_code == 200
        data = _d(resp)
        assert data["status"] == "connected"
        assert data["provider"] == "in_memory"

    @patch("domains.inference.vector_store.create_vector_store", new_callable=AsyncMock)
    def test_init_default_provider(self, mock_create):
        mock_create.return_value = _mock_vector_store()
        resp = self.client.post("/vector/init", json={})
        assert resp.status_code == 200
        data = _d(resp)
        assert data["status"] == "connected"


class TestGetStats:
    def setup_method(self):
        self.client = get_test_client()

    def test_stats_empty_store(self):
        resp = self.client.get("/vector/stats")
        assert resp.status_code == 200
        data = _d(resp)
        assert "provider" in data
        assert "count" in data

    @patch("domains.inference.vector_store.create_vector_store", new_callable=AsyncMock)
    def test_stats_with_store(self, mock_create):
        store = _mock_vector_store()
        store.count.return_value = 42
        mock_create.return_value = store
        # Initialize first
        self.client.post("/vector/init", json={"provider": "in_memory"})
        resp = self.client.get("/vector/stats")
        assert resp.status_code == 200
        data = _d(resp)
        assert data["count"] == 42


class TestUpsertVectors:
    def setup_method(self):
        self.client = get_test_client()

    def test_upsert_missing_texts(self):
        resp = self.client.post("/vector/upsert", json={})
        assert resp.status_code == 422

    @patch("domains.inference.vector_store.create_vector_store", new_callable=AsyncMock)
    def test_upsert_with_texts(self, mock_create):
        store = _mock_vector_store()
        mock_create.return_value = store
        self.client.post("/vector/init", json={"provider": "in_memory"})
        resp = self.client.post(
            "/vector/upsert",
            json={"texts": ["hello", "world"], "ids": ["1", "2"]},
        )
        assert resp.status_code == 200
        data = _d(resp)
        assert data["status"] == "upserted"
        assert data["count"] == 2

    @patch("domains.inference.vector_store.create_vector_store", new_callable=AsyncMock)
    def test_upsert_without_store(self, mock_create):
        mock_create.return_value = _mock_vector_store()
        resp = self.client.post(
            "/vector/upsert", json={"texts": ["hello"]}
        )
        assert resp.status_code == 200


class TestSearchVectors:
    def setup_method(self):
        self.client = get_test_client()

    def test_search_missing_query(self):
        resp = self.client.post("/vector/search", json={})
        assert resp.status_code == 422

    @patch("domains.inference.vector_store.create_vector_store", new_callable=AsyncMock)
    def test_search_with_query(self, mock_create):
        store = _mock_vector_store()
        mock_create.return_value = store
        self.client.post("/vector/init", json={"provider": "in_memory"})
        resp = self.client.post(
            "/vector/search", json={"query": "hello world", "top_k": 3}
        )
        assert resp.status_code == 200
        data = _d(resp)
        assert "results" in data
        assert "elapsed_ms" in data

    def test_search_without_store(self):
        resp = self.client.post("/vector/search", json={"query": "test"})
        assert resp.status_code == 200
        data = _d(resp)
        assert data["results"] == []


class TestIngestStatus:
    def setup_method(self):
        self.client = get_test_client()

    def test_returns_ready(self):
        resp = self.client.get("/vector/ingest/status")
        assert resp.status_code == 200
        data = _d(resp)
        assert data["status"] == "ready"


class TestVectorLifecycle:
    """End-to-end: init → upsert → search → stats."""

    def setup_method(self):
        self.client = get_test_client()

    @patch("domains.inference.vector_store.create_vector_store", new_callable=AsyncMock)
    def test_full_lifecycle(self, mock_create):
        store = _mock_vector_store()
        store.count.return_value = 2
        mock_create.return_value = store

        # Init
        resp = self.client.post("/vector/init", json={"provider": "in_memory"})
        assert resp.status_code == 200

        # Upsert
        resp = self.client.post(
            "/vector/upsert", json={"texts": ["hello", "world"]}
        )
        assert resp.status_code == 200

        # Search
        resp = self.client.post("/vector/search", json={"query": "hello"})
        assert resp.status_code == 200

        # Stats
        resp = self.client.get("/vector/stats")
        assert resp.status_code == 200

        # Ingest status
        resp = self.client.get("/vector/ingest/status")
        assert resp.status_code == 200
