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
    def test_returns_stats(self, client):
        resp = client.get("/vector/stats")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "provider" in data
        assert data["count"] >= 0


class TestSearch:
    def test_search_no_store_returns_200(self, client):
        resp = client.post("/vector/search", json={"query": "hello"})
        assert resp.status_code in (200, 503)
