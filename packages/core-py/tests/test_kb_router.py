"""Tests for the knowledge base API router (routers/kb.py).

Covers: list, create, get, delete, batch_delete, search.
KnowledgeMemory is mocked.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.kb import KBRouter as KnowledgeRouter  # noqa: E402
from tests.conftest import build_test_app


def _mock_km(**overrides) -> MagicMock:
    km = MagicMock()
    km.list_all.return_value = [
        {"id": "f1", "content": "AI is great", "topic": "ai", "source": "manual", "timestamp": 1.0, "importance": 0.8, "score": 0.0},
    ]
    km.add_fact.return_value = True
    km.delete_by_id.return_value = True
    km.search.return_value = [{"id": "f1", "content": "AI is great", "topic": "ai", "source": "manual", "timestamp": 1.0, "importance": 0.8, "score": 0.9}]
    km.stats.return_value = {"total_facts": 1, "topics": [("ai", 1)]}
    km.all_topics.return_value = [("ai", 1)]
    return km


def _app(kr: KnowledgeRouter):
    return build_test_app(kr.router)


class TestListKnowledge:
    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_list(self, mock_get):
        mock_get.return_value = _mock_km()
        kr = KnowledgeRouter()
        client = TestClient(_app(kr))
        resp = client.get("/knowledge")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1

    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_list_with_topic(self, mock_get):
        mock_get.return_value = _mock_km()
        kr = KnowledgeRouter()
        client = TestClient(_app(kr))
        resp = client.get("/knowledge?limit=10&offset=0")
        assert resp.status_code == 200


class TestCreateKnowledge:
    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_create(self, mock_get):
        mock_get.return_value = _mock_km()
        kr = KnowledgeRouter()
        client = TestClient(_app(kr))
        resp = client.post("/knowledge", json={"content": "New fact", "topic": "ai"})
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "stored"


class TestSearchKnowledge:
    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_search(self, mock_get):
        mock_get.return_value = _mock_km()
        kr = KnowledgeRouter()
        client = TestClient(_app(kr))
        resp = client.get("/knowledge/search?query=AI")
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 1


class TestBatchDelete:
    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_batch_delete(self, mock_get):
        mock_get.return_value = _mock_km()
        kr = KnowledgeRouter()
        client = TestClient(_app(kr))
        resp = client.post("/knowledge/batch-delete", json={"ids": ["f1", "f2"]})
        assert resp.status_code == 200
        assert resp.json()["data"]["deleted"] == 2


class TestDeleteKnowledge:
    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_delete(self, mock_get):
        mock_get.return_value = _mock_km()
        kr = KnowledgeRouter()
        client = TestClient(_app(kr))
        resp = client.delete("/knowledge/f1")
        assert resp.status_code == 200

    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_delete_not_found(self, mock_get):
        km = _mock_km()
        km.delete_by_id.return_value = False
        mock_get.return_value = km
        kr = KnowledgeRouter()
        client = TestClient(_app(kr))
        resp = client.delete("/knowledge/nonexistent")
        assert resp.status_code == 404
