"""Tests for the knowledge base API router (routers/kb.py).

Covers: list, create, get, delete, batch_delete, search.
KnowledgeMemory is mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.kb import KBRouter as KnowledgeRouter  # noqa: E402

# Bind domain.knowledge.engine into the attr chain: py3.12 mock.patch resolves targets
# via pkgutil.resolve_name, which is getattr-only — the lazy import chain
# (domain.knowledge -> .engine) must already be materialised or every decorator below
# dies with AttributeError at decoration time.
import domain.knowledge.engine  # noqa: E402,F401
from tests.conftest import build_test_app


def _mock_km(**overrides) -> MagicMock:
    km = MagicMock()
    km.list_all.return_value = [
        {
            "id": "f1",
            "content": "AI is great",
            "topic": "ai",
            "source": "manual",
            "timestamp": 1.0,
            "importance": 0.8,
            "score": 0.0,
        },
    ]
    km.add_fact.return_value = True
    km.delete_by_id.return_value = True
    km.search.return_value = [
        {
            "id": "f1",
            "content": "AI is great",
            "topic": "ai",
            "source": "manual",
            "timestamp": 1.0,
            "importance": 0.8,
            "score": 0.9,
        }
    ]
    km.stats.return_value = {"total_facts": 1, "topics": [("ai", 1)]}
    km.all_topics.return_value = [("ai", 1)]
    return km


def _mock_engine() -> MagicMock:
    """Engine-shaped mock: KBRouter.__init__ binds get_knowledge_engine() and every
    endpoint goes through engine.get_memory() (list_all/add_fact), engine.query
    (.success/.data), engine.batch_delete (.success/.data) and engine.delete
    (.success) — the pre-restructure get_knowledge_memory target is not in the
    chain at all (card 3ef4f1c0; consumer-read-point per 0e36c1658)."""
    engine = MagicMock()
    engine.get_memory.return_value = _mock_km()
    engine.query.return_value = SimpleNamespace(
        success=True,
        data=[
            {
                "id": "f1",
                "content": "AI is great",
                "topic": "ai",
                "source": "manual",
                "timestamp": 1.0,
                "importance": 0.8,
                "score": 0.9,
            }
        ],
    )
    engine.batch_delete.return_value = SimpleNamespace(success=True, data={"deleted": 2})
    engine.delete.return_value = SimpleNamespace(success=True)
    return engine


def _app(kr: KnowledgeRouter):
    return build_test_app(kr.router)


class TestListKnowledge:
    @patch("domain.knowledge.engine.get_knowledge_engine")
    def test_list(self, mock_get):
        mock_get.return_value = _mock_engine()
        kr = KnowledgeRouter()
        client = TestClient(_app(kr))
        resp = client.get("/knowledge")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1

    @patch("domain.knowledge.engine.get_knowledge_engine")
    def test_list_with_topic(self, mock_get):
        mock_get.return_value = _mock_engine()
        kr = KnowledgeRouter()
        client = TestClient(_app(kr))
        resp = client.get("/knowledge?limit=10&offset=0")
        assert resp.status_code == 200


class TestCreateKnowledge:
    @patch("domain.knowledge.engine.get_knowledge_engine")
    def test_create(self, mock_get):
        mock_get.return_value = _mock_engine()
        kr = KnowledgeRouter()
        client = TestClient(_app(kr))
        resp = client.post("/knowledge", json={"content": "New fact", "topic": "ai"})
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "stored"


class TestSearchKnowledge:
    @patch("domain.knowledge.engine.get_knowledge_engine")
    def test_search(self, mock_get):
        mock_get.return_value = _mock_engine()
        kr = KnowledgeRouter()
        client = TestClient(_app(kr))
        resp = client.get("/knowledge/search?query=AI")
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 1


class TestBatchDelete:
    @patch("domain.knowledge.engine.get_knowledge_engine")
    def test_batch_delete(self, mock_get):
        mock_get.return_value = _mock_engine()
        kr = KnowledgeRouter()
        client = TestClient(_app(kr))
        resp = client.post("/knowledge/batch-delete", json={"ids": ["f1", "f2"]})
        assert resp.status_code == 200
        assert resp.json()["data"]["deleted"] == 2


class TestDeleteKnowledge:
    @patch("domain.knowledge.engine.get_knowledge_engine")
    def test_delete(self, mock_get):
        mock_get.return_value = _mock_engine()
        kr = KnowledgeRouter()
        client = TestClient(_app(kr))
        resp = client.delete("/knowledge/f1")
        assert resp.status_code == 200

    @patch("domain.knowledge.engine.get_knowledge_engine")
    def test_delete_not_found(self, mock_get):
        engine = _mock_engine()
        engine.delete.return_value = SimpleNamespace(success=False)
        mock_get.return_value = engine
        kr = KnowledgeRouter()
        client = TestClient(_app(kr))
        resp = client.delete("/knowledge/nonexistent")
        assert resp.status_code == 404
