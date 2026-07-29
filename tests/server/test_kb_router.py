"""
Tests for the knowledge base router.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.kb import router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestListKnowledge:
    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_returns_empty_list(self, mock_get_mem, client):
        mem = mock_get_mem.return_value
        mem.list_all.return_value = []
        resp = client.get("/knowledge")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestAddKnowledge:
    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.infrastructure.truth_labeler.get_truth_labeler")
    def test_adds_knowledge(self, mock_get_label, mock_get_mem, client):
        labeler = mock_get_label.return_value
        labeler.label.return_value = MagicMock(label="factual")
        mem = mock_get_mem.return_value
        mem.add_fact.return_value = True
        resp = client.post("/knowledge", json={"content": "Earth is round"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "stored"


class TestSearchKnowledge:
    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_searches(self, mock_get_mem, client):
        mem = mock_get_mem.return_value
        mem.search.return_value = [{"id": "1", "content": "test", "score": 0.9}]
        resp = client.get("/knowledge/search?query=test")
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 1


class TestKnowledgeStats:
    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_returns_stats(self, mock_get_mem, client):
        mem = mock_get_mem.return_value
        mem.list_all.return_value = [{"topic": "general", "source": "manual", "importance": 0.5}]
        mem._fact_counter = 1
        resp = client.get("/knowledge/stats")
        assert resp.status_code == 200
        assert resp.json()["data"]["total_items"] == 1


class TestDeleteKnowledge:
    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_deletes_existing(self, mock_get_mem, client):
        mem = mock_get_mem.return_value
        mem.delete_by_id.return_value = True
        resp = client.delete("/knowledge/some-id")
        assert resp.status_code == 200

    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_returns_404_for_missing(self, mock_get_mem, client):
        mem = mock_get_mem.return_value
        mem.delete_by_id.return_value = False
        resp = client.delete("/knowledge/nonexistent")
        assert resp.status_code == 404


class TestListTopics:
    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_returns_topics(self, mock_get_mem, client):
        mem = mock_get_mem.return_value
        mem.list_all.return_value = [{"topic": "general"}]
        resp = client.get("/knowledge/topics")
        assert resp.status_code == 200
        assert resp.json()["data"]["total"] >= 1


class TestGetContext:
    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_returns_context(self, mock_get_mem, client):
        mem = mock_get_mem.return_value
        mem.get_context_string.return_value = "context"
        mem.list_all.return_value = [{"id": "1"}]
        resp = client.get("/knowledge/context")
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 1


class TestSuggestTopic:
    def test_suggests_topic(self, client):
        resp = client.post("/knowledge/suggest-topic", json={"content": "function def foo"})
        assert resp.status_code == 200
        assert resp.json()["data"]["topic"] == "code"


class TestLabelText:
    @patch("domains.infrastructure.truth_labeler.get_truth_labeler")
    def test_labels_text(self, mock_get_label, client):
        labeler = mock_get_label.return_value
        labeler.label.return_value = MagicMock(to_dict=lambda: {"label": "factual"})
        resp = client.get("/knowledge/label?text=hello")
        assert resp.status_code == 200
        assert resp.json()["data"]["label"] == "factual"
