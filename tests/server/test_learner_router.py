"""
Tests for the learner router — search, feed, ingest, train, deploy, evaluate, status.
"""

import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.learner import router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestLearnSearch:
    @patch("domains.learner.get_learner")
    def test_searches_and_learns(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.search_and_learn.return_value = {"tokens_ingested": 100, "new_facts": 5, "rejected": 0, "filter_stats": {}}
        learner.status.return_value = {"current_loss": 0.5}
        resp = client.post("/learn/search", json={"query": "AI safety", "max_results": 3})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["tokens_ingested"] == 100

    @patch("domains.learner.get_learner")
    def test_search_empty_query(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.search_and_learn.return_value = {"tokens_ingested": 0, "new_facts": 0, "rejected": 0, "filter_stats": {}}
        learner.status.return_value = {}
        resp = client.post("/learn/search", json={"query": ""})
        assert resp.status_code == 200
        learner.search_and_learn.assert_called_once_with("", 5)

    @patch("domains.learner.get_learner")
    def test_search_rejected_facts(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.search_and_learn.return_value = {"tokens_ingested": 50, "new_facts": 0, "rejected": 10, "filter_stats": {"low_quality": 10}}
        learner.status.return_value = {}
        resp = client.post("/learn/search", json={"query": "spam query"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["rejected"] == 10
        assert data["filter_stats"]["low_quality"] == 10


class TestLearnFeed:
    @patch("domains.learner.get_learner")
    def test_list_feeds(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.list_feeds.return_value = []
        resp = client.post("/learn/feed?action=list")
        assert resp.status_code == 200
        assert resp.json()["data"]["feeds"] == []

    @patch("domains.learner.get_learner")
    def test_feed_subscribe_missing_url(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        resp = client.post("/learn/feed?action=subscribe")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "error"
        assert "url required" in body["message"]

    @patch("domains.learner.get_learner")
    def test_feed_unknown_action(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        resp = client.post("/learn/feed?action=invalid_action")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "error"
        assert "unknown action" in body["message"]

    @patch("domains.learner.get_learner")
    def test_feed_unsubscribe_missing_url(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        resp = client.post("/learn/feed?action=unsubscribe")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "error"
        assert "url required" in body["message"]


class TestLearnKnowledge:
    @patch("domains.learner.get_learner")
    def test_queries_by_topic(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.query_knowledge.return_value = [{"content": "test"}]
        resp = client.get("/learn/knowledge?topic=general")
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] >= 1

    @patch("domains.learner.get_learner")
    def test_knowledge_no_params_returns_empty(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        resp = client.get("/learn/knowledge")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["facts"] == []
        assert data["count"] == 0

    @patch("domains.learner.get_learner")
    def test_knowledge_query_search(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.search_knowledge.return_value = [{"content": "result"}]
        resp = client.get("/learn/knowledge?query=neural+network&top_k=5")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["count"] == 1
        learner.search_knowledge.assert_called_once_with("neural network", top_k=5)


class TestLearnStatus:
    @patch("domains.learner.get_learner")
    def test_returns_status(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.status.return_value = {"current_loss": 0.5, "train_steps_completed": 10}
        resp = client.get("/learn/status")
        assert resp.status_code == 200
        assert resp.json()["data"]["current_loss"] == 0.5

    @patch("domains.learner.get_learner")
    def test_status_after_training(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.status.return_value = {
            "current_loss": 0.1,
            "train_steps_completed": 100,
            "buffer_size": 500,
            "total_tokens_ingested": 10000,
        }
        resp = client.get("/learn/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["train_steps_completed"] == 100
        assert data["buffer_size"] == 500


class TestLearnIngest:
    @patch("domains.learner.get_learner")
    def test_ingests_text(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.status.return_value = {"current_loss": 0.5}
        resp = client.post("/learn/ingest?text=hello")
        assert resp.status_code == 200

    @patch("domains.learner.get_learner")
    def test_ingest_conversations_via_query(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.status.return_value = {"current_loss": 0.5}
        resp = client.post("/learn/ingest?text=some+text")
        assert resp.status_code == 200
        learner.ingest_text.assert_called_once_with("some text")


class TestLearnTrain:
    @patch("domains.learner.get_learner")
    def test_trains(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.train_now.return_value = {"current_loss": 0.3, "steps_completed": 1}
        resp = client.post("/learn/train")
        assert resp.status_code == 200
        assert resp.json()["data"]["current_loss"] == 0.3

    @patch("domains.learner.get_learner")
    def test_train_returns_loss_and_steps(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.train_now.return_value = {"current_loss": 0.05, "steps_completed": 50, "epochs": 3}
        resp = client.post("/learn/train")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["current_loss"] == 0.05
        assert data["steps_completed"] == 50


class TestLearnDeploy:
    @patch("domains.learner.get_learner")
    def test_deploys(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.deploy.return_value = {"path": "/tmp/model.soul", "steps": 10}
        resp = client.post("/learn/deploy?name=test")
        assert resp.status_code == 200
        assert resp.json()["data"]["path"] == "/tmp/model.soul"

    @patch("domains.learner.get_learner")
    def test_deploy_without_name(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.deploy.return_value = {"path": "/tmp/default.soul", "soul_name": "learner"}
        resp = client.post("/learn/deploy")
        assert resp.status_code == 200
        learner.deploy.assert_called_once_with(name=None)


class TestLearnEvaluate:
    @patch("domains.learner.get_learner")
    def test_evaluates(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.evaluate.return_value = {"loss": 0.5, "perplexity": 1.6}
        resp = client.post("/learn/evaluate")
        assert resp.status_code == 200

    @patch("domains.learner.get_learner")
    def test_evaluate_with_text(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.evaluate.return_value = {"loss": 0.3, "perplexity": 1.35, "eval_tokens": 50}
        resp = client.post("/learn/evaluate?text=hello+world")
        assert resp.status_code == 200
        learner.evaluate.assert_called_once_with(text="hello world")
