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


class TestLearnFeed:
    @patch("domains.learner.get_learner")
    def test_list_feeds(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.list_feeds.return_value = []
        resp = client.post("/learn/feed?action=list")
        assert resp.status_code == 200
        assert resp.json()["data"]["feeds"] == []


class TestLearnKnowledge:
    @patch("domains.learner.get_learner")
    def test_queries_by_topic(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.query_knowledge.return_value = [{"content": "test"}]
        resp = client.get("/learn/knowledge?topic=general")
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] >= 1


class TestLearnStatus:
    @patch("domains.learner.get_learner")
    def test_returns_status(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.status.return_value = {"current_loss": 0.5, "train_steps_completed": 10}
        resp = client.get("/learn/status")
        assert resp.status_code == 200
        assert resp.json()["data"]["current_loss"] == 0.5


class TestLearnIngest:
    @patch("domains.learner.get_learner")
    def test_ingests_text(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.status.return_value = {"current_loss": 0.5}
        resp = client.post("/learn/ingest?text=hello")
        assert resp.status_code == 200


class TestLearnTrain:
    @patch("domains.learner.get_learner")
    def test_trains(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.train_now.return_value = {"current_loss": 0.3, "steps_completed": 1}
        resp = client.post("/learn/train")
        assert resp.status_code == 200
        assert resp.json()["data"]["current_loss"] == 0.3


class TestLearnDeploy:
    @patch("domains.learner.get_learner")
    def test_deploys(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.deploy.return_value = {"path": "/tmp/model.soul", "steps": 10}
        resp = client.post("/learn/deploy?name=test")
        assert resp.status_code == 200
        assert resp.json()["data"]["path"] == "/tmp/model.soul"


class TestLearnEvaluate:
    @patch("domains.learner.get_learner")
    def test_evaluates(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.evaluate.return_value = {"loss": 0.5, "perplexity": 1.6}
        resp = client.post("/learn/evaluate")
        assert resp.status_code == 200
