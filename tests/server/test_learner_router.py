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
        assert "url required" in body["error"]

    @patch("domains.learner.get_learner")
    def test_feed_unknown_action(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        resp = client.post("/learn/feed?action=invalid_action")
        assert resp.status_code == 200
        body = resp.json()
        assert "unknown action" in body["error"]

    @patch("domains.learner.get_learner")
    def test_feed_unsubscribe_missing_url(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        resp = client.post("/learn/feed?action=unsubscribe")
        assert resp.status_code == 200
        body = resp.json()
        assert "url required" in body["error"]


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


class TestLearnIngestUrl:
    """POST /learn/ingest-url — single URL scraping."""

    @patch("domains.learner.get_learner")
    def test_ingests_url(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.ingest_url.return_value = {"status": "ok", "facts": 3}
        resp = client.post("/learn/ingest-url?url=https://example.com/article")
        assert resp.status_code == 200
        assert resp.json()["data"]["facts"] == 3
        learner.ingest_url.assert_called_once_with("https://example.com/article")

    @patch("domains.learner.get_learner")
    def test_ingest_url_result_forwarded(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.ingest_url.return_value = {"status": "error", "message": "fetch failed"}
        resp = client.post("/learn/ingest-url?url=https://bad.example")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "error"


class TestLearnIngestConversations:
    """POST /learn/ingest — conversation pairs (bare-array request body)."""

    @patch("domains.learner.get_learner")
    def test_ingests_conversation_pairs(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.status.return_value = {"current_loss": 0.5}
        resp = client.post("/learn/ingest", json=[
            ["hi", "hello"], ["what is ai", "ai is..."],
        ])
        assert resp.status_code == 200
        learner.ingest_conversation.assert_called_once_with([("hi", "hello"), ("what is ai", "ai is...")])

    @patch("domains.learner.get_learner")
    def test_ingest_skips_short_pairs(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.status.return_value = {"current_loss": 0.5}
        resp = client.post("/learn/ingest", json=[
            ["only_one"], ["a", "b"],
        ])
        assert resp.status_code == 200
        learner.ingest_conversation.assert_called_once_with([("a", "b")])

    @patch("domains.learner.get_learner")
    def test_ingest_bad_body_422(self, mock_get_learner, client):
        resp = client.post("/learn/ingest", json={"conversations": [["q", "a"]]})
        assert resp.status_code == 422

    @patch("domains.learner.get_learner")
    def test_ingest_neither_returns_status(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.status.return_value = {"current_loss": 0.5}
        resp = client.post("/learn/ingest")
        assert resp.status_code == 200
        assert resp.json()["data"]["current_loss"] == 0.5


class TestLearnValidation:
    """Validation and method-mismatch coverage."""

    @patch("domains.learner.get_learner")
    def test_search_missing_query_422(self, mock_get_learner, client):
        resp = client.post("/learn/search", json={})
        assert resp.status_code == 422

    @patch("domains.learner.get_learner")
    def test_ingest_url_missing_param_422(self, mock_get_learner, client):
        resp = client.post("/learn/ingest-url")
        assert resp.status_code == 422

    @patch("domains.learner.get_learner")
    def test_feed_action_too_long_422(self, mock_get_learner, client):
        resp = client.post("/learn/feed?action=" + "x" * 21)
        assert resp.status_code == 422

    @patch("domains.learner.get_learner")
    def test_feed_poll_interval_below_min_422(self, mock_get_learner, client):
        resp = client.post("/learn/feed?action=list&poll_interval=59")
        assert resp.status_code == 422

    @patch("domains.learner.get_learner")
    def test_knowledge_top_k_above_max_422(self, mock_get_learner, client):
        resp = client.get("/learn/knowledge?query=x&top_k=101")
        assert resp.status_code == 422

    @patch("domains.learner.get_learner")
    def test_feed_subscribe_success(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.subscribe_feed.return_value = True
        learner.list_feeds.return_value = [{"url": "https://rss.example"}]
        resp = client.post("/learn/feed?action=subscribe&url=https://rss.example")
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["status"] == "ok"
        assert body["feeds"][0]["url"] == "https://rss.example"
        learner.subscribe_feed.assert_called_once_with("https://rss.example", 3600)

    @patch("domains.learner.get_learner")
    def test_feed_subscribe_already_subscribed(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.subscribe_feed.return_value = False
        learner.list_feeds.return_value = []
        resp = client.post("/learn/feed?action=subscribe&url=https://rss.example")
        assert resp.json()["data"]["status"] == "already_subscribed"

    @patch("domains.learner.get_learner")
    def test_feed_unsubscribe_success(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.unsubscribe_feed.return_value = True
        learner.list_feeds.return_value = []
        resp = client.post("/learn/feed?action=unsubscribe&url=https://rss.example")
        assert resp.json()["data"]["status"] == "ok"

    @patch("domains.learner.get_learner")
    def test_feed_unsubscribe_not_found(self, mock_get_learner, client):
        learner = mock_get_learner.return_value
        learner.unsubscribe_feed.return_value = False
        learner.list_feeds.return_value = []
        resp = client.post("/learn/feed?action=unsubscribe&url=https://rss.example")
        assert resp.json()["data"]["status"] == "not_found"

    def test_search_wrong_method_405(self, client):
        resp = client.get("/learn/search")
        assert resp.status_code == 405

    def test_status_wrong_method_405(self, client):
        resp = client.post("/learn/status")
        assert resp.status_code == 405

    def test_knowledge_wrong_method_405(self, client):
        resp = client.post("/learn/knowledge")
        assert resp.status_code == 405

    def test_train_wrong_method_405(self, client):
        resp = client.get("/learn/train")
        assert resp.status_code == 405
