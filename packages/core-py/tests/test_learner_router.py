"""Tests for the learner API router (routers/learner.py).

Covers: LearnerRouter CRUD, search, feed, ingest, train, deploy, evaluate, status.
All domain calls are mocked; only HTTP-level behavior is tested.

Note: the learner router imports get_learner INSIDE each handler function body,
so we must patch 'domains.learner.get_learner' (the import target), not
'routers.learner.get_learner'.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — identical to the other router test files
# ---------------------------------------------------------------------------
_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.learner import router  # noqa: E402
from tests.conftest import build_test_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_learner(**overrides):
    """Build a fake learner object with all methods the router calls."""
    defaults = dict(
        search_and_learn=lambda q, n: {
            "tokens_ingested": 100,
            "new_facts": 5,
            "rejected": 1,
            "filter_stats": {"removed": 1},
        },
        status=lambda: {
            "soul_name": "test",
            "total_tokens_ingested": 500,
            "train_steps_completed": 10,
            "current_loss": 2.5,
            "buffer_size": 200,
            "pending_tokens": 50,
        },
        subscribe_feed=lambda url, interval: True,
        unsubscribe_feed=lambda url: True,
        list_feeds=lambda: ["http://example.com/rss"],
        ingest_url=lambda url: {"facts": 3, "status": "ok"},
        query_knowledge=lambda topic: [{"content": "fact1", "topic": topic}],
        search_knowledge=lambda q, top_k=10: [{"content": "fact1"}],
        ingest_text=lambda text: None,
        ingest_conversation=lambda pairs: None,
        train_now=lambda: {"loss": 1.2, "steps": 1},
        deploy=lambda name=None: {"path": "/tmp/model.soul", "soul_name": "test", "steps": 10, "loss": 1.0, "file_size": 1024},
        evaluate=lambda text=None: {"loss": 2.0, "perplexity": 7.4, "eval_tokens": 100, "train_steps": 10, "total_tokens_ingested": 500},
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _app():
    return build_test_app(router)


# ---------------------------------------------------------------------------
# Tests — patch at 'domains.learner.get_learner' (lazy import inside handler)
# ---------------------------------------------------------------------------

MOCK_TARGET = "domains.learner.get_learner"


class TestLearnSearch:
    @patch(MOCK_TARGET)
    def test_search_returns_tokens_and_facts(self, mock_get):
        mock_get.return_value = _make_learner()
        client = TestClient(_app())
        resp = client.post("/learn/search", json={"query": "python", "max_results": 3})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        data = body["data"]
        assert data["tokens_ingested"] == 100
        assert data["new_facts"] == 5
        assert data["rejected"] == 1

    @patch(MOCK_TARGET)
    def test_search_includes_status_fields(self, mock_get):
        mock_get.return_value = _make_learner()
        client = TestClient(_app())
        resp = client.post("/learn/search", json={"query": "test"})
        data = resp.json()["data"]
        assert "soul_name" in data
        assert "total_tokens_ingested" in data


class TestLearnFeed:
    @patch(MOCK_TARGET)
    def test_subscribe_adds_feed(self, mock_get):
        mock_get.return_value = _make_learner()
        client = TestClient(_app())
        resp = client.post("/learn/feed", params={"action": "subscribe", "url": "http://example.com/rss"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "ok"
        assert "feeds" in data

    @patch(MOCK_TARGET)
    def test_subscribe_without_url_returns_error(self, mock_get):
        mock_get.return_value = _make_learner()
        client = TestClient(_app())
        resp = client.post("/learn/feed", params={"action": "subscribe"})
        assert resp.status_code == 422
        assert resp.json()["code"] == "E_VAL_REQUEST"

    @patch(MOCK_TARGET)
    def test_unsubscribe_removes_feed(self, mock_get):
        mock_get.return_value = _make_learner()
        client = TestClient(_app())
        resp = client.post("/learn/feed", params={"action": "unsubscribe", "url": "http://example.com/rss"})
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "ok"

    @patch(MOCK_TARGET)
    def test_list_feeds(self, mock_get):
        mock_get.return_value = _make_learner()
        client = TestClient(_app())
        resp = client.post("/learn/feed", params={"action": "list"})
        assert resp.status_code == 200
        assert resp.json()["data"]["feeds"] == ["http://example.com/rss"]

    @patch(MOCK_TARGET)
    def test_unknown_action_returns_error(self, mock_get):
        mock_get.return_value = _make_learner()
        client = TestClient(_app())
        resp = client.post("/learn/feed", params={"action": "bogus"})
        assert resp.status_code == 422
        assert resp.json()["code"] == "E_VAL_REQUEST"


class TestLearnIngestUrl:
    @patch(MOCK_TARGET)
    def test_ingest_url_returns_result(self, mock_get):
        mock_get.return_value = _make_learner()
        client = TestClient(_app())
        resp = client.post("/learn/ingest-url", params={"url": "http://example.com/article"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["facts"] == 3
        assert data["status"] == "ok"


class TestLearnKnowledge:
    @patch(MOCK_TARGET)
    def test_query_by_topic(self, mock_get):
        mock_get.return_value = _make_learner()
        client = TestClient(_app())
        resp = client.get("/learn/knowledge", params={"topic": "python"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["count"] == 1
        assert data["facts"][0]["topic"] == "python"

    @patch(MOCK_TARGET)
    def test_search_by_query(self, mock_get):
        mock_get.return_value = _make_learner()
        client = TestClient(_app())
        resp = client.get("/learn/knowledge", params={"query": "python"})
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 1

    @patch(MOCK_TARGET)
    def test_no_topic_or_query_returns_empty(self, mock_get):
        mock_get.return_value = _make_learner()
        client = TestClient(_app())
        resp = client.get("/learn/knowledge")
        assert resp.status_code == 200
        assert resp.json()["data"]["facts"] == []


class TestLearnIngest:
    @patch(MOCK_TARGET)
    def test_ingest_text(self, mock_get):
        mock_get.return_value = _make_learner()
        client = TestClient(_app())
        resp = client.post("/learn/ingest", params={"text": "hello world"})
        assert resp.status_code == 200
        assert "soul_name" in resp.json()["data"]

    @patch(MOCK_TARGET)
    def test_ingest_conversations(self, mock_get):
        mock_get.return_value = _make_learner()
        client = TestClient(_app())
        resp = client.post("/learn/ingest", params={"conversations": [["hi", "hello"]]})
        assert resp.status_code == 200


class TestLearnTrain:
    @patch(MOCK_TARGET)
    def test_train_returns_status(self, mock_get):
        mock_get.return_value = _make_learner()
        client = TestClient(_app())
        resp = client.post("/learn/train")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["loss"] == 1.2
        assert data["steps"] == 1


class TestLearnDeploy:
    @patch(MOCK_TARGET)
    def test_deploy_returns_path(self, mock_get):
        mock_get.return_value = _make_learner()
        client = TestClient(_app())
        resp = client.post("/learn/deploy")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "path" in data
        assert data["soul_name"] == "test"


class TestLearnEvaluate:
    @patch(MOCK_TARGET)
    def test_evaluate_returns_metrics(self, mock_get):
        mock_get.return_value = _make_learner()
        client = TestClient(_app())
        resp = client.post("/learn/evaluate")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "loss" in data
        assert "perplexity" in data


class TestLearnStatus:
    @patch(MOCK_TARGET)
    def test_status_returns_all_fields(self, mock_get):
        mock_get.return_value = _make_learner()
        client = TestClient(_app())
        resp = client.get("/learn/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "soul_name" in data
        assert "total_tokens_ingested" in data
        assert "current_loss" in data
