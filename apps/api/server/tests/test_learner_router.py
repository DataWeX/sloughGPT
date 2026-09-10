"""Tests for the /learn (learner) router.

Covers all 10 endpoints:
  POST /learn/search       — learn_search
  GET  /learn/feed         — learn_feed (list)
  POST /learn/feed         — learn_feed (subscribe/unsubscribe)
  POST /learn/ingest-url   — learn_ingest_url
  GET  /learn/knowledge    — learn_knowledge
  POST /learn/ingest       — learn_ingest
  POST /learn/train        — learn_train
  POST /learn/deploy       — learn_deploy
  POST /learn/evaluate     — learn_evaluate
  GET  /learn/status       — learn_status
"""

import pytest
from unittest.mock import patch, MagicMock
from test_support import get_test_client


def _d(resp):
    """Unwrap success_response envelope."""
    j = resp.json()
    return j.get("data", j)


def _mock_learner():
    """Create a mock ContinualLearner."""
    m = MagicMock()
    m.status.return_value = {
        "soul_name": "test",
        "total_tokens_ingested": 0,
        "train_steps_completed": 0,
        "current_loss": 0.0,
        "buffer_size": 0,
    }
    m.search_and_learn.return_value = {
        "tokens_ingested": 100,
        "new_facts": 5,
        "rejected": 2,
        "filter_stats": {},
    }
    m.list_feeds.return_value = []
    m.subscribe_feed.return_value = True
    m.unsubscribe_feed.return_value = True
    m.ingest_url.return_value = {"status": "ok", "facts": 3}
    m.query_knowledge.return_value = [{"text": "fact1", "source": "test"}]
    m.search_knowledge.return_value = [{"text": "fact1", "source": "test"}]
    m.ingest_text.return_value = None
    m.ingest_conversation.return_value = None
    m.train_now.return_value = {"status": "trained"}
    m.deploy.return_value = {"path": "/tmp/test.soul", "soul_name": "test", "steps": 0, "loss": 0.0, "file_size": 1024}
    m.evaluate.return_value = {"loss": 0.5, "perplexity": 1.65, "eval_tokens": 100}
    return m


@pytest.fixture(autouse=True)
def _mock_learner_singleton():
    """Mock the learner singleton for all tests."""
    mock = _mock_learner()
    with patch("domains.learner.get_learner", return_value=mock):
        yield mock


class TestLearnStatus:
    def setup_method(self):
        self.client = get_test_client()

    def test_returns_status(self, _mock_learner_singleton):
        resp = self.client.get("/learn/status")
        assert resp.status_code == 200
        data = _d(resp)
        assert "soul_name" in data
        assert "total_tokens_ingested" in data


class TestLearnSearch:
    def setup_method(self):
        self.client = get_test_client()

    def test_search_basic(self, _mock_learner_singleton):
        resp = self.client.post("/learn/search", json={"query": "python"})
        assert resp.status_code == 200
        data = _d(resp)
        assert "new_facts" in data
        assert "tokens_ingested" in data
        assert "elapsed_ms" in data
        _mock_learner_singleton.search_and_learn.assert_called_once_with("python", 5)

    def test_search_with_max_results(self, _mock_learner_singleton):
        resp = self.client.post("/learn/search", json={"query": "test", "max_results": 10})
        assert resp.status_code == 200
        _mock_learner_singleton.search_and_learn.assert_called_once_with("test", 10)

    def test_search_missing_query(self):
        resp = self.client.post("/learn/search", json={})
        assert resp.status_code == 422

    def test_search_empty_query(self):
        resp = self.client.post("/learn/search", json={"query": ""})
        assert resp.status_code == 422


class TestLearnFeed:
    def setup_method(self):
        self.client = get_test_client()

    def test_list_feeds(self, _mock_learner_singleton):
        resp = self.client.get("/learn/feed", params={"action": "list"})
        assert resp.status_code == 200
        data = _d(resp)
        assert "feeds" in data

    def test_subscribe_feed(self, _mock_learner_singleton):
        resp = self.client.get(
            "/learn/feed", params={"action": "subscribe", "url": "https://example.com/rss"}
        )
        assert resp.status_code == 200
        data = _d(resp)
        assert data["status"] == "ok"
        _mock_learner_singleton.subscribe_feed.assert_called_once()

    def test_subscribe_missing_url(self):
        resp = self.client.get("/learn/feed", params={"action": "subscribe"})
        assert resp.status_code in (400, 422)

    def test_unsubscribe_feed(self, _mock_learner_singleton):
        resp = self.client.get(
            "/learn/feed", params={"action": "unsubscribe", "url": "https://example.com/rss"}
        )
        assert resp.status_code == 200
        data = _d(resp)
        assert data["status"] == "ok"

    def test_unsubscribe_missing_url(self):
        resp = self.client.get("/learn/feed", params={"action": "unsubscribe"})
        assert resp.status_code in (400, 422)

    def test_unknown_action(self):
        resp = self.client.get("/learn/feed", params={"action": "bad"})
        assert resp.status_code in (400, 422)


class TestLearnIngestUrl:
    def setup_method(self):
        self.client = get_test_client()

    def test_ingest_url(self, _mock_learner_singleton):
        resp = self.client.post("/learn/ingest-url", params={"url": "https://example.com"})
        assert resp.status_code == 200
        data = _d(resp)
        assert "elapsed_ms" in data
        _mock_learner_singleton.ingest_url.assert_called_once_with("https://example.com")

    def test_ingest_url_missing(self):
        resp = self.client.post("/learn/ingest-url")
        assert resp.status_code == 422


class TestLearnKnowledge:
    def setup_method(self):
        self.client = get_test_client()

    def test_query_by_topic(self, _mock_learner_singleton):
        resp = self.client.get("/learn/knowledge", params={"topic": "python"})
        assert resp.status_code == 200
        data = _d(resp)
        assert "facts" in data
        assert data["count"] >= 0

    def test_query_by_search(self, _mock_learner_singleton):
        resp = self.client.get("/learn/knowledge", params={"query": "python", "top_k": 5})
        assert resp.status_code == 200
        data = _d(resp)
        assert "facts" in data

    def test_query_no_params(self):
        resp = self.client.get("/learn/knowledge")
        assert resp.status_code == 200
        data = _d(resp)
        assert data["facts"] == []


class TestLearnIngest:
    def setup_method(self):
        self.client = get_test_client()

    def test_ingest_text(self, _mock_learner_singleton):
        resp = self.client.post("/learn/ingest", params={"text": "hello world"})
        assert resp.status_code == 200
        data = _d(resp)
        assert "elapsed_ms" in data
        _mock_learner_singleton.ingest_text.assert_called_once_with("hello world")

    def test_ingest_conversations(self, _mock_learner_singleton):
        resp = self.client.post(
            "/learn/ingest",
            json={"conversations": [["hi", "hello"], ["bye", "goodbye"]]},
        )
        if resp.status_code == 200:
            _mock_learner_singleton.ingest_conversation.assert_called_once()
        else:
            assert resp.status_code in (400, 422)

    def test_ingest_both(self, _mock_learner_singleton):
        resp = self.client.post(
            "/learn/ingest",
            json={"text": "context", "conversations": [["q", "a"]]},
        )
        if resp.status_code == 200:
            _mock_learner_singleton.ingest_text.assert_called_once()
            _mock_learner_singleton.ingest_conversation.assert_called_once()
        else:
            assert resp.status_code in (400, 422)

    def test_ingest_empty(self):
        resp = self.client.post("/learn/ingest")
        assert resp.status_code == 200


class TestLearnTrain:
    def setup_method(self):
        self.client = get_test_client()

    def test_train(self, _mock_learner_singleton):
        resp = self.client.post("/learn/train")
        assert resp.status_code == 200
        data = _d(resp)
        assert "elapsed_ms" in data
        _mock_learner_singleton.train_now.assert_called_once()


class TestLearnDeploy:
    def setup_method(self):
        self.client = get_test_client()

    def test_deploy_default(self, _mock_learner_singleton):
        resp = self.client.post("/learn/deploy")
        assert resp.status_code == 200
        data = _d(resp)
        assert "path" in data
        assert "soul_name" in data
        assert "elapsed_ms" in data

    def test_deploy_with_name(self, _mock_learner_singleton):
        resp = self.client.post("/learn/deploy", params={"name": "my-model"})
        assert resp.status_code == 200
        _mock_learner_singleton.deploy.assert_called_once_with(name="my-model")


class TestLearnEvaluate:
    def setup_method(self):
        self.client = get_test_client()

    def test_evaluate_default(self, _mock_learner_singleton):
        resp = self.client.post("/learn/evaluate")
        assert resp.status_code == 200
        data = _d(resp)
        assert "loss" in data
        assert "perplexity" in data
        assert "elapsed_ms" in data

    def test_evaluate_with_text(self, _mock_learner_singleton):
        resp = self.client.post("/learn/evaluate", params={"text": "test text"})
        assert resp.status_code == 200
        _mock_learner_singleton.evaluate.assert_called_once_with(text="test text")


class TestLearnerLifecycle:
    """End-to-end: status → search → ingest → train → evaluate."""

    def setup_method(self):
        self.client = get_test_client()

    def test_full_lifecycle(self, _mock_learner_singleton):
        # Status
        resp = self.client.get("/learn/status")
        assert resp.status_code == 200

        # Search
        resp = self.client.post("/learn/search", json={"query": "test"})
        assert resp.status_code == 200

        # Ingest
        resp = self.client.post("/learn/ingest", params={"text": "data"})
        assert resp.status_code == 200

        # Train
        resp = self.client.post("/learn/train")
        assert resp.status_code == 200

        # Evaluate
        resp = self.client.post("/learn/evaluate")
        assert resp.status_code == 200
