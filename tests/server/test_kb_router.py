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

    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_returns_list_of_items(self, mock_get_mem, client):
        mem = mock_get_mem.return_value
        mem.list_all.return_value = [
            {"id": "1", "content": "fact 1", "topic": "general", "source": "manual", "importance": 0.5, "score": 1.0},
            {"id": "2", "content": "fact 2", "topic": "code", "source": "manual", "importance": 0.7, "score": 1.0},
        ]
        resp = client.get("/knowledge")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_respects_pagination(self, mock_get_mem, client):
        mem = mock_get_mem.return_value
        mem.list_all.return_value = [
            {"id": str(i), "content": f"f{i}", "topic": "general", "source": "manual", "importance": 0.5, "score": 1.0}
            for i in range(10)
        ]
        resp = client.get("/knowledge", params={"limit": 3, "offset": 2})
        assert resp.status_code == 200
        ids = [it["id"] for it in resp.json()]
        assert ids == ["2", "3", "4"]

    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_default_field_fill(self, mock_get_mem, client):
        mem = mock_get_mem.return_value
        mem.list_all.return_value = [{"id": "x"}]
        resp = client.get("/knowledge")
        item = resp.json()[0]
        assert item["topic"] == "general"
        assert item["importance"] == 0.5
        assert item["score"] == 0.0

    def test_invalid_pagination_params_422(self, client):
        assert client.get("/knowledge", params={"limit": 0}).status_code == 422
        assert client.get("/knowledge", params={"offset": -1}).status_code == 422
        assert client.get("/knowledge", params={"limit": 99999}).status_code == 422


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

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.infrastructure.truth_labeler.get_truth_labeler")
    def test_adds_knowledge_with_topic(self, mock_get_label, mock_get_mem, client):
        labeler = mock_get_label.return_value
        labeler.label.return_value = MagicMock(label="factual")
        mem = mock_get_mem.return_value
        mem.add_fact.return_value = True
        resp = client.post("/knowledge", json={"content": "Python is a language", "topic": "code"})
        assert resp.status_code == 200


class TestSearchKnowledge:
    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_searches(self, mock_get_mem, client):
        mem = mock_get_mem.return_value
        mem.search.return_value = [{"id": "1", "content": "test", "score": 0.9}]
        resp = client.get("/knowledge/search?query=test")
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 1

    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_search_empty_results(self, mock_get_mem, client):
        mem = mock_get_mem.return_value
        mem.search.return_value = []
        resp = client.get("/knowledge/search?query=nonexistent")
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 0


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


class TestBatchDelete:
    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_batch_delete(self, mock_get_mem, client):
        mem = mock_get_mem.return_value
        mem.delete_by_id.return_value = True
        resp = client.post("/knowledge/batch-delete", json={"ids": ["id1", "id2"]})
        assert resp.status_code == 200

    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_batch_delete_empty(self, mock_get_mem, client):
        mem = mock_get_mem.return_value
        mem.delete_by_id.return_value = True
        resp = client.post("/knowledge/batch-delete", json={"ids": []})
        assert resp.status_code == 200


class TestCheckDuplicate:
    @patch("domains.learner.knowledge_ops.DuplicateDetector")
    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_check_duplicate(self, mock_get_mem, mock_detector_cls, client):
        mem = mock_get_mem.return_value
        mem._vector_store = MagicMock()
        mem._get_embedding = MagicMock()
        dup = mock_detector_cls.return_value
        dup.check.return_value = (False, None, 0.0)
        resp = client.post("/knowledge/check-duplicate", json={"content": "Earth is round"})
        assert resp.status_code == 200


class TestKnowledgeGaps:
    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_knowledge_gaps(self, mock_get_mem, client):
        mem = mock_get_mem.return_value
        mem.list_all.return_value = []
        resp = client.get("/knowledge/gaps")
        assert resp.status_code == 200


class TestUpdateKnowledge:
    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_updates_existing_item(self, mock_get_mem, client):
        mem = mock_get_mem.return_value
        mem.list_all.return_value = [{"id": "abc", "content": "old", "topic": "docs", "source": "manual", "importance": 0.5}]
        mem.delete_by_id.return_value = True
        mem.add_fact.return_value = True
        resp = client.patch("/knowledge/abc", json={"content": "new content", "topic": "code"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "updated"

    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_update_missing_item_returns_404(self, mock_get_mem, client):
        mem = mock_get_mem.return_value
        mem.list_all.return_value = [{"id": "other"}]
        resp = client.patch("/knowledge/ghost", json={"content": "x"})
        assert resp.status_code == 404

    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_update_partial_fields_keep_existing(self, mock_get_mem, client):
        mem = mock_get_mem.return_value
        mem.list_all.return_value = [{"id": "abc", "content": "keep this", "topic": "docs", "source": "manual", "url": "u", "timestamp": 5.0, "importance": 0.5}]
        mem.delete_by_id.return_value = True
        mem.add_fact.return_value = True
        resp = client.patch("/knowledge/abc", json={"importance": 0.9})
        assert resp.status_code == 200
        added = mem.add_fact.call_args.args[0]
        assert added.content == "keep this"
        assert added.topic == "docs"
        assert added.importance == 0.9
        assert added.url == "u"
        assert added.timestamp == 5.0


class TestAddKnowledgeEdge:
    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_duplicate_content_reports_duplicate(self, mock_get_mem, client):
        mem = mock_get_mem.return_value
        mem.add_fact.return_value = False
        resp = client.post("/knowledge", json={"content": "duplicate text"})
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "duplicate"

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.infrastructure.truth_labeler.get_truth_labeler")
    def test_auto_tag_always_along(self, mock_get_label, mock_get_mem, client):
        labeler = mock_get_label.return_value
        labeler.label.return_value = MagicMock(label="factual")
        mem = mock_get_mem.return_value
        mem.add_fact.return_value = True
        resp = client.post("/knowledge", json={"content": "import os function", "auto_tag": True})
        assert resp.status_code == 200
        assert resp.json()["data"]["topic"] != "general"

    def test_empty_content_is_422(self, client):
        resp = client.post("/knowledge", json={"content": ""})
        assert resp.status_code == 422


class TestBatchIngest:
    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_batch_ingest_stores(self, mock_get_mem, client):
        mem = mock_get_mem.return_value
        mem.add_fact.return_value = True
        resp = client.post("/knowledge/batch", json={"items": [{"content": "a"}, {"content": "b"}]})
        assert resp.status_code == 200
        assert resp.json()["data"]["stored"] == 2

    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_batch_ingest_skips_duplicates(self, mock_get_mem, client):
        mem = mock_get_mem.return_value
        mem.add_fact.side_effect = [True, False]
        resp = client.post("/knowledge/batch", json={"items": [{"content": "a"}, {"content": "dup"}]})
        assert resp.json()["data"]["stored"] == 1


class TestRelatedKnowledge:
    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_related_excludes_self(self, mock_get_mem, client):
        mem = mock_get_mem.return_value
        mem.list_all.return_value = [{"id": "abc", "content": "topic text"}]
        mem.search.return_value = [
            {"id": "abc", "content": "a", "topic": "general", "source": "manual", "importance": 0.5, "score": 0.9},
            {"id": "def", "content": "b", "topic": "general", "source": "manual", "importance": 0.5, "score": 0.8},
        ]
        resp = client.get("/knowledge/abc/related")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["count"] == 1
        assert data["items"][0]["id"] == "def"

    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_related_missing_item_404(self, mock_get_mem, client):
        mem = mock_get_mem.return_value
        mem.list_all.return_value = [{"id": "other"}]
        resp = client.get("/knowledge/ghost/related")
        assert resp.status_code == 404


class TestIngestUrl:
    def test_blocked_scheme_is_400(self, client):
        resp = client.post("/knowledge/ingest-url", json={"url": "file:///etc/passwd"})
        assert resp.status_code == 400

    def test_internal_host_is_400(self, client):
        resp = client.post("/knowledge/ingest-url", json={"url": "http://127.0.0.1/admin"})
        assert resp.status_code == 400

    def test_no_scheme_is_400(self, client):
        resp = client.post("/knowledge/ingest-url", json={"url": "example.com/foo"})
        assert resp.status_code == 400

    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_https_ingest_ok(self, mock_get_ingestor, client):
        ing = mock_get_ingestor.return_value
        ing.ingest_url.return_value = {"status": "ok", "new_facts": 3, "title": "T", "content_length": 100}
        resp = client.post("/knowledge/ingest-url", json={"url": "https://example.com/foo"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["new_facts"] == 3

    @patch("domains.infrastructure.errors.classify_exception")
    @patch("domains.infrastructure.errors.emit_error_event")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_ingestor_exception_maps_to_http(self, mock_get_ing, mock_emit, mock_classify, client):
        mock_get_ing.side_effect = RuntimeError("boom")
        err = MagicMock()
        err.http_status = 503
        err.user_message = "upstream failed"
        mock_classify.return_value = err
        resp = client.post("/knowledge/ingest-url", json={"url": "https://example.com/foo"})
        assert resp.status_code == 503
        mock_emit.assert_called_once()


class TestLabelMethods:
    def test_label_text_empty_query_422(self, client):
        resp = client.get("/knowledge/label")
        assert resp.status_code == 422

    @patch("domains.infrastructure.truth_labeler.get_truth_labeler")
    def test_label_text_error_returns_500(self, mock_get_label, client):
        mock_get_label.side_effect = RuntimeError("broken")
        resp = client.get("/knowledge/label?text=hello")
        assert resp.status_code == 500


class TestMethodChecks:
    def test_knowledge_put_not_allowed(self, client):
        assert client.put("/knowledge/abc").status_code == 405

    def test_knowledge_search_wrong_method_405(self, client):
        assert client.post("/knowledge/search").status_code == 405
