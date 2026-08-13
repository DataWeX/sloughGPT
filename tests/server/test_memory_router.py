"""Tests for the memory API router (apps/api/server/routers/memory.py)."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.memory import router

app = FastAPI()
app.include_router(router)
client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def fake_service():
    """Substitute a fake MemoryService for all router tests."""
    svc = MagicMock()
    svc.enabled = True
    svc.stats.return_value = {"total_facts": 3, "topics": 1, "visited_urls": 0}
    svc.list_all.return_value = [
        {"content": "The capital of France is Paris", "topic": "geo",
         "source": "task", "score": 0.5},
        {"content": "The sky appears blue by day", "topic": "science",
         "source": "task", "score": 0.4},
    ]
    svc.retrieve.return_value = [
        {"content": "The capital of France is Paris", "topic": "geo",
         "source": "task", "score": 0.9},
    ]
    svc.store.return_value = True
    svc.remember.return_value = True
    svc.clear.return_value = 2
    svc.delete.return_value = 1
    svc.update.return_value = True
    with patch("routers.memory.get_memory_service", return_value=svc) as m:
        yield svc


class TestStats:
    def test_stats_shape(self, fake_service):
        resp = client.get("/memory/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["total_facts"] == 3
        assert data["topics"] == 1

    def test_stats_delegates_to_service(self, fake_service):
        client.get("/memory/stats")
        fake_service.stats.assert_called_once()


class TestList:
    def test_list_items(self, fake_service):
        resp = client.get("/memory/list")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["items"][0]["content"] == "The capital of France is Paris"

    def test_list_passes_limit(self, fake_service):
        client.get("/memory/list?limit=7")
        fake_service.list_all.assert_called_with(limit=7)

    def test_list_clamps_limits(self, fake_service):
        client.get("/memory/list?limit=0")
        fake_service.list_all.assert_called_with(limit=1)
        client.get("/memory/list?limit=99999")
        fake_service.list_all.assert_called_with(limit=1000)


class TestSearch:
    def test_search_requires_query(self):
        resp = client.get("/memory/search")
        assert resp.status_code == 400

    def test_search_returns_results(self, fake_service):
        resp = client.get("/memory/search?q=france")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert "Paris" in data["results"][0]["content"]
        fake_service.retrieve.assert_called_with("france", limit=5)

    def test_search_passes_limit(self, fake_service):
        client.get("/memory/search?q=test&limit=3")
        fake_service.retrieve.assert_called_with("test", limit=3)


class TestStore:
    def test_store_requires_content(self):
        resp = client.post("/memory/store", json={"content": ""})
        assert resp.status_code == 422

    def test_store_persists(self, fake_service):
        resp = client.post("/memory/store", json={"content": "a fact", "topic": "t"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["stored"] is True
        assert data["content"] == "a fact"
        fake_service.store.assert_called_with("a fact", "t", "api")


class TestRemember:
    def test_remember_requires_both_fields(self):
        resp = client.post("/memory/remember", json={"user_message": "hi"})
        assert resp.status_code == 422

    def test_remember_stores_turn(self, fake_service):
        resp = client.post(
            "/memory/remember",
            json={"user_message": "question", "assistant_response": "answer"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["stored"] is True
        fake_service.remember.assert_called_with("question", "answer")


class TestClear:
    def test_clear(self, fake_service):
        resp = client.post("/memory/clear")
        assert resp.status_code == 200
        assert resp.json() == {"cleared": 2}
        fake_service.clear.assert_called_once()


class TestDelete:
    def test_delete_item(self, fake_service):
        resp = client.delete("/memory/abc123")
        assert resp.status_code == 200
        assert resp.json() == {"deleted": 1}
        fake_service.delete.assert_called_with(["abc123"])

    def test_delete_strips_whitespace(self, fake_service):
        client.delete("/memory/%20abc123%20")
        fake_service.delete.assert_called_with(["abc123"])

    def test_delete_missing_id(self, fake_service):
        resp = client.delete("/memory/")
        assert resp.status_code >= 400

    def test_delete_returns_zero_when_not_found(self, fake_service):
        fake_service.delete.return_value = 0
        resp = client.delete("/memory/unknown")
        assert resp.status_code == 200
        assert resp.json() == {"deleted": 0}


class TestConsolidate:
    def test_consolidate_merges_near_duplicates(self, fake_service):
        fake_service.list_all.return_value = [
            {"id": "a", "content": "Machine learning learns patterns from data."},
            {"id": "b", "content": "Machine learning learns patterns from data very effectively."},
        ]
        fake_service.delete.return_value = 1
        with patch("domains.memory.consolidation.plan_consolidation") as plan:
            plan.return_value = {"remove_ids": ["a"], "keep_ids": ["b"]}
            resp = client.post("/memory/consolidate")
        assert resp.status_code == 200
        data = resp.json()
        assert data == {"removed": 1, "kept": 1, "threshold": 0.8}
        fake_service.list_all.assert_called_with(limit=5000)
        fake_service.delete.assert_called_with(["a"])

    def test_consolidate_passes_threshold(self, fake_service):
        fake_service.list_all.return_value = []
        with patch("domains.memory.consolidation.plan_consolidation") as plan:
            plan.return_value = {"remove_ids": [], "keep_ids": []}
            client.post("/memory/consolidate?threshold=0.5")
        plan.assert_called_with([], threshold=0.5)

    def test_consolidate_empty_store(self, fake_service):
        fake_service.list_all.return_value = []
        with patch("domains.memory.consolidation.plan_consolidation") as plan:
            plan.return_value = {"remove_ids": [], "keep_ids": []}
            resp = client.post("/memory/consolidate")
        assert resp.status_code == 200
        assert resp.json()["removed"] == 0


class TestArchive:
    def test_archive_lists_records(self, fake_service):
        with patch("domains.memory.task_memory.list_archive") as la:
            la.return_value = [{"ts": 1, "task_type": "memory.store"}]
            resp = client.get("/memory/archive")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["records"][0]["task_type"] == "memory.store"
        la.assert_called_with(limit=20)

    def test_archive_passes_limit(self, fake_service):
        with patch("domains.memory.task_memory.list_archive") as la:
            la.return_value = []
            client.get("/memory/archive?limit=5")
        la.assert_called_with(limit=5)

    def test_archive_stats(self, fake_service):
        with patch("domains.memory.task_memory.archive_stats") as st:
            st.return_value = {"records": 3, "path": "/tmp/facts.jsonl"}
            resp = client.get("/memory/archive/stats")
        assert resp.status_code == 200
        assert resp.json()["records"] == 3

    def test_archive_prune_defaults_to_config(self, fake_service):
        with patch("domains.memory.task_memory.prune_archive") as pr:
            pr.return_value = 2
            resp = client.post("/memory/archive/prune")
        assert resp.status_code == 200
        assert resp.json() == {"pruned": 2}
        pr.assert_called_with(retain_days=None)

    def test_archive_prune_passes_retention(self, fake_service):
        with patch("domains.memory.task_memory.prune_archive") as pr:
            pr.return_value = 0
            client.post("/memory/archive/prune?retain_days=7")
        pr.assert_called_with(retain_days=7.0)


class TestUpdate:
    def test_patch_updates_content_and_topic(self, fake_service):
        resp = client.patch("/memory/fact_1_abc", json={"content": "New fact text", "topic": "drinks"})
        assert resp.status_code == 200
        assert resp.json() == {"updated": 1, "duplicate": False}
        fake_service.update.assert_called_once_with("fact_1_abc", "New fact text", topic="drinks", importance=None)

    def test_patch_omits_topic(self, fake_service):
        client.patch("/memory/fact_1_abc", json={"content": "New fact text"})
        fake_service.update.assert_called_once_with("fact_1_abc", "New fact text", topic=None, importance=None)

    def test_patch_passes_importance(self, fake_service):
        resp = client.patch("/memory/fact_1_abc", json={"content": "New fact text", "importance": 0.9})
        assert resp.status_code == 200
        fake_service.update.assert_called_once_with("fact_1_abc", "New fact text", topic=None, importance=0.9)

    def test_patch_rejects_importance_above_range(self, fake_service):
        resp = client.patch("/memory/fact_1_abc", json={"content": "New fact text", "importance": 1.5})
        assert resp.status_code == 422

    def test_patch_rejects_importance_below_range(self, fake_service):
        resp = client.patch("/memory/fact_1_abc", json={"content": "New fact text", "importance": -0.1})
        assert resp.status_code == 422

    def test_patch_rejects_empty_content(self, fake_service):
        resp = client.patch("/memory/fact_1_abc", json={"content": "   "})
        assert resp.status_code == 400

    def test_patch_rejects_missing_content(self, fake_service):
        resp = client.patch("/memory/fact_1_abc", json={"topic": "drinks"})
        assert resp.status_code == 422

    def test_patch_rejects_empty_item_id(self, fake_service):
        resp = client.patch("/memory/   ", json={"content": "New fact text"})
        assert resp.status_code == 400

    def test_patch_reports_duplicate(self, fake_service):
        fake_service.update.return_value = False
        resp = client.patch("/memory/fact_1_abc", json={"content": "Duplicate text"})
        assert resp.status_code == 200
        assert resp.json() == {"updated": 0, "duplicate": True}
