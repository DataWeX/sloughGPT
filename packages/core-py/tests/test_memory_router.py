"""Tests for the memory API router (routers/memory.py).

Covers: stats, list, search, store, remember, config (GET/POST),
clear, consolidate, archive, archive_stats, archive_prune, delete, update.
MemoryService is mocked throughout.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.memory import (
    MemoryRouter, StoreRequest, RememberRequest, ConfigRequest, UpdateRequest,
)


def _mock_svc():
    svc = MagicMock()
    svc.enabled = True
    svc.stats.return_value = {"fact_count": 5, "topics": ["a", "b"]}
    svc.list_all.return_value = [{"id": "1", "content": "fact"}]
    svc.retrieve.return_value = [{"id": "1", "content": "match"}]
    svc.store.return_value = True
    svc.remember.return_value = True
    svc.delete.return_value = 2
    svc.clear.return_value = 5
    svc.config_snapshot.return_value = {"enabled": True, "consolidation_threshold": 0.8}
    svc.update.return_value = True
    return svc


def _app():
    app = FastAPI()
    mr = MemoryRouter()
    app.include_router(mr.router)
    from infrastructure.exception_handlers import register_all_handlers
    register_all_handlers(app)
    return app


class TestStoreRequest:
    def test_valid(self):
        req = StoreRequest(content="hello", topic="t", source="s")
        assert req.content == "hello"
        assert req.topic == "t"
        assert req.source == "s"

    def test_defaults(self):
        req = StoreRequest(content="x")
        assert req.topic == "manual"
        assert req.source == "api"


class TestRememberRequest:
    def test_valid(self):
        req = RememberRequest(user_message="hi", assistant_response="hello")
        assert req.user_message == "hi"


class TestConfigRequest:
    def test_none_defaults(self):
        req = ConfigRequest()
        assert req.enabled is None
        assert req.archive_retention_days is None


class TestUpdateRequest:
    def test_valid(self):
        req = UpdateRequest(content="new text")
        assert req.content == "new text"
        assert req.topic is None
        assert req.importance is None


class TestStats:
    @patch("routers.memory.get_memory_service")
    def test_stats(self, mock_get):
        mock_get.return_value = _mock_svc()
        client = TestClient(_app())
        resp = client.get("/memory/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["fact_count"] == 5
        assert body["enabled"] is True


class TestListMemory:
    @patch("routers.memory.get_memory_service")
    def test_list(self, mock_get):
        mock_get.return_value = _mock_svc()
        client = TestClient(_app())
        resp = client.get("/memory/list?limit=10")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["total"] == 1


class TestSearch:
    @patch("routers.memory.get_memory_service")
    def test_search(self, mock_get):
        mock_get.return_value = _mock_svc()
        client = TestClient(_app())
        resp = client.get("/memory/search?q=hello")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["total"] == 1


class TestStore:
    @patch("routers.memory.get_memory_service")
    def test_store(self, mock_get):
        mock_get.return_value = _mock_svc()
        client = TestClient(_app())
        resp = client.post("/memory/store", json={"content": "fact"})
        assert resp.status_code == 200
        assert resp.json()["data"]["stored"] is True


class TestRemember:
    @patch("routers.memory.get_memory_service")
    def test_remember(self, mock_get):
        mock_get.return_value = _mock_svc()
        client = TestClient(_app())
        resp = client.post("/memory/remember", json={
            "user_message": "hi", "assistant_response": "hello"
        })
        assert resp.status_code == 200
        assert resp.json()["data"]["stored"] is True


class TestConfig:
    @patch("routers.memory.get_memory_service")
    def test_get_config(self, mock_get):
        mock_get.return_value = _mock_svc()
        client = TestClient(_app())
        resp = client.get("/memory/config")
        assert resp.status_code == 200
        assert resp.json()["data"]["enabled"] is True

    @patch("routers.memory.get_memory_service")
    def test_set_config(self, mock_get):
        svc = _mock_svc()
        mock_get.return_value = svc
        client = TestClient(_app())
        resp = client.post("/memory/config", json={"enabled": False})
        assert resp.status_code == 200
        svc.set_enabled.assert_called_once_with(False)


class TestClear:
    @patch("routers.memory.get_memory_service")
    def test_clear(self, mock_get):
        mock_get.return_value = _mock_svc()
        client = TestClient(_app())
        resp = client.post("/memory/clear")
        assert resp.status_code == 200
        assert resp.json()["data"]["cleared"] == 5


class TestDeleteItem:
    @patch("routers.memory.get_memory_service")
    def test_delete(self, mock_get):
        mock_get.return_value = _mock_svc()
        client = TestClient(_app())
        resp = client.delete("/memory/item-1")
        assert resp.status_code == 200
        assert resp.json()["data"]["deleted"] == 2

    @patch("routers.memory.get_memory_service")
    def test_delete_empty_id(self, mock_get):
        mock_get.return_value = _mock_svc()
        client = TestClient(_app())
        resp = client.delete("/memory/%20")
        assert resp.status_code == 400


class TestUpdateItem:
    @patch("routers.memory.get_memory_service")
    def test_update(self, mock_get):
        mock_get.return_value = _mock_svc()
        client = TestClient(_app())
        resp = client.patch("/memory/item-1", json={"content": "new"})
        assert resp.status_code == 200
        assert resp.json()["data"]["updated"] == 1

    @patch("routers.memory.get_memory_service")
    def test_update_empty_id(self, mock_get):
        mock_get.return_value = _mock_svc()
        client = TestClient(_app())
        resp = client.patch("/memory/%20", json={"content": "new"})
        assert resp.status_code == 400


class TestConsolidate:
    @patch("domains.memory.consolidation.plan_consolidation")
    @patch("routers.memory.get_memory_service")
    def test_consolidate(self, mock_get, mock_plan):
        svc = _mock_svc()
        mock_get.return_value = svc
        mock_plan.return_value = {"remove_ids": ["1"], "keep_ids": ["2"]}
        client = TestClient(_app())
        resp = client.post("/memory/consolidate")
        assert resp.status_code == 200
        assert resp.json()["data"]["removed"] == 2


class TestArchive:
    @patch("domains.memory.task_memory.list_archive")
    def test_archive(self, mock_list):
        mock_list.return_value = [{"id": "a"}]
        client = TestClient(_app())
        resp = client.get("/memory/archive")
        assert resp.status_code == 200
        assert resp.json()["data"]["total"] == 1

    @patch("domains.memory.task_memory.archive_stats")
    def test_archive_stats(self, mock_stats):
        mock_stats.return_value = {"total": 10}
        client = TestClient(_app())
        resp = client.get("/memory/archive/stats")
        assert resp.status_code == 200
        assert resp.json()["data"]["total"] == 10

    @patch("domains.memory.task_memory.prune_archive")
    def test_archive_prune(self, mock_prune):
        mock_prune.return_value = 3
        client = TestClient(_app())
        resp = client.post("/memory/archive/prune")
        assert resp.status_code == 200
        assert resp.json()["data"]["pruned"] == 3


class TestErrorHandling:
    @patch("routers.memory.get_memory_service")
    def test_stats_exception(self, mock_get):
        mock_get.side_effect = RuntimeError("db down")
        client = TestClient(_app())
        resp = client.get("/memory/stats")
        assert resp.status_code >= 400
