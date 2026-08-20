"""Tests for the collections API router (routers/collections.py).

Covers: list_pipelines, create_pipeline, run_pipeline, get_stats,
get_pipeline, delete_pipeline, collect, get_records.
Registry and pipeline are mocked.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.collections import (
    CollectionsRouter, PipelineConfigRequest, LogMetricRequest,
    _build_source, _build_store, _build_filter,
)


def _mock_pipeline(name="test_pipe", stats=None):
    p = MagicMock()
    p.name = name
    p.stats = stats or {"total": 0}
    p.collect.return_value = 0
    p.read.return_value = []
    return p


def _mock_registry(pipelines=None):
    reg = MagicMock()
    pipes = pipelines or []
    reg.list.return_value = pipes
    reg.get.side_effect = lambda name: next((p for p in pipes if p.name == name), None)
    reg.register.return_value = None
    reg.unregister.return_value = None
    return reg


def _app() -> FastAPI:
    cr = CollectionsRouter()
    app = FastAPI()
    app.include_router(cr.router)
    from infrastructure.exception_handlers import register_all_handlers
    register_all_handlers(app)
    return app


class TestPipelineConfigRequest:
    def test_valid(self):
        req = PipelineConfigRequest(name="p1", source_type="file")
        assert req.name == "p1"
        assert req.source_type == "file"
        assert req.store_type == "memory"
        assert req.filter_chain == []

    def test_defaults(self):
        req = PipelineConfigRequest(name="p", source_type="url")
        assert req.source_config == {}
        assert req.store_config == {}


class TestLogMetricRequest:
    def test_valid(self):
        req = LogMetricRequest(metric_name="loss", value=0.5)
        assert req.metric_name == "loss"
        assert req.value == 0.5


class TestListPipelines:
    @patch("routers.collections.get_registry")
    def test_list_empty(self, mock_gr):
        mock_gr.return_value = _mock_registry()
        app = _app()
        client = TestClient(app)
        resp = client.get("/collections")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["data"]["pipelines"] == []
        assert body["data"]["count"] == 0

    @patch("routers.collections.get_registry")
    def test_list_with_pipelines(self, mock_gr):
        pipes = [_mock_pipeline("a"), _mock_pipeline("b")]
        mock_gr.return_value = _mock_registry(pipes)
        app = _app()
        client = TestClient(app)
        resp = client.get("/collections")
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 2

    @patch("routers.collections.get_registry")
    def test_list_exception_returns_empty(self, mock_gr):
        mock_gr.side_effect = RuntimeError("db down")
        app = _app()
        client = TestClient(app)
        resp = client.get("/collections")
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 0


class TestCreatePipeline:
    @patch("routers.collections.get_registry")
    def test_create_with_memory_store(self, mock_gr):
        mock_gr.return_value = _mock_registry()
        app = _app()
        client = TestClient(app)
        resp = client.post("/collections", json={
            "name": "pipe1",
            "source_type": "generator",
            "source_config": {},
            "store_type": "memory",
            "store_config": {},
            "filter_chain": [],
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["name"] == "pipe1"
        assert data["source_type"] == "generator"
        assert data["store_type"] == "memory"

    @patch("routers.collections.get_registry")
    def test_create_with_filters(self, mock_gr):
        mock_gr.return_value = _mock_registry()
        app = _app()
        client = TestClient(app)
        resp = client.post("/collections", json={
            "name": "pipe2",
            "source_type": "generator",
            "source_config": {},
            "filter_chain": [{"type": "length", "min_length": 5}],
        })
        assert resp.status_code == 200
        assert resp.json()["data"]["filters"] == 1


class TestRunPipeline:
    @patch("routers.collections.get_registry")
    def test_run_existing(self, mock_gr):
        pipe = _mock_pipeline("run1", stats={"total": 10})
        pipe.collect.return_value = 5
        mock_gr.return_value = _mock_registry([pipe])
        app = _app()
        client = TestClient(app)
        resp = client.post("/collections/run?name=run1")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["collected"] == 5
        assert data["pipeline"] == "run1"

    @patch("routers.collections.get_registry")
    def test_run_missing(self, mock_gr):
        mock_gr.return_value = _mock_registry()
        app = _app()
        client = TestClient(app)
        resp = client.post("/collections/run?name=nope")
        assert resp.status_code in (404, 422)


class TestGetStats:
    @patch("routers.collections.get_registry")
    def test_stats_empty(self, mock_gr):
        mock_gr.return_value = _mock_registry()
        app = _app()
        client = TestClient(app)
        resp = client.get("/collections/stats")
        assert resp.status_code == 200
        assert resp.json()["data"]["total_pipelines"] == 0

    @patch("routers.collections.get_registry")
    def test_stats_with_pipelines(self, mock_gr):
        mock_gr.return_value = _mock_registry([_mock_pipeline("x"), _mock_pipeline("y")])
        app = _app()
        client = TestClient(app)
        resp = client.get("/collections/stats")
        assert resp.status_code == 200
        assert resp.json()["data"]["total_pipelines"] == 2


class TestGetPipeline:
    @patch("routers.collections.get_registry")
    def test_get_existing(self, mock_gr):
        pipe = _mock_pipeline("g1", stats={"total": 42})
        mock_gr.return_value = _mock_registry([pipe])
        app = _app()
        client = TestClient(app)
        resp = client.get("/collections/g1")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["id"] == "g1"
        assert data["stats"]["total"] == 42

    @patch("routers.collections.get_registry")
    def test_get_missing(self, mock_gr):
        mock_gr.return_value = _mock_registry()
        app = _app()
        client = TestClient(app)
        resp = client.get("/collections/nope")
        assert resp.status_code in (404, 422)


class TestDeletePipeline:
    @patch("routers.collections.get_registry")
    def test_delete(self, mock_gr):
        mock_gr.return_value = _mock_registry()
        app = _app()
        client = TestClient(app)
        resp = client.delete("/collections/d1")
        assert resp.status_code == 200
        assert resp.json()["data"]["deleted"] == "d1"


class TestCollect:
    @patch("routers.collections.get_registry")
    def test_collect_existing(self, mock_gr):
        pipe = _mock_pipeline("c1")
        pipe.collect.return_value = 3
        pipe.stats = {"total": 3}
        mock_gr.return_value = _mock_registry([pipe])
        app = _app()
        client = TestClient(app)
        resp = client.post("/collections/c1/collect")
        assert resp.status_code == 200
        assert resp.json()["data"]["collected"] == 3

    @patch("routers.collections.get_registry")
    def test_collect_missing(self, mock_gr):
        mock_gr.return_value = _mock_registry()
        app = _app()
        client = TestClient(app)
        resp = client.post("/collections/none/collect")
        assert resp.status_code in (404, 422)


class TestGetRecords:
    @patch("routers.collections.get_registry")
    def test_records_empty(self, mock_gr):
        pipe = _mock_pipeline("r1")
        pipe.read.return_value = []
        mock_gr.return_value = _mock_registry([pipe])
        app = _app()
        client = TestClient(app)
        resp = client.get("/collections/r1/records")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["records"] == []
        assert data["total"] == 0

    @patch("routers.collections.get_registry")
    def test_records_with_data(self, mock_gr):
        rec1 = MagicMock(content="hello", source="file1")
        rec2 = MagicMock(content="world", source="file2")
        pipe = _mock_pipeline("r2")
        pipe.read.return_value = [rec1, rec2]
        mock_gr.return_value = _mock_registry([pipe])
        app = _app()
        client = TestClient(app)
        resp = client.get("/collections/r2/records?limit=1")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data["records"]) == 1
        assert data["total"] == 2
        assert data["returned"] == 1

    @patch("routers.collections.get_registry")
    def test_records_missing_pipeline(self, mock_gr):
        mock_gr.return_value = _mock_registry()
        app = _app()
        client = TestClient(app)
        resp = client.get("/collections/x/records")
        assert resp.status_code in (404, 422)
