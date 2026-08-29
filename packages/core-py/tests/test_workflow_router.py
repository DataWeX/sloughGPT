"""Tests for workflow router — status, start, stop, trigger delegation."""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

pytest.importorskip("fastapi")

# Ensure apps/api/server is on the path for schemas.common import
_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.workflow import WorkflowRouter, WorkflowStartRequest


@pytest.fixture
def mock_workflow():
    """Create a mock FeedbackWorkflowManager."""
    wf = MagicMock()
    wf.get_status.return_value = {
        "running": True,
        "config": {"aggregate_interval_minutes": 60},
        "stats": {"feedback_count": 42},
    }
    wf.trigger_aggregate.return_value = {"status": "aggregated", "count": 5}
    wf.trigger_prune.return_value = {"status": "pruned", "removed": 2}
    wf.trigger_export.return_value = {"status": "exported", "path": "/tmp/export"}
    return wf


@pytest.fixture
def app(mock_workflow):
    """Create FastAPI app with mocked workflow."""
    router_instance = WorkflowRouter()
    app = FastAPI()
    app.include_router(router_instance.router)
    from infrastructure.exception_handlers import register_all_handlers
    register_all_handlers(app)
    with patch.object(router_instance, "_get_workflow", return_value=mock_workflow):
        yield app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestGetWorkflowStatus:
    def test_returns_status(self, client, mock_workflow):
        resp = client.get("/workflow/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["running"] is True
        assert data["stats"]["feedback_count"] == 42


class TestStartWorkflow:
    def test_start(self, client, mock_workflow):
        resp = client.post("/workflow/start", json={
            "aggregate_interval_minutes": 30,
            "prune_interval_minutes": 60,
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "started"
        mock_workflow.start.assert_called_once()

    def test_start_sets_config(self, client, mock_workflow):
        client.post("/workflow/start", json={
            "aggregate_interval_minutes": 15,
            "prune_interval_minutes": 30,
            "export_interval_hours": 12,
            "health_check_interval_seconds": 60,
        })
        assert mock_workflow.config is not None


class TestStopWorkflow:
    def test_stop(self, client, mock_workflow):
        resp = client.post("/workflow/stop")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "stopped"
        mock_workflow.stop.assert_called_once()


class TestTriggerWorkflow:
    def test_trigger_aggregate(self, client, mock_workflow):
        resp = client.post("/workflow/trigger/aggregate")
        assert resp.status_code == 200
        mock_workflow.trigger_aggregate.assert_called_once()

    def test_trigger_prune(self, client, mock_workflow):
        resp = client.post("/workflow/trigger/prune")
        assert resp.status_code == 200
        mock_workflow.trigger_prune.assert_called_once()

    def test_trigger_export(self, client, mock_workflow):
        resp = client.post("/workflow/trigger/export")
        assert resp.status_code == 200
        mock_workflow.trigger_export.assert_called_once()

    def test_trigger_unknown_action(self, client):
        resp = client.post("/workflow/trigger/unknown")
        assert resp.status_code == 400
        assert "Unknown action" in resp.json()["error"]


class TestWorkflowModuleLevel:
    def test_get_workflow_function(self):
        """Module-level _get_workflow delegates to the router instance."""
        from apps.api.server.routers.workflow import _get_workflow, _workflow_router
        mock_wf = MagicMock()
        with patch.object(_workflow_router, "_get_workflow", return_value=mock_wf):
            result = _get_workflow()
            assert result is mock_wf
