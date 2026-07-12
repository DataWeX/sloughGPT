"""
Tests for the workflow router — status, start, stop, triggers.

Uses a standalone FastAPI app with only the router under test.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.workflow import router

app = FastAPI()
app.include_router(router)
client = TestClient(app, raise_server_exceptions=False)

WF_TARGET = "apps.api.server.routers.workflow._get_workflow"


def _make_workflow():
    """Create a mock FeedbackWorkflowManager."""
    wf = MagicMock()
    wf.get_status.return_value = {
        "running": True,
        "aggregate_count": 5,
        "prune_count": 2,
        "export_count": 1,
        "last_aggregate": "2026-01-01T00:00:00",
        "last_prune": None,
        "last_export": None,
    }
    wf.trigger_aggregate.return_value = {"status": "aggregated", "count": 3}
    wf.trigger_prune.return_value = {"status": "pruned", "deleted": 1}
    wf.trigger_export.return_value = {"status": "exported", "path": "/tmp/export.json"}
    return wf


class TestWorkflowStatus:
    """GET /workflow/status"""

    @patch(WF_TARGET)
    def test_get_status(self, mock_get_wf):
        mock_get_wf.return_value = _make_workflow()

        resp = client.get("/workflow/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["running"] is True
        assert data["aggregate_count"] == 5
        assert data["prune_count"] == 2

    @patch(WF_TARGET)
    def test_get_status_error(self, mock_get_wf):
        mock_get_wf.side_effect = Exception("broken")
        resp = client.get("/workflow/status")
        assert resp.status_code == 500

    @patch("domains.feedback.get_feedback_workflow", side_effect=ImportError("no module"))
    def test_get_status_import_error(self, _):
        resp = client.get("/workflow/status")
        assert resp.status_code == 503


class TestWorkflowStart:
    """POST /workflow/start"""

    @patch(WF_TARGET)
    def test_start_workflow(self, mock_get_wf):
        wf = _make_workflow()
        mock_get_wf.return_value = wf

        resp = client.post(
            "/workflow/start",
            json={
                "aggregate_interval_minutes": 30,
                "prune_interval_minutes": 60,
                "export_interval_hours": 12,
                "health_check_interval_seconds": 15,
            },
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "started"
        assert data["config"]["aggregate_interval_minutes"] == 30
        assert data["config"]["prune_interval_minutes"] == 60
        assert data["config"]["export_interval_hours"] == 12
        assert data["config"]["health_check_interval_seconds"] == 15
        wf.start.assert_called_once()

    @patch(WF_TARGET)
    def test_start_workflow_defaults(self, mock_get_wf):
        wf = _make_workflow()
        mock_get_wf.return_value = wf

        resp = client.post("/workflow/start", json={})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["config"]["aggregate_interval_minutes"] == 60
        assert data["config"]["prune_interval_minutes"] == 120
        assert data["config"]["export_interval_hours"] == 24
        assert data["config"]["health_check_interval_seconds"] == 30

    @patch(WF_TARGET)
    def test_start_sets_config(self, mock_get_wf):
        wf = _make_workflow()
        mock_get_wf.return_value = wf

        client.post("/workflow/start", json={"aggregate_interval_minutes": 45})
        assert wf.config is not None

    @patch(WF_TARGET)
    def test_start_error(self, mock_get_wf):
        mock_get_wf.side_effect = Exception("broken")
        resp = client.post("/workflow/start", json={})
        assert resp.status_code == 500

    @patch("domains.feedback.get_feedback_workflow", side_effect=ImportError("no module"))
    def test_start_import_error(self, _):
        resp = client.post("/workflow/start", json={})
        assert resp.status_code == 503


class TestWorkflowStop:
    """POST /workflow/stop"""

    @patch(WF_TARGET)
    def test_stop_workflow(self, mock_get_wf):
        wf = _make_workflow()
        mock_get_wf.return_value = wf

        resp = client.post("/workflow/stop")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "stopped"
        wf.stop.assert_called_once()

    @patch(WF_TARGET)
    def test_stop_error(self, mock_get_wf):
        mock_get_wf.side_effect = Exception("broken")
        resp = client.post("/workflow/stop")
        assert resp.status_code == 500


class TestWorkflowTrigger:
    """POST /workflow/trigger/{action}"""

    @patch(WF_TARGET)
    def test_trigger_aggregate(self, mock_get_wf):
        wf = _make_workflow()
        mock_get_wf.return_value = wf

        resp = client.post("/workflow/trigger/aggregate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "aggregated"
        assert data["count"] == 3
        wf.trigger_aggregate.assert_called_once()

    @patch(WF_TARGET)
    def test_trigger_prune(self, mock_get_wf):
        wf = _make_workflow()
        mock_get_wf.return_value = wf

        resp = client.post("/workflow/trigger/prune")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pruned"
        wf.trigger_prune.assert_called_once()

    @patch(WF_TARGET)
    def test_trigger_export(self, mock_get_wf):
        wf = _make_workflow()
        mock_get_wf.return_value = wf

        resp = client.post("/workflow/trigger/export")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "exported"
        assert data["path"] == "/tmp/export.json"
        wf.trigger_export.assert_called_once()

    @patch(WF_TARGET)
    def test_trigger_invalid_action(self, mock_get_wf):
        mock_get_wf.return_value = _make_workflow()

        resp = client.post("/workflow/trigger/invalid")
        assert resp.status_code == 400
        assert "Unknown action" in resp.json()["detail"]

    @patch(WF_TARGET)
    def test_trigger_foobar(self, mock_get_wf):
        mock_get_wf.return_value = _make_workflow()

        resp = client.post("/workflow/trigger/foobar")
        assert resp.status_code == 400
        assert "foobar" in resp.json()["detail"]

    @patch(WF_TARGET)
    def test_trigger_aggregate_error(self, mock_get_wf):
        wf = _make_workflow()
        wf.trigger_aggregate.side_effect = Exception("aggregate failed")
        mock_get_wf.return_value = wf

        resp = client.post("/workflow/trigger/aggregate")
        assert resp.status_code == 500

    @patch(WF_TARGET)
    def test_trigger_prune_error(self, mock_get_wf):
        wf = _make_workflow()
        wf.trigger_prune.side_effect = Exception("prune failed")
        mock_get_wf.return_value = wf

        resp = client.post("/workflow/trigger/prune")
        assert resp.status_code == 500

    @patch("domains.feedback.get_feedback_workflow", side_effect=ImportError("no module"))
    def test_trigger_import_error(self, _):
        resp = client.post("/workflow/trigger/aggregate")
        assert resp.status_code == 503
