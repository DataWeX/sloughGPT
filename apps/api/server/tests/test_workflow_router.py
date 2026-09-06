"""Tests for the /workflow router (feedback workflow management)."""

from unittest.mock import MagicMock, patch

from test_support import get_test_client


def _data(resp):
    body = resp.json()
    return body.get("data", body)


def _make_workflow():
    wf = MagicMock()
    wf.get_status.return_value = {
        "running": False,
        "stats": {
            "workflow_runs": 0,
            "aggregations_performed": 0,
            "prunes_performed": 0,
            "exports_performed": 0,
        },
        "pending_thumbs_up": 0,
        "auto_train_threshold": 3,
        "config": {
            "aggregate_interval_minutes": 60,
            "prune_interval_minutes": 120,
            "export_interval_hours": 24,
        },
        "last_runs": {},
        "systems": {},
    }
    wf.trigger_aggregate.return_value = {"status": "ok", "aggregated": 0}
    wf.trigger_prune.return_value = {"status": "ok", "pruned": 0}
    wf.trigger_export.return_value = {"status": "ok", "exported": 0}
    return wf


class TestWorkflowStatus:
    def test_status(self):
        with patch("domains.feedback.get_feedback_workflow", return_value=_make_workflow()):
            client = get_test_client()
            resp = client.get("/workflow/status")
            assert resp.status_code == 200
            data = _data(resp)
            assert "running" in data

    def test_status_has_stats(self):
        with patch("domains.feedback.get_feedback_workflow", return_value=_make_workflow()):
            client = get_test_client()
            resp = client.get("/workflow/status")
            data = _data(resp)
            assert "stats" in data
            assert "config" in data
            assert "pending_thumbs_up" in data


class TestWorkflowStartStop:
    def test_start_workflow(self):
        wf = _make_workflow()
        with patch("domains.feedback.get_feedback_workflow", return_value=wf):
            client = get_test_client()
            resp = client.post("/workflow/start", json={})
            assert resp.status_code == 200
            data = _data(resp)
            assert data["status"] == "started"
            wf.start.assert_called_once()

    def test_start_workflow_custom_config(self):
        wf = _make_workflow()
        with patch("domains.feedback.get_feedback_workflow", return_value=wf):
            client = get_test_client()
            resp = client.post(
                "/workflow/start",
                json={
                    "aggregate_interval_minutes": 30,
                    "prune_interval_minutes": 60,
                    "export_interval_hours": 12,
                },
            )
            assert resp.status_code == 200

    def test_stop_workflow(self):
        wf = _make_workflow()
        with patch("domains.feedback.get_feedback_workflow", return_value=wf):
            client = get_test_client()
            resp = client.post("/workflow/stop")
            assert resp.status_code == 200
            data = _data(resp)
            assert data["status"] == "stopped"
            wf.stop.assert_called_once()


class TestWorkflowTrigger:
    def test_trigger_aggregate(self):
        wf = _make_workflow()
        with patch("domains.feedback.get_feedback_workflow", return_value=wf):
            client = get_test_client()
            resp = client.post("/workflow/trigger/aggregate")
            assert resp.status_code == 200
            wf.trigger_aggregate.assert_called_once()

    def test_trigger_prune(self):
        wf = _make_workflow()
        with patch("domains.feedback.get_feedback_workflow", return_value=wf):
            client = get_test_client()
            resp = client.post("/workflow/trigger/prune")
            assert resp.status_code == 200
            wf.trigger_prune.assert_called_once()

    def test_trigger_export(self):
        wf = _make_workflow()
        with patch("domains.feedback.get_feedback_workflow", return_value=wf):
            client = get_test_client()
            resp = client.post("/workflow/trigger/export")
            assert resp.status_code == 200
            wf.trigger_export.assert_called_once()

    def test_trigger_unknown_action(self):
        with patch("domains.feedback.get_feedback_workflow", return_value=_make_workflow()):
            client = get_test_client()
            resp = client.post("/workflow/trigger/bogus")
            assert resp.status_code == 400
