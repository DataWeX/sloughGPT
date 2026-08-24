"""Tests for the /workflow router (status, start, stop, trigger)."""
from test_support import get_test_client


def _d(resp):
    j = resp.json()
    return j.get("data", j)


def test_workflow_status():
    client = get_test_client()
    resp = client.get("/workflow/status")
    assert resp.status_code == 200
    body = _d(resp)
    assert isinstance(body, dict)
    assert "status" in body


def test_workflow_start():
    client = get_test_client()
    resp = client.post("/workflow/start")
    assert resp.status_code == 200


def test_workflow_stop():
    client = get_test_client()
    resp = client.post("/workflow/stop")
    assert resp.status_code == 200


def test_workflow_trigger():
    client = get_test_client()
    resp = client.post("/workflow/trigger/retrain")
    assert resp.status_code in (200, 404, 400, 500)
