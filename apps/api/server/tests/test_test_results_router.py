"""Tests for the descriptor-projected test-results crossing.

The route lives in ``routers/test_results.py`` (descriptor + projection, see
docs/TRANSPORT_PROJECTIONS.md) — its tests live here so the module name, the
test file name and the docs parity check agree. Card 20261111_110106: the
``tests/core-py/test_results/*.json`` artifacts were written on every run with
zero consumers; this crossing makes them readable by the monitoring surface.
"""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client(results_dir, monkeypatch):
    """App serving only the test-results router with a stubbed store root."""
    import routers.test_results as test_results_mod
    from infrastructure.auth import require_auth_if_enabled
    from infrastructure.exception_handlers import register_app_error_handler
    from routers.test_results import router as test_results_router

    monkeypatch.setattr(test_results_mod, "RESULTS_DIR", results_dir)

    app = FastAPI()
    register_app_error_handler(app)
    app.include_router(test_results_router)
    app.dependency_overrides[require_auth_if_enabled] = lambda: None
    return TestClient(app)


def _write_store(root):
    """Fabricate a two-file store: one green run, one run with a failure."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "green_journey_results.json").write_text(
        json.dumps(
            [
                {"test": "dashboard_loads", "passed": True, "detail": "ok"},
                {"test": "chat_streams", "passed": True, "detail": "ok"},
            ]
        )
    )
    (root / "red_journey_results.json").write_text(
        json.dumps(
            [
                {"test": "training_config", "passed": True, "detail": "ok"},
                {"test": "checkpoint_input", "passed": False, "detail": "missing element"},
            ]
        )
    )
    # Non-JSON noise in the directory must not break the crossing.
    (root / "notes.txt").write_text("not a results file")
    return root


def test_test_results_spec_is_descriptor_projected():
    """The capability is declared once as a ToolSpec and projects to GET."""
    from routers.test_results import SPEC

    assert SPEC.name == "test_results.list"
    assert SPEC.idempotent is True
    assert SPEC.version == "1"


def test_test_results_list_summarizes_each_file(tmp_path, monkeypatch):
    """Each results file becomes a summary row; failures are counted."""
    _write_store(tmp_path)
    res = _client(tmp_path, monkeypatch).get("/test-results")

    assert res.status_code == 200
    files = res.json()["data"]["files"]
    by_name = {f["file"]: f for f in files}
    assert set(by_name) == {"green_journey_results.json", "red_journey_results.json"}

    green = by_name["green_journey_results.json"]
    assert green["total"] == 2
    assert green["passed"] == 2
    assert green["failed"] == 0

    red = by_name["red_journey_results.json"]
    assert red["total"] == 2
    assert red["passed"] == 1
    assert red["failed"] == 1
    # mtime is present so the UI can show freshness without a second call.
    assert isinstance(red["mtime"], (int, float))


def test_test_results_empty_store_serves_empty_list(tmp_path, monkeypatch):
    """An empty (or absent) store is a valid state, not an error."""
    res = _client(tmp_path / "missing", monkeypatch).get("/test-results")
    assert res.status_code == 200
    assert res.json()["data"]["files"] == []


def test_test_results_raw_entries_collapse_behind_summary(tmp_path, monkeypatch):
    """Summaries carry counts; raw entries ride along for the one-click view."""
    _write_store(tmp_path)
    res = _client(tmp_path, monkeypatch).get("/test-results")
    files = res.json()["data"]["files"]
    red = next(f for f in files if f["file"] == "red_journey_results.json")
    assert red["entries"][1] == {
        "test": "checkpoint_input",
        "passed": False,
        "detail": "missing element",
    }
