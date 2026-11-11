"""Contract test for ``test_results.list`` — the descriptor-projected read of
``tests/core-py/test_results/*.json`` (card 20261111_110106).

The journey suites have been writing their evidence JSON into
``tests/core-py/test_results/`` with zero consumers anywhere in the UI. This
crossing is one descriptor (``test_results.list``) projected by
``create_router`` — the test pins verb, auth, envelope, honest normalization
and the registry↔served invariant, mirroring ``test_contracts.py`` for the
``contracts.list`` pilot.

Normalization is the safety-relevant part: the journey files store ``passed``
as the *strings* ``'True'``/``'False'``, so ``bool('False')`` is ``True`` —
the classic string-to-boolean bug. The projection must deliver real booleans
(or ``None`` when unparseable), never a truthy string.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from infrastructure.auth import require_auth_if_enabled
from infrastructure.contract import REGISTRY, RouteSpec, create_router, derive_method
from infrastructure.exception_handlers import register_app_error_handler

import routers.test_results as test_results_mod
from routers.test_results import SPEC, router


def _app(auth_user: dict | None = None) -> FastAPI:
    app = FastAPI()
    register_app_error_handler(app)
    app.include_router(router)
    if auth_user is not None:
        app.dependency_overrides[require_auth_if_enabled] = lambda: auth_user
    return app


# ── descriptor → projection (verb + auth come from the SPEC, not re-decided) ─


def test_descriptor_names_the_contract_and_projects_to_get():
    assert SPEC.name == "test_results.list"
    assert SPEC.idempotent is True
    assert SPEC.auth_scope == "authenticated"
    assert derive_method(SPEC) == "GET"


def test_route_is_served_at_test_results():
    served = {
        (route.path, method)
        for route in router.routes
        for method in route.methods
        if method != "HEAD"
    }
    assert ("/test-results", "GET") in served


def test_registry_claim_is_actually_served():
    """The registry's claim must match the router's route table, not a paper one."""
    claimed = {
        (e["path"], e["method"])
        for e in REGISTRY.routes()
        if e["module"] == "routers.test_results"
    }
    assert ("/test-results", "GET") in claimed
    served = {
        (route.path, method)
        for route in router.routes
        for method in route.methods
        if method != "HEAD"
    }
    assert claimed <= served, f"registry claims routes no router serves: {claimed - served}"


def test_openapi_documents_the_contract():
    op = _app().openapi()["paths"]["/test-results"]["get"]
    assert op["operationId"] == "test_results.list.get"
    assert op["x-contract"]["name"] == "test_results.list"


# ── the crossing: GET /test-results ────────────────────────────────────


def test_serves_envelope_with_run_totals():
    resp = TestClient(_app(auth_user={"sub": "user1"})).get("/test-results")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    data = body["data"]
    assert isinstance(data["suites"], list)
    assert data["total"] == data["passed"] + data["failed"] + data["unknown"]
    for suite in data["suites"]:
        assert set(suite) >= {"file", "mtime", "total", "passed", "failed", "unknown", "items"}
        for item in suite["items"]:
            # real bool or None — never the source's 'True'/'False' strings
            assert item["passed"] is None or isinstance(item["passed"], bool)


def test_devtools_telemetry_is_not_a_test_run():
    """devtools_telemetry.json is a different artifact (step/console_count/…)
    — it must not be projected as a suite of test runs."""
    resp = TestClient(_app(auth_user={"sub": "user1"})).get("/test-results")
    files = [s["file"] for s in resp.json()["data"]["suites"]]
    assert "devtools_telemetry.json" not in files


def _write_results_dir(tmp_path: Path, monkeypatch) -> Path:
    results = tmp_path / "test_results"
    results.mkdir()
    (results / "journey.json").write_text(
        json.dumps(
            [
                {"test": "ok case", "passed": "True", "detail": "fine"},
                {"test": "bad case", "passed": "False", "detail": "broke"},
            ]
        )
    )
    (results / "weird.json").write_text(
        json.dumps([{"test": "odd case", "passed": "MAYBE", "detail": "?"}])
    )
    (results / "devtools_telemetry.json").write_text(
        json.dumps([{"step": 1, "console_count": 0, "error_count": 0}])
    )
    (results / "notes.md").write_text("narrative, not a run")
    monkeypatch.setattr(test_results_mod, "RESULTS_DIR", results)
    return results


def test_false_string_normalizes_to_real_false(tmp_path, monkeypatch):
    """bool('False') is True — the projection must deliver an actual False."""
    _write_results_dir(tmp_path, monkeypatch)
    resp = TestClient(_app(auth_user={"sub": "user1"})).get("/test-results")
    data = resp.json()["data"]
    by_file = {s["file"]: s for s in data["suites"]}
    assert set(by_file) == {"journey.json", "weird.json"}  # telemetry + md excluded
    items = {i["test"]: i["passed"] for i in by_file["journey.json"]["items"]}
    assert items["ok case"] is True
    assert items["bad case"] is False  # identity check: not 'False', not truthy
    assert {i["test"]: i["passed"] for i in by_file["weird.json"]["items"]} == {"odd case": None}
    assert by_file["journey.json"]["passed"] == 1
    assert by_file["journey.json"]["failed"] == 1
    assert by_file["weird.json"]["unknown"] == 1
    assert data["passed"] == 1 and data["failed"] == 1 and data["unknown"] == 1
    assert data["total"] == 3


def test_missing_results_dir_is_empty_not_500(tmp_path, monkeypatch):
    monkeypatch.setattr(test_results_mod, "RESULTS_DIR", tmp_path / "does-not-exist")
    resp = TestClient(_app(auth_user={"sub": "user1"})).get("/test-results")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["suites"] == [] and data["total"] == 0
