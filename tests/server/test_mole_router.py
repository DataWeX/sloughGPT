"""
Tests for the Mole router — GET /mole/report, POST /mole/run.

Hermetic: the report path is redirected to ``tmp_path`` via the
``SLO_DOCTOR_REPORT`` env var, and every live probe is stubbed at the
mole-package seam (``PROBES`` registry + ``_preflight``), so no network,
no SSE consumption, and no browser sweep can happen here.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.exception_handlers import register_all_handlers

import domain.core._internal.mole as mole_mod
from apps.api.server.routers.mole import router
from domain.core._internal.mole.models import Finding
from domain.core._internal.mole.probes import PROBES, ProbeResult
from domain.core._internal.mole.probes import journey as journey_probe
from domain.infrastructure._internal.health_flow import Severity

# ── fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    register_all_handlers(_app)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def mole_env(monkeypatch, tmp_path):
    """Isolated report path + live-stack target defaults (no gateway)."""
    report_path = tmp_path / "findings-report.json"
    journey_path = tmp_path / "ux-flows-report.json"
    monkeypatch.setenv("SLO_DOCTOR_REPORT", str(report_path))
    monkeypatch.setenv("SLO_WEB_URL", "http://localhost:5173")
    monkeypatch.setenv("SLO_API_URL", "http://localhost:8000")
    monkeypatch.setenv("SLO_JOURNEY_REPORT", str(journey_path))
    monkeypatch.delenv("SLO_GATEWAY_URL", raising=False)
    return {"report": report_path, "journey": journey_path}


def _stub_probe(name: str, finding_check: str) -> Callable[..., ProbeResult]:
    """A hermetic probe: no network, no stream, one finding."""

    def _run(**_kwargs) -> ProbeResult:
        return ProbeResult(
            name=name,
            ok=True,
            findings=[
                Finding(
                    source=name,
                    check=finding_check,
                    severity=Severity.OK,
                    score=100.0,
                    message=f"{name} nominal",
                    component="api",
                )
            ],
        )

    return _run


def _stub_probes(monkeypatch) -> list[str]:
    """Replace the http/sse registry entries and neutralize preflight.

    Returns the list the journey-sweep spy is expected to leave empty.
    """
    sweep_calls: list[str] = []

    def _sweep_must_not_run() -> tuple[str, str]:
        sweep_calls.append("sweep")
        raise AssertionError("journey sweep must never run from the API")

    monkeypatch.setattr(journey_probe, "_sweep", _sweep_must_not_run)
    monkeypatch.setattr(mole_mod, "_preflight", lambda api: None)
    # Index by NAME, not position — the registry grew (system/gates/benchmarks
    # now precede http/sse), so PROBES[0]/[1] no longer mean http/sse and the
    # real network probes used to slip past this "hermetic" helper.
    by_name = {entry["name"]: entry for entry in PROBES}
    monkeypatch.setitem(by_name["http"], "run", _stub_probe("http", "api.health"))
    monkeypatch.setitem(by_name["sse"], "run", _stub_probe("sse", "stream.cadence"))
    return sweep_calls


# ── GET /mole/report ──────────────────────────────────────────────────


class TestReportRead:
    def test_envelope_shape(self, client, mole_env):
        resp = client.get("/mole/report")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        data = body["data"]
        assert set(data) == {"report", "path", "age_s"}

    def test_missing_file_is_empty_state_not_error(self, client, mole_env):
        resp = client.get("/mole/report")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["report"] is None
        assert data["age_s"] is None
        assert data["path"] == str(mole_env["report"])

    def test_returns_report_and_age(self, client, mole_env):
        ts = time.time() - 120
        mole_env["report"].write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "ts": ts,
                    "targets": {"web": "http://localhost:5173"},
                    "probes": [{"name": "http", "ok": True, "error": ""}],
                    "findings": [
                        {
                            "source": "http",
                            "check": "api.health",
                            "severity": "ok",
                            "score": 100.0,
                            "message": "nominal",
                            "detail": "",
                            "component": "api",
                        }
                    ],
                    "summary": {"total": 1, "by_severity": {"ok": 1}},
                    "overall": "ok",
                }
            ),
            encoding="utf-8",
        )
        resp = client.get("/mole/report")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["report"]["overall"] == "ok"
        assert data["report"]["findings"][0]["component"] == "api"
        assert data["age_s"] is not None
        assert 115.0 <= data["age_s"] <= 135.0

    def test_corrupt_file_is_empty_state(self, client, mole_env):
        mole_env["report"].write_text("{not json", encoding="utf-8")
        resp = client.get("/mole/report")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["report"] is None
        assert data["age_s"] is None

    def test_non_object_json_is_empty_state(self, client, mole_env):
        mole_env["report"].write_text("[1, 2, 3]", encoding="utf-8")
        data = client.get("/mole/report").json()["data"]
        assert data["report"] is None


# ── POST /mole/run ────────────────────────────────────────────────────


class TestMoleRun:
    def test_returns_report_and_writes_it(self, client, mole_env, monkeypatch):
        sweep_calls = _stub_probes(monkeypatch)
        resp = client.post("/mole/run")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        report = body["data"]["report"]
        assert report["schema_version"] == 1
        assert report["summary"]["total"] == len(report["findings"])
        assert report["overall"] in {"ok", "info", "warn", "critical"}

        # Written atomically to $SLO_DOCTOR_REPORT and equal to the response.
        assert mole_env["report"].exists()
        on_disk = json.loads(mole_env["report"].read_text(encoding="utf-8"))
        assert on_disk == report

        # Light run only — the browser journey sweep was never triggered.
        assert sweep_calls == []
        journey_probe_result = next(p for p in report["probes"] if p["name"] == "journey")
        assert "must never run" not in str(journey_probe_result.get("error"))

    def test_reads_journey_findings_from_disk(self, client, mole_env, monkeypatch):
        sweep_calls = _stub_probes(monkeypatch)
        mole_env["journey"].write_text(
            json.dumps(
                {
                    "flows": [
                        {"task": "flow-0", "label": "Sign up", "status": "passed"},
                        {
                            "task": "flow-1",
                            "label": "Train a model",
                            "status": "failed",
                            "error": "boom",
                        },
                    ],
                    "passed": 1,
                    "failed": 1,
                    "console_error_count": 0,
                    "network_error_count": 0,
                }
            ),
            encoding="utf-8",
        )
        resp = client.post("/mole/run")
        assert resp.status_code == 200
        report = resp.json()["data"]["report"]
        journey = [f for f in report["findings"] if f["source"] == "journey"]
        assert journey, "journey findings must come from the on-disk sweep report"
        assert any("1/2 journeys failed" in f["message"] for f in journey)
        # _sweep (which raises) was never invoked.
        assert sweep_calls == []

    def test_probe_crash_does_not_fail_the_endpoint(self, client, mole_env, monkeypatch):
        _stub_probes(monkeypatch)

        def _boom(**_kwargs):
            raise RuntimeError("probe exploded")

        monkeypatch.setitem(PROBES[0], "run", _boom)
        resp = client.post("/mole/run")
        assert resp.status_code == 200
        report = resp.json()["data"]["report"]
        assert any(f["check"] == "mole.probe_crash" for f in report["findings"])

    def test_written_report_is_served_by_get(self, client, mole_env, monkeypatch):
        _stub_probes(monkeypatch)
        posted = client.post("/mole/run").json()["data"]["report"]
        fetched = client.get("/mole/report").json()["data"]
        assert fetched["report"] == posted
        assert fetched["age_s"] is not None
        assert fetched["age_s"] < 5.0


# ── POST /mole/run — optional body {window_s, skip} ────────────────────


class TestMoleRunBody:
    """The optional body that gives the shell `mole` command flag parity.

    Omission = historical contract (regression guard), out-of-range window
    and unknown probe names = 422 *before* anything runs, and ``run_sweep``
    stays inexpressible (``_stub_probes`` makes the sweep raise on call).
    """

    @pytest.fixture
    def window_seen(self, monkeypatch):
        """Hermetic probes; the sse probe records the window_s it was given."""
        _stub_probes(monkeypatch)
        seen: dict[str, float | None] = {}

        def _run(**kwargs):
            seen["window_s"] = kwargs.get("window_s")
            return ProbeResult(name="sse", ok=True, findings=[])

        sse = next(entry for entry in PROBES if entry["name"] == "sse")
        monkeypatch.setitem(sse, "run", _run)
        return seen

    def test_omitted_body_keeps_historical_defaults(self, client, mole_env, window_seen):
        resp = client.post("/mole/run")
        assert resp.status_code == 200
        # LIGHT_WINDOW_S (6.0), not the standalone CLI's 9.0 default.
        assert window_seen["window_s"] == 6.0

    def test_empty_body_object_equals_defaults(self, client, mole_env, window_seen):
        resp = client.post("/mole/run", json={})
        assert resp.status_code == 200
        assert window_seen["window_s"] == 6.0

    def test_window_s_body_is_honored(self, client, mole_env, window_seen):
        resp = client.post("/mole/run", json={"window_s": 11.0})
        assert resp.status_code == 200
        assert window_seen["window_s"] == 11.0

    @pytest.mark.parametrize("window", [0, 0.5, 301])
    def test_window_s_out_of_bounds_is_422(self, client, mole_env, window_seen, window):
        resp = client.post("/mole/run", json={"window_s": window})
        assert resp.status_code == 422
        # Validation precedes the run: nothing executed, report untouched.
        assert "window_s" not in window_seen
        assert not mole_env["report"].exists()

    def test_skip_is_honored(self, client, mole_env, monkeypatch):
        _stub_probes(monkeypatch)

        def _must_not_run(**_kwargs):
            raise AssertionError("a skipped probe must never execute")

        sse = next(entry for entry in PROBES if entry["name"] == "sse")
        monkeypatch.setitem(sse, "run", _must_not_run)
        resp = client.post("/mole/run", json={"skip": ["sse"]})
        assert resp.status_code == 200
        report = resp.json()["data"]["report"]
        sse = next(p for p in report["probes"] if p["name"] == "sse")
        assert sse["ok"] is None
        assert sse["error"] == "skipped"

    def test_unknown_probe_is_422_and_runs_nothing(self, client, mole_env, monkeypatch):
        _stub_probes(monkeypatch)
        resp = client.post("/mole/run", json={"skip": ["nope"]})
        assert resp.status_code == 422
        assert not mole_env["report"].exists()

    def test_body_cannot_enable_the_browser_sweep(self, client, mole_env, monkeypatch):
        # _stub_probes replaces journey's _sweep with a raising sentinel —
        # if the body could turn run_sweep on, this would fail loudly.
        sweep_calls = _stub_probes(monkeypatch)
        resp = client.post("/mole/run", json={"window_s": 9.0, "skip": ["system"]})
        assert resp.status_code == 200
        assert sweep_calls == []
