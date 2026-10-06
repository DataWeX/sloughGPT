"""Tests for the Mole (domain/core/_internal/mole) — Phase A.

Hermetic by construction: no network, no browser, no live ports. The
journey runner is stubbed via ``sys.modules``; http/sse probes take
injected readers; env defaults are set (and torn down) by fixture.
"""

from __future__ import annotations

import json
import os
import sys
import time
import types
from pathlib import Path

import pytest

from domain.core._internal.mole import run_mole
from domain.core._internal.mole.models import Finding, band, rank, worst
from domain.core._internal.mole.probes import ProbeResult
from domain.core._internal.mole.probes import benchmarks as benchmarks_probe
from domain.core._internal.mole.probes import gates as gates_probe
from domain.core._internal.mole.probes import http as http_probe
from domain.core._internal.mole.probes import journey as journey_probe
from domain.core._internal.mole.probes import sse as sse_probe
from domain.core._internal.mole.probes.sse import StreamAuthError
from domain.core._internal.mole.report import default_report_path, default_targets, merge
from domain.infrastructure._internal.health_flow import Diagnosis, Severity


@pytest.fixture(autouse=True)
def mole_env(monkeypatch, tmp_path):
    """Live-stack defaults + isolated journey report path for every test."""
    monkeypatch.setenv("SLO_WEB_URL", "http://localhost:5173")
    monkeypatch.setenv("SLO_API_URL", "http://localhost:8000")
    monkeypatch.setenv("SLO_JOURNEY_REPORT", str(tmp_path / "ux-flows-report.json"))
    monkeypatch.setenv("SLO_GATES_DIR", str(tmp_path / "gates"))
    monkeypatch.setenv("SLO_BENCH_RESULTS_DIR", str(tmp_path / "benchmarks"))
    monkeypatch.delenv("SLO_GATEWAY_URL", raising=False)
    monkeypatch.delenv("SLO_DOCTOR_REPORT", raising=False)
    monkeypatch.delenv("SLO_JOURNEY_CACHE", raising=False)
    monkeypatch.delenv("SLO_GATES_DRIFT_MAX", raising=False)
    monkeypatch.delenv("SLO_GATES_STALE_H", raising=False)
    monkeypatch.delenv("SLO_GATES_FULL_PASSED", raising=False)


# ──────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────

HEALTH_FRAME = json.dumps(
    {
        "stream": "health",
        "phase": "HEALTH",
        "status": "working",
        "data": {"model_loaded": True},
        "meta": {"ts": 0},
        "message": "ok",
    }
)


def _journey_report(path: Path, *, total=4, failed=0, console=0, network=0) -> dict:
    flows = []
    for i in range(total):
        failed_flow = i < failed
        flows.append(
            {
                "task": f"flow-{i}",
                "status": "failed" if failed_flow else "passed",
                "error": "boom" if failed_flow else "",
                "label": f"Flow {i}",
                "url": "/x",
                "spec": "### x",
                "steps_passed": 3,
                "steps_failed": 1 if failed_flow else 0,
                "steps_skipped": 0,
                "duration_ms": 10,
                "duration_s": 0.01,
            }
        )
    report = {
        "started_web": "http://localhost:5173",
        "browser": "firefox",
        "flows": flows,
        "passed": total - failed,
        "failed": failed,
        "console_error_count": console,
        "network_error_count": network,
        "console_errors_sample": ["console boom"][: min(console, 20)],
        "network_errors_sample": ["500 /chat"][: min(network, 20)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    return report


def _fetch(routes: dict[str, tuple[int, object]]):
    """Build an injected fetch from {path: (status, payload-bytes-or-JSON)}."""

    def fetch(url: str) -> tuple[int, bytes]:
        path = url.split("?", 1)[0]
        for route, (status, payload) in routes.items():
            if path.endswith(route):
                if isinstance(payload, bytes):
                    return status, payload
                return status, json.dumps({"status": "success", "data": payload}).encode()
        raise AssertionError(f"unexpected url: {url}")

    return fetch


def _stream_reader(frames_by_url: dict[str, list]):
    def reader(url: str, window_s: float):
        for fragment, frames in frames_by_url.items():
            if fragment in url:
                return iter(frames)
        return iter([])

    return reader


def _finding(result: ProbeResult, check: str) -> Finding:
    matches = [f for f in result.findings if f.check == check]
    assert matches, f"no finding {check!r} in {[f.check for f in result.findings]}"
    return matches[0]


UNHEALTHY_DETAIL = {
    "request_count": 100,
    "error_count": 7,
    "avg_latency_ms": 12.0,
    "p95_latency_ms": 88.0,
    "health_score": {
        "score": 41.0,
        "status": "unhealthy",
        "summary": "too many errors",
        "diagnoses": [],
    },
    "degraded": ["gpu"],
    "recent_errors": [
        {
            "path": "/chat",
            "method": "POST",
            "status": 500,
            "message": "boom",
            "error_type": "E_INTERNAL",
            "ts": 1.0,
        }
    ],
    "rate_violations": [{"path": "/api/x", "count": 3, "limit": 1, "ts": 2.0}],
}

HEALTHY_DETAIL = {
    "request_count": 10,
    "error_count": 0,
    "avg_latency_ms": 5.0,
    "p95_latency_ms": 9.0,
    "health_score": {"score": 92.0, "status": "healthy", "summary": "all good", "diagnoses": []},
    "degraded": [],
    "recent_errors": [],
    "rate_violations": [],
}


def _routes(detail: dict) -> dict:
    return {
        "/health/detailed": (200, detail),
        "/health": (200, {"status": "healthy", "model_loaded": True}),
        "/errors/grouped": (200, {"groups": [], "total_groups": 0}),
        "/errors/trends": (200, {"trends": [], "hours": 24}),
        "/errors/recent": (200, {"errors": [], "unread_count": 0, "total": 0}),
    }


# ──────────────────────────────────────────────────────────────────
# band / rank / worst
# ──────────────────────────────────────────────────────────────────


class TestBanding:
    def test_ok_band(self):
        assert band(100) == Severity.OK
        assert band(80) == Severity.OK

    def test_warn_band(self):
        assert band(79.9) == Severity.WARN
        assert band(50) == Severity.WARN

    def test_critical_band(self):
        assert band(49.9) == Severity.CRITICAL
        assert band(0) == Severity.CRITICAL

    def test_band_never_returns_info(self):
        for score in (0, 25, 49, 50, 79, 80, 100):
            assert band(score) != Severity.INFO


class TestRankWorst:
    def test_rank_worst_first_then_score(self):
        ok = Finding("s", "c.ok", Severity.OK, 100.0, "fine")
        info = Finding("s", "c.info", Severity.INFO, 90.0, "note")
        warn = Finding("s", "c.warn", Severity.WARN, 60.0, "meh")
        crit_high = Finding("s", "c.crit41", Severity.CRITICAL, 41.0, "bad")
        crit_low = Finding("s", "c.crit0", Severity.CRITICAL, 0.0, "worse")
        ranked = rank([ok, info, warn, crit_high, crit_low])
        assert [f.check for f in ranked] == ["c.crit0", "c.crit41", "c.warn", "c.info", "c.ok"]

    def test_worst_empty_is_ok(self):
        assert worst([]) == Severity.OK

    def test_worst_picks_highest(self):
        ok = Finding("s", "a", Severity.OK, 100.0, "x")
        info = Finding("s", "b", Severity.INFO, 90.0, "x")
        warn = Finding("s", "c", Severity.WARN, 60.0, "x")
        crit = Finding("s", "d", Severity.CRITICAL, 10.0, "x")
        assert worst([ok]) == Severity.OK
        assert worst([ok, info]) == Severity.INFO
        assert worst([info, warn]) == Severity.WARN
        assert worst([ok, warn, crit, info]) == Severity.CRITICAL

    def test_to_diagnosis_roundtrip(self):
        finding = Finding("http", "api.x", Severity.WARN, 55.0, "msg", "detail", "api")
        diag = finding.to_diagnosis()
        assert isinstance(diag, Diagnosis)
        assert (diag.check, diag.severity, diag.score, diag.message, diag.detail) == (
            "api.x",
            Severity.WARN,
            55.0,
            "msg",
            "detail",
        )


# ──────────────────────────────────────────────────────────────────
# journey probe
# ──────────────────────────────────────────────────────────────────


class TestJourneyProbe:
    def test_missing_report_is_info_and_probe_fails(self):
        result = journey_probe.run_probe(False)
        assert result.ok is False
        assert result.error == "no report — run with sweep"
        finding = _finding(result, "journey.report")
        assert finding.severity == Severity.INFO

    def test_failed_flows_fraction_is_critical(self):
        _journey_report(Path(os.environ["SLO_JOURNEY_REPORT"]), total=4, failed=2, console=3)
        result = journey_probe.run_probe(False)
        assert result.ok is True
        flows = _finding(result, "journey.flows")
        assert flows.severity == Severity.CRITICAL  # 2/4 = 50% >= 25%
        assert "2/4 journeys failed" in flows.message
        assert "Flow 0" in flows.detail and "Flow 1" in flows.detail
        console = _finding(result, "journey.console_errors")
        assert console.severity == Severity.WARN
        assert "console boom" in console.detail

    def test_single_failure_is_warn(self):
        _journey_report(Path(os.environ["SLO_JOURNEY_REPORT"]), total=13, failed=1)
        result = journey_probe.run_probe(False)
        assert _finding(result, "journey.flows").severity == Severity.WARN

    def test_three_failures_is_critical_even_below_fraction(self):
        _journey_report(Path(os.environ["SLO_JOURNEY_REPORT"]), total=20, failed=3)
        result = journey_probe.run_probe(False)
        assert _finding(result, "journey.flows").severity == Severity.CRITICAL

    def test_all_passed_is_ok(self):
        _journey_report(Path(os.environ["SLO_JOURNEY_REPORT"]), total=4)
        result = journey_probe.run_probe(False)
        assert _finding(result, "journey.flows").severity == Severity.OK
        assert result.findings == [_finding(result, "journey.flows")]

    def test_stale_report_is_info(self):
        path = Path(os.environ["SLO_JOURNEY_REPORT"])
        _journey_report(path, total=4)
        old = time.time() - 48 * 3600
        os.utime(path, (old, old))
        result = journey_probe.run_probe(False)
        stale = _finding(result, "journey.stale")
        assert stale.severity == Severity.INFO

    def test_env_default_applied_before_import(self, monkeypatch):
        monkeypatch.delenv("SLO_WEB_URL", raising=False)
        monkeypatch.delenv("SLO_API_URL", raising=False)
        journey_probe.run_probe(False)
        assert os.environ["SLO_WEB_URL"] == "http://localhost:5173"
        assert os.environ["SLO_API_URL"] == "http://localhost:8000"

    def test_sweep_uses_stub_runner(self, monkeypatch, tmp_path):
        # Import the real package first so its attributes bind to the REAL
        # runner; then swap only sys.modules so the probe picks up the stub.
        import domain.journeys  # noqa: F401

        report_path = tmp_path / "sweep-report.json"
        stub = types.ModuleType("domain.journeys.runner")
        calls: dict = {}

        def fake_run(flows, headed, strict_errors):
            calls["flows"] = len(flows)
            calls["headed"] = headed
            calls["strict_errors"] = strict_errors
            _journey_report(report_path, total=len(flows))
            return 0

        stub.run = fake_run
        stub.REPORT = str(report_path)
        stub.main = lambda *a, **k: 0
        monkeypatch.setitem(sys.modules, "domain.journeys.runner", stub)

        result = journey_probe.run_probe(True)

        assert report_path.exists()
        assert calls == {"flows": 13, "headed": False, "strict_errors": False}
        assert result.ok is True
        assert _finding(result, "journey.flows").severity == Severity.OK


# ──────────────────────────────────────────────────────────────────
# http probe
# ──────────────────────────────────────────────────────────────────


class TestHttpProbe:
    def test_healthy_surface_produces_no_bad_findings(self):
        result = http_probe.run_probe(fetch=_fetch(_routes(HEALTHY_DETAIL)))
        assert result.ok is True
        assert _finding(result, "api.health_score").severity == Severity.OK
        assert not any(f.severity in (Severity.WARN, Severity.CRITICAL) for f in result.findings)
        assert set(result.raw) >= {"/health", "/health/detailed", "/errors/grouped"}

    def test_unhealthy_score_critical_with_recent_and_violations(self):
        result = http_probe.run_probe(fetch=_fetch(_routes(UNHEALTHY_DETAIL)))
        assert result.ok is True
        assert _finding(result, "api.health_score").severity == Severity.CRITICAL
        recent = _finding(result, "api.recent_error")
        assert recent.severity == Severity.WARN
        assert "POST /chat → 500" in recent.message
        assert _finding(result, "api.rate_violations").severity == Severity.WARN
        assert _finding(result, "api.degraded").severity == Severity.WARN
        traffic = _finding(result, "api.traffic")
        assert traffic.severity == Severity.INFO  # errors present → informational

    def test_degraded_score_is_warn(self):
        detail = dict(HEALTHY_DETAIL, health_score={"score": 62.0, "status": "degraded"})
        result = http_probe.run_probe(fetch=_fetch(_routes(detail)))
        assert _finding(result, "api.health_score").severity == Severity.WARN

    def test_error_group_thresholds(self):
        routes = _routes(HEALTHY_DETAIL)
        routes["/errors/grouped"] = (
            200,
            {
                "groups": [
                    {"fingerprint": "a", "message": "x" * 50, "count": 3},
                    {"fingerprint": "b", "message": "y" * 50, "count": 10},
                    {"fingerprint": "c", "message": "z" * 50, "count": 50},
                ],
                "total_groups": 3,
            },
        )
        result = http_probe.run_probe(fetch=_fetch(routes))
        groups = [f for f in result.findings if f.check == "api.error_group"]
        assert len(groups) == 2  # count 3 stays silent
        severities = sorted(str(f.severity) for f in groups)
        assert severities == ["critical", "warn"]

    def test_unreachable_api_is_critical(self):
        def fetch(url):
            raise OSError("connection refused")

        result = http_probe.run_probe(fetch=fetch)
        assert result.ok is False
        assert result.error == "connection refused"
        finding = _finding(result, "api.reachable")
        assert finding.severity == Severity.CRITICAL
        assert finding.message == "API unreachable"

    def test_health_http_error_is_critical(self):
        result = http_probe.run_probe(fetch=_fetch({"/health": (503, b"")}))
        assert result.ok is False
        assert _finding(result, "api.reachable").severity == Severity.CRITICAL

    def test_errors_endpoints_require_auth_is_info(self):
        routes = _routes(HEALTHY_DETAIL)
        routes["/errors/grouped"] = (401, b"")
        result = http_probe.run_probe(fetch=_fetch(routes))
        assert result.ok is True
        assert _finding(result, "api.auth_required").severity == Severity.INFO


# ──────────────────────────────────────────────────────────────────
# sse probe
# ──────────────────────────────────────────────────────────────────


class TestSseProbe:
    def test_oversized_frame_is_critical_with_byte_count(self):
        big = json.dumps(
            {
                "stream": "health",
                "phase": "HEALTH",
                "status": "working",
                "data": {"blob": "x" * 300_000},
                "message": "ok",
            }
        )
        frames = [(0.0, HEALTH_FRAME), (3.0, HEALTH_FRAME), (6.0, HEALTH_FRAME), (9.0, big)]
        reader = _stream_reader({"/health/stream": frames})
        result = sse_probe.run_probe(window_s=9.0, fetch_stream=reader)
        assert result.ok is True
        finding = _finding(result, "stream.frame_size")
        assert finding.severity == Severity.CRITICAL
        assert "oversized" in finding.message
        assert "B" in finding.message
        # Critical present → no nominal-cadence info claims everything is fine.
        assert not any(
            f.check == "stream.cadence" and f.severity == Severity.INFO for f in result.findings
        )

    def test_gap_is_stall_critical(self):
        frames = [(0.0, HEALTH_FRAME), (3.0, HEALTH_FRAME), (15.0, HEALTH_FRAME)]
        reader = _stream_reader({"/health/stream": frames})
        result = sse_probe.run_probe(window_s=9.0, fetch_stream=reader)
        finding = _finding(result, "stream.stall")
        assert finding.severity == Severity.CRITICAL
        assert "12.0s" in finding.message

    def test_error_frame_is_critical(self):
        err = json.dumps(
            {
                "stream": "health",
                "phase": "ERROR",
                "status": "error",
                "data": {"error": "snapshot failed"},
                "message": "snapshot failed",
            }
        )
        frames = [(0.0, HEALTH_FRAME), (3.0, err)]
        reader = _stream_reader({"/health/stream": frames})
        result = sse_probe.run_probe(window_s=9.0, fetch_stream=reader)
        finding = _finding(result, "stream.error_frame")
        assert finding.severity == Severity.CRITICAL
        assert "snapshot failed" in finding.message

    def test_single_frame_is_warn(self):
        reader = _stream_reader({"/health/stream": [(0.0, HEALTH_FRAME)]})
        result = sse_probe.run_probe(window_s=9.0, fetch_stream=reader)
        cadence = _finding(result, "stream.cadence")
        assert cadence.severity == Severity.WARN

    def test_nominal_cadence_is_info(self):
        frames = [(0.0, HEALTH_FRAME), (3.0, HEALTH_FRAME), (6.0, HEALTH_FRAME)]
        reader = _stream_reader({"/health/stream": frames})
        result = sse_probe.run_probe(window_s=9.0, fetch_stream=reader)
        cadence = _finding(result, "stream.cadence")
        assert cadence.severity == Severity.INFO
        assert "3 frames" in cadence.message

    def test_no_frames_is_probe_failure(self):
        result = sse_probe.run_probe(window_s=0.1, fetch_stream=_stream_reader({}))
        assert result.ok is False
        assert result.error == "no frames received"
        assert _finding(result, "stream.frames").severity == Severity.WARN

    def test_errors_stream_auth_is_info(self):
        def reader(url, window_s):
            if "/errors/stream" in url:
                raise StreamAuthError("HTTP 401")
            return iter([(0.0, HEALTH_FRAME), (3.0, HEALTH_FRAME)])

        result = sse_probe.run_probe(window_s=9.0, fetch_stream=reader)
        assert _finding(result, "errors_stream.auth").severity == Severity.INFO

    def test_errors_stream_events_are_warn(self):
        client_error = json.dumps(
            {
                "stream": "errors",
                "phase": "CLIENT_ERROR",
                "status": "working",
                "data": {"message": "frontend TypeError"},
            }
        )

        def reader(url, window_s):
            if "/errors/stream" in url:
                return iter([(0.5, client_error)])
            return iter([(0.0, HEALTH_FRAME), (3.0, HEALTH_FRAME)])

        result = sse_probe.run_probe(window_s=9.0, fetch_stream=reader)
        events = _finding(result, "errors_stream.events")
        assert events.severity == Severity.WARN
        assert "frontend TypeError" in events.detail


# ──────────────────────────────────────────────────────────────────
# local file probes — gates + benchmarks
# ──────────────────────────────────────────────────────────────────


def _gate_artifact(d: Path, name: str, summary: str, mtime: float | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text("===FAILURES===\n(stub output)\n" + summary + "\n", encoding="utf-8")
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def _bench_record(d: Path, kind: str, name: str, payload: dict, mtime: float) -> None:
    kd = d / kind
    kd.mkdir(parents=True, exist_ok=True)
    p = kd / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    os.utime(p, (mtime, mtime))


class TestGatesProbe:
    """Newest parseable suite*.out vs the known-drift baseline (card 56b49cf1)."""

    OK_SUMMARY = "= 641 failed, 46235 passed, 32 skipped, 197 errors in 5963.47s (1:39:23) ="

    def test_within_baseline_is_ok(self, monkeypatch, tmp_path):
        d = tmp_path / "gates"
        _gate_artifact(d, "suite-a.out", self.OK_SUMMARY)
        monkeypatch.setenv("SLO_GATES_DIR", str(d))
        result = gates_probe.run_probe()
        assert result.ok is True
        finding = _finding(result, "gates.baseline")
        assert finding.severity == Severity.OK
        assert finding.score == 100.0
        assert "838" in finding.message and "641" in finding.message
        assert result.raw["failed"] == 641
        assert result.raw["errors"] == 197

    def test_over_baseline_is_warn(self, monkeypatch, tmp_path):
        d = tmp_path / "gates"
        _gate_artifact(d, "suite-a.out", "= 900 failed, 45000 passed, 100 errors in 1.00s =")
        monkeypatch.setenv("SLO_GATES_DIR", str(d))
        result = gates_probe.run_probe()
        assert result.ok is True
        finding = _finding(result, "gates.baseline")
        assert finding.severity == Severity.WARN
        assert "1000" in finding.message and "838" in finding.message

    def test_double_baseline_is_critical(self, monkeypatch, tmp_path):
        d = tmp_path / "gates"
        _gate_artifact(d, "suite-a.out", "= 1000 failed, 44000 passed, 700 errors in 1.00s =")
        monkeypatch.setenv("SLO_GATES_DIR", str(d))
        result = gates_probe.run_probe()
        assert _finding(result, "gates.baseline").severity == Severity.CRITICAL

    def test_passed_only_summary_is_clean(self, monkeypatch, tmp_path):
        d = tmp_path / "gates"
        _gate_artifact(d, "suite-a.out", "= 46241 passed in 6000.00s (1:40:00) =")
        monkeypatch.setenv("SLO_GATES_DIR", str(d))
        result = gates_probe.run_probe()
        assert result.ok is True
        assert _finding(result, "gates.baseline").severity == Severity.OK
        assert result.raw["failed"] == 0 and result.raw["errors"] == 0
        assert result.raw["passed"] == 46241
        assert result.raw["full"] is True

    def test_scope_limited_run_is_info_not_a_verdict(self, monkeypatch, tmp_path):
        d = tmp_path / "gates"
        _gate_artifact(
            d,
            "suite-subset.out",
            "= 65 failed, 11785 passed, 188 deselected, 13 errors in 3260.02s (0:54:20) =",
        )
        monkeypatch.setenv("SLO_GATES_DIR", str(d))
        result = gates_probe.run_probe()
        assert result.ok is False
        scope = _finding(result, "gates.scope")
        assert scope.severity == Severity.INFO
        assert "11785" in scope.message and "40000" in scope.message
        assert "gates.baseline" not in {f.check for f in result.findings}
        assert result.raw["full"] is False

    def test_full_run_preferred_over_newer_subset(self, monkeypatch, tmp_path):
        d = tmp_path / "gates"
        now = time.time()
        _gate_artifact(d, "suite-full.out", self.OK_SUMMARY, mtime=now - 100)
        _gate_artifact(
            d, "suite-subset.out", "= 65 failed, 11785 passed, 13 errors in 3260.02s =", mtime=now
        )
        monkeypatch.setenv("SLO_GATES_DIR", str(d))
        result = gates_probe.run_probe()
        assert result.ok is True
        assert result.raw["file"] == "suite-full.out"
        assert _finding(result, "gates.baseline").severity == Severity.OK

    def test_baseline_env_override(self, monkeypatch, tmp_path):
        d = tmp_path / "gates"
        _gate_artifact(d, "suite-a.out", "= 900 failed, 45000 passed, 100 errors in 1.00s =")
        monkeypatch.setenv("SLO_GATES_DIR", str(d))
        monkeypatch.setenv("SLO_GATES_DRIFT_MAX", "1000")  # bad=1000 → still within
        result = gates_probe.run_probe()
        assert _finding(result, "gates.baseline").severity == Severity.OK

    def test_missing_dir_is_info_and_probe_fails(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SLO_GATES_DIR", str(tmp_path / "nope"))
        result = gates_probe.run_probe()
        assert result.ok is False
        assert result.error
        assert _finding(result, "gates.artifact").severity == Severity.INFO

    def test_unparsable_artifacts_warn(self, monkeypatch, tmp_path):
        d = tmp_path / "gates"
        d.mkdir()
        (d / "suite-a.out").write_text("killed mid-run\nno summary here\n", encoding="utf-8")
        monkeypatch.setenv("SLO_GATES_DIR", str(d))
        result = gates_probe.run_probe()
        assert result.ok is False
        assert _finding(result, "gates.parse").severity == Severity.WARN
        assert result.error

    def test_newest_artifact_wins(self, monkeypatch, tmp_path):
        d = tmp_path / "gates"
        now = time.time()
        _gate_artifact(d, "suite-old.out", "= 1 failed, 1 passed in 0.10s =", mtime=now - 100)
        _gate_artifact(d, "suite-new.out", self.OK_SUMMARY, mtime=now)
        monkeypatch.setenv("SLO_GATES_DIR", str(d))
        result = gates_probe.run_probe()
        assert result.raw["file"] == "suite-new.out"
        assert result.raw["failed"] == 641

    def test_falls_back_to_older_parsable_artifact(self, monkeypatch, tmp_path):
        d = tmp_path / "gates"
        now = time.time()
        _gate_artifact(d, "suite-old.out", self.OK_SUMMARY, mtime=now - 100)
        _gate_artifact(d, "suite-junk.out", "killed mid-run\n", mtime=now)
        monkeypatch.setenv("SLO_GATES_DIR", str(d))
        result = gates_probe.run_probe()
        assert result.ok is True
        assert result.raw["file"] == "suite-old.out"

    def test_stale_artifact_is_info(self, monkeypatch, tmp_path):
        d = tmp_path / "gates"
        old = time.time() - 48 * 3600
        _gate_artifact(d, "suite-a.out", self.OK_SUMMARY, mtime=old)
        monkeypatch.setenv("SLO_GATES_DIR", str(d))
        result = gates_probe.run_probe()
        assert _finding(result, "gates.stale").severity == Severity.INFO
        assert _finding(result, "gates.baseline").severity == Severity.OK


class TestBenchmarksProbe:
    """Verdict comes from scripts/benchmark_results.py's own is_regression."""

    def test_latency_regression_is_warn(self, monkeypatch, tmp_path):
        d = tmp_path / "benchmarks"
        now = time.time()
        _bench_record(d, "latency", "a.json", {"mean_ms": 100.0, "p95_ms": 200.0}, now - 60)
        _bench_record(d, "latency", "b.json", {"mean_ms": 150.0, "p95_ms": 210.0}, now)
        monkeypatch.setenv("SLO_BENCH_RESULTS_DIR", str(d))
        result = benchmarks_probe.run_probe()
        assert result.ok is True
        finding = _finding(result, "benchmarks.latency")
        assert finding.severity == Severity.WARN
        assert "latency" in finding.message
        assert "mean_ms" in finding.detail  # +50% vs the 20% rel threshold
        assert result.raw["latency"] == 2

    def test_clean_pair_is_ok(self, monkeypatch, tmp_path):
        d = tmp_path / "benchmarks"
        now = time.time()
        _bench_record(d, "latency", "a.json", {"mean_ms": 100.0, "p95_ms": 200.0}, now - 60)
        _bench_record(d, "latency", "b.json", {"mean_ms": 110.0, "p95_ms": 210.0}, now)
        monkeypatch.setenv("SLO_BENCH_RESULTS_DIR", str(d))
        result = benchmarks_probe.run_probe()
        assert result.ok is True
        finding = _finding(result, "benchmarks.latency")
        assert finding.severity == Severity.OK
        assert result.findings == [finding]

    def test_higher_is_better_drop_regresses(self, monkeypatch, tmp_path):
        d = tmp_path / "benchmarks"
        now = time.time()
        _bench_record(d, "stability", "a.json", {"overall": 95.0}, now - 60)
        _bench_record(d, "stability", "b.json", {"overall": 89.0}, now)  # -6 > 5.0 abs
        monkeypatch.setenv("SLO_BENCH_RESULTS_DIR", str(d))
        result = benchmarks_probe.run_probe()
        finding = _finding(result, "benchmarks.stability")
        assert finding.severity == Severity.WARN
        assert "overall" in finding.detail

    def test_missing_dir_is_info_and_probe_fails(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SLO_BENCH_RESULTS_DIR", str(tmp_path / "nope"))
        result = benchmarks_probe.run_probe()
        assert result.ok is False
        assert result.error
        assert _finding(result, "benchmarks.store").severity == Severity.INFO

    def test_single_record_is_silent(self, monkeypatch, tmp_path):
        d = tmp_path / "benchmarks"
        _bench_record(d, "latency", "a.json", {"mean_ms": 100.0}, time.time())
        monkeypatch.setenv("SLO_BENCH_RESULTS_DIR", str(d))
        result = benchmarks_probe.run_probe()
        assert result.ok is True
        assert result.findings == []  # no baseline yet — no nagging
        assert result.raw["latency"] == 1

    def test_untracked_kind_is_ignored(self, monkeypatch, tmp_path):
        d = tmp_path / "benchmarks"
        now = time.time()
        _bench_record(d, "training", "a.json", {"loss": 1.0}, now - 60)
        _bench_record(d, "training", "b.json", {"loss": 9.0}, now)  # no thresholds → no verdict
        monkeypatch.setenv("SLO_BENCH_RESULTS_DIR", str(d))
        result = benchmarks_probe.run_probe()
        assert result.ok is True
        assert result.findings == []

    def test_unreadable_record_is_warn(self, monkeypatch, tmp_path):
        d = tmp_path / "benchmarks"
        now = time.time()
        _bench_record(d, "latency", "a.json", {"mean_ms": 100.0}, now - 60)
        kd = d / "latency"
        (kd / "b.json").write_text("{truncated", encoding="utf-8")
        os.utime(kd / "b.json", (now, now))
        monkeypatch.setenv("SLO_BENCH_RESULTS_DIR", str(d))
        result = benchmarks_probe.run_probe()
        finding = _finding(result, "benchmarks.latency")
        assert finding.severity == Severity.WARN
        assert "unreadable" in finding.message


# ──────────────────────────────────────────────────────────────────
# merge + report + exit codes
# ──────────────────────────────────────────────────────────────────


class TestMerge:
    def _full_report(self):
        _journey_report(Path(os.environ["SLO_JOURNEY_REPORT"]), total=4, failed=2, console=3)
        journey_result = journey_probe.run_probe(False)
        http_result = http_probe.run_probe(fetch=_fetch(_routes(UNHEALTHY_DETAIL)))
        big = json.dumps(
            {
                "stream": "health",
                "phase": "HEALTH",
                "status": "working",
                "data": {"blob": "x" * 300_000},
            }
        )
        reader = _stream_reader(
            {"/health/stream": [(0.0, HEALTH_FRAME), (3.0, HEALTH_FRAME), (9.0, big)]}
        )
        sse_result = sse_probe.run_probe(window_s=9.0, fetch_stream=reader)
        return merge([journey_result, http_result, sse_result])

    def test_merge_ranks_and_counts(self):
        report = self._full_report()
        severities = [f.severity for f in report.findings]
        assert report.findings[0].check == "stream.frame_size"  # score 0 leads
        assert severities == ([Severity.CRITICAL] * 3 + [Severity.WARN] * 4 + [Severity.INFO] * 1)
        assert report.summary == {
            "total": 8,
            "by_severity": {"ok": 0, "info": 1, "warn": 4, "critical": 3},
        }
        assert report.overall == Severity.CRITICAL
        assert [p["name"] for p in report.probes] == ["journey", "http", "sse"]
        assert all(p["ok"] for p in report.probes)

    def test_exit_codes(self):
        report = self._full_report()
        assert report.exit_code() == 2

        warn_only = merge(
            [
                ProbeResult(
                    name="http",
                    findings=[Finding("http", "x", Severity.WARN, 60.0, "meh", component="api")],
                )
            ]
        )
        assert warn_only.exit_code() == 1
        assert warn_only.overall == Severity.WARN

        clean = merge(
            [
                ProbeResult(
                    name="http",
                    findings=[Finding("http", "x", Severity.OK, 100.0, "fine", component="api")],
                )
            ]
        )
        assert clean.exit_code() == 0
        assert clean.overall == Severity.OK

    def test_info_exit_codes(self):
        info_only = merge(
            [
                ProbeResult(
                    name="journey",
                    findings=[
                        Finding(
                            "journey",
                            "journey.report",
                            Severity.INFO,
                            90.0,
                            "note",
                            component="journeys",
                        )
                    ],
                )
            ]
        )
        assert info_only.exit_code() == 0
        assert info_only.exit_code(strict=True) == 1

    def test_empty_merge_is_ok(self):
        report = merge([])
        assert report.overall == Severity.OK
        assert report.exit_code() == 0
        assert report.summary == {
            "total": 0,
            "by_severity": {"ok": 0, "info": 0, "warn": 0, "critical": 0},
        }

    def test_skipped_probes_recorded(self):
        report = merge([], skipped=["http", "sse"])
        assert report.probes == [
            {"name": "http", "ok": None, "error": "skipped"},
            {"name": "sse", "ok": None, "error": "skipped"},
        ]


class TestReportSerialization:
    def test_write_creates_dirs_and_parses(self, tmp_path):
        target = tmp_path / "nested" / "findings.json"
        report = merge(
            [
                ProbeResult(
                    name="http",
                    findings=[Finding("http", "x", Severity.WARN, 60.0, "meh")],
                )
            ],
            targets={"web": "http://localhost:5173", "api": "http://localhost:8000"},
        )
        written = report.write(str(target))
        assert written == str(target)
        data = json.loads(target.read_text(encoding="utf-8"))
        assert data["schema_version"] == 1
        assert set(data) == {
            "schema_version",
            "ts",
            "targets",
            "probes",
            "findings",
            "summary",
            "overall",
        }
        assert data["overall"] == "warn"
        assert data["findings"][0]["severity"] == "warn"
        assert data["targets"]["web"] == "http://localhost:5173"
        assert not list(target.parent.glob("*.tmp"))  # tmp file was renamed away

    def test_default_path_env(self, monkeypatch):
        monkeypatch.setenv("SLO_DOCTOR_REPORT", "/tmp/x/mole.json")
        assert default_report_path() == "/tmp/x/mole.json"

    def test_default_targets_apply_5173_fix(self, monkeypatch):
        monkeypatch.delenv("SLO_WEB_URL", raising=False)
        monkeypatch.delenv("SLO_API_URL", raising=False)
        targets = default_targets()
        assert targets["web"] == "http://localhost:5173"
        assert targets["api"] == "http://localhost:8000"
        assert "gateway" not in targets


class TestRunMole:
    def test_all_skipped_is_clean_and_never_touches_network(self):
        report = run_mole(
            skip={"gates", "benchmarks", "http", "sse", "journey"},
            preflight=False,
            write=False,
        )
        assert report.overall == Severity.OK
        assert report.exit_code() == 0
        assert {p["name"] for p in report.probes} == {
            "gates",
            "benchmarks",
            "http",
            "sse",
            "journey",
        }
        assert all(p["ok"] is None and p["error"] == "skipped" for p in report.probes)

    def test_registry_runs_local_file_probes_first(self):
        from domain.core._internal.mole.probes import PROBES

        assert [p["name"] for p in PROBES] == [
            "gates",
            "benchmarks",
            "http",
            "sse",
            "journey",
        ]

    def test_writes_to_env_report_path(self, tmp_path, monkeypatch):
        path = tmp_path / "sub" / "findings.json"
        monkeypatch.setenv("SLO_DOCTOR_REPORT", str(path))
        run_mole(
            skip={"gates", "benchmarks", "http", "sse", "journey"},
            preflight=False,
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["overall"] == "ok"
        assert data["targets"]["web"] == "http://localhost:5173"
