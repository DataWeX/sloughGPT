"""Tests for Mole's system/host probe — thermal / load / uptime (card a101ee3a).

Port of ``55ba68693`` (doctor namespace) onto the rebranded Mole package
(rebrand ``dbe19d26c``: one package, one name — ``domain.core._internal.mole``).

TDD contract: written FIRST against the probe spec (RED — the probe module
and its registry entry do not exist yet), then the implementation makes it
GREEN. ``test_mole_probes.py`` must stay green alongside.

Hermetic by construction: the sensor readers are monkeypatched, so the
assertions never depend on the real host's temperature or load. One
runner-integration test executes the probe against the real host, but it
asserts structure only (probe executed, skipped probes recorded), never the
machine's current readings.

Probe spec (brief 2026-10-13 / card a101ee3a):

    thermal   every ``/sys/class/thermal/thermal_zone*`` (``type`` + ``temp``,
              millidegrees -> Celsius): average AND max across zones;
              severity from the max (warn >= 85 C, critical >= 95 C).
              Missing sysfs (non-Linux / CI) = explicit "unavailable"
              finding, never an exception.
    load      ``os.getloadavg()`` 1/5/15-min vs ``os.cpu_count()`` —
              load1 > ncpu is a busy (warn) finding (warn > 0.7x cores,
              critical > 1.5x).
    uptime    ``/proc/uptime`` seconds — < 30 min = recently rebooted
              (warn); long uptime surfaced as informational context only,
              never a failure.
"""

from __future__ import annotations

import pytest

from domain.core._internal.mole import run_mole
from domain.core._internal.mole.__main__ import _KNOWN_PROBES
from domain.core._internal.mole.models import band
from domain.core._internal.mole.probes import PROBES, ProbeResult
from domain.core._internal.mole.probes import system as system_probe
from domain.infrastructure._internal.health_flow import Severity

ALL_CHECKS = {"system.thermal", "system.load", "system.uptime"}


def _boom(*_args, **_kwargs):
    raise OSError("sensor unavailable")


@pytest.fixture
def host(monkeypatch):
    """Quiet 8-core host baseline; individual tests re-patch readers."""
    monkeypatch.setattr(system_probe, "_read_thermal_zones", lambda: [("x86_pkg_temp", 45.0)])
    monkeypatch.setattr(system_probe, "_read_loadavgs", lambda: (1.0, 1.0, 0.9))
    monkeypatch.setattr(system_probe, "_core_count", lambda: 8)
    monkeypatch.setattr(system_probe, "_read_uptime_s", lambda: 7200.0)
    return monkeypatch


def _check(findings, check: str):
    matches = [f for f in findings if f.check == check]
    assert matches, f"no {check} finding in {[f.check for f in findings]}"
    return matches[0]


class TestRegistration:
    def test_registered_first_in_registry(self):
        names = [entry["name"] for entry in PROBES]
        assert "system" in names
        # Registry is ordered fast → slow; local file reads beat network.
        assert names[0] == "system"

    def test_cli_skip_knows_system(self):
        assert "system" in _KNOWN_PROBES


class TestShape:
    def test_probe_result_shape(self, host):
        res = system_probe.run_probe()
        assert isinstance(res, ProbeResult)
        assert res.name == "system"
        assert res.ok is True
        assert res.error == ""
        assert {f.check for f in res.findings} >= ALL_CHECKS

    def test_findings_carry_system_component_and_own_source(self, host):
        res = system_probe.run_probe()
        assert res.findings
        for finding in res.findings:
            assert finding.component == "system"
            assert finding.source == "system"


class TestThermal:
    def test_cool_is_ok(self, host):
        finding = _check(system_probe.run_probe().findings, "system.thermal")
        assert finding.severity == Severity.OK
        assert finding.score >= 80

    def test_avg_and_max_across_zones_are_reported(self, host):
        host.setattr(
            system_probe,
            "_read_thermal_zones",
            lambda: [("x86_pkg_temp", 50.0), ("acpitz", 60.0)],
        )
        finding = _check(system_probe.run_probe().findings, "system.thermal")
        # Severity tracks the MAX zone (60 C → ok); both stats are surfaced.
        assert finding.severity == Severity.OK
        assert "max=60.0" in finding.detail
        assert "avg=55.0" in finding.detail

    def test_warm_is_warn(self, host):
        host.setattr(system_probe, "_read_thermal_zones", lambda: [("x86_pkg_temp", 90.0)])
        finding = _check(system_probe.run_probe().findings, "system.thermal")
        assert finding.severity == Severity.WARN
        assert band(finding.score) == Severity.WARN

    def test_hot_is_critical(self, host):
        host.setattr(
            system_probe,
            "_read_thermal_zones",
            lambda: [("x86_pkg_temp", 97.0), ("acpitz", 40.0)],
        )
        finding = _check(system_probe.run_probe().findings, "system.thermal")
        assert finding.severity == Severity.CRITICAL
        assert band(finding.score) == Severity.CRITICAL

    def test_missing_sysfs_is_explicit_unavailable_info(self, host):
        """Non-Linux / CI: no zones at all → an 'unavailable' finding, no exception."""
        host.setattr(system_probe, "_read_thermal_zones", lambda: [])
        finding = _check(system_probe.run_probe().findings, "system.thermal")
        assert finding.severity == Severity.INFO
        assert finding.score == 90
        assert "unavailable" in finding.message.lower()


class TestLoad:
    def test_quiet_is_ok(self, host):
        host.setattr(system_probe, "_read_loadavgs", lambda: (4.0, 3.0, 2.0))  # 0.5x of 8
        finding = _check(system_probe.run_probe().findings, "system.load")
        assert finding.severity == Severity.OK

    def test_busy_when_load1_exceeds_cpu_count(self, host):
        # Brief rule: load1 > ncpu = busy finding (1.125x > 0.7x warn line).
        host.setattr(system_probe, "_read_loadavgs", lambda: (9.0, 8.0, 7.0))
        finding = _check(system_probe.run_probe().findings, "system.load")
        assert finding.severity == Severity.WARN
        assert band(finding.score) == Severity.WARN

    def test_saturated_is_critical(self, host):
        host.setattr(system_probe, "_read_loadavgs", lambda: (16.0, 12.0, 8.0))  # 2.0x
        finding = _check(system_probe.run_probe().findings, "system.load")
        assert finding.severity == Severity.CRITICAL
        assert band(finding.score) == Severity.CRITICAL

    def test_detail_carries_1_5_15_minute_loads(self, host):
        finding = _check(system_probe.run_probe().findings, "system.load")
        assert "load1=" in finding.detail
        assert "load5=" in finding.detail
        assert "load15=" in finding.detail

    def test_unreadable_is_info(self, host):
        host.setattr(system_probe, "_read_loadavgs", _boom)
        finding = _check(system_probe.run_probe().findings, "system.load")
        assert finding.severity == Severity.INFO
        assert finding.score == 90

    def test_missing_core_count_is_info(self, host):
        host.setattr(system_probe, "_core_count", lambda: None)
        finding = _check(system_probe.run_probe().findings, "system.load")
        assert finding.severity == Severity.INFO
        assert finding.score == 90


class TestUptime:
    def test_recent_reboot_warns(self, host):
        host.setattr(system_probe, "_read_uptime_s", lambda: 60.0)
        finding = _check(system_probe.run_probe().findings, "system.uptime")
        assert finding.severity == Severity.WARN
        assert band(finding.score) == Severity.WARN

    def test_established_is_ok(self, host):
        finding = _check(system_probe.run_probe().findings, "system.uptime")
        assert finding.severity == Severity.OK

    def test_long_uptime_is_informational_never_a_failure(self, host):
        host.setattr(system_probe, "_read_uptime_s", lambda: 10 * 24 * 3600.0)
        finding = _check(system_probe.run_probe().findings, "system.uptime")
        assert finding.severity == Severity.INFO
        assert finding.score == 90
        assert finding.severity not in (Severity.WARN, Severity.CRITICAL)

    def test_unreadable_is_info(self, host):
        host.setattr(system_probe, "_read_uptime_s", _boom)
        finding = _check(system_probe.run_probe().findings, "system.uptime")
        assert finding.severity == Severity.INFO
        assert finding.score == 90


class TestNeverRaises:
    def test_every_reader_failing_still_returns_result(self, host):
        host.setattr(system_probe, "_read_thermal_zones", _boom)
        host.setattr(system_probe, "_read_loadavgs", _boom)
        host.setattr(system_probe, "_core_count", _boom)
        host.setattr(system_probe, "_read_uptime_s", _boom)
        res = system_probe.run_probe()
        assert isinstance(res, ProbeResult)
        assert res.ok is True
        assert {f.check for f in res.findings} >= ALL_CHECKS

    def test_scores_match_severity_bands(self, host):
        host.setattr(
            system_probe,
            "_read_thermal_zones",
            lambda: [("x86_pkg_temp", 90.0)],
        )
        host.setattr(system_probe, "_read_loadavgs", lambda: (16.0, 12.0, 8.0))
        res = system_probe.run_probe()
        assert res.findings
        for finding in res.findings:
            if finding.severity == Severity.INFO:
                assert finding.score == 90  # INFO convention (models.py)
            else:
                assert band(finding.score) == finding.severity


class TestRunnerIntegration:
    # Mole has five sibling probes (gates/benchmarks/http/sse/journey) —
    # all five are skipped so the runner executes ONLY the system probe
    # against the real host (structural assertions, never raw readings).
    _OTHERS = {"gates", "benchmarks", "http", "sse", "journey"}

    def test_run_mole_executes_system_when_others_skipped(self):
        report = run_mole(write=False, preflight=False, skip=self._OTHERS)
        by_name = {entry["name"]: entry for entry in report.probes}
        assert by_name["system"]["ok"] is True
        assert by_name["http"]["ok"] is None  # recorded as skipped
        assert any(f.component == "system" for f in report.findings)
        # Structural only — no assertion on the real host's readings.

    def test_report_serializes_system_findings_for_the_api(self):
        import json

        report = run_mole(write=False, preflight=False, skip=self._OTHERS)
        data = json.loads(json.dumps(report.to_dict()))
        checks = {f["check"] for f in data["findings"]}
        assert ALL_CHECKS <= checks
        assert {p["name"] for p in data["probes"]} >= {"system"}
