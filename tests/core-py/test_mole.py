"""Tests for Mole (``domain/core/_internal/mole``) — the always-on monitor.

Hermetic by construction: the mole sweep is injected, sleeps are
captured, no network, no browser, no AI. Mole adds *time* to the
existing ``run_mole`` contract — cadence, a findings fingerprint, an
append-only journal, and change-only events (dedupe so the same
finding never alerts twice).
"""

from __future__ import annotations

import json
import os

from domain.core._internal.mole import (
    default_journal_path,
    fingerprint,
    load_context,
    run_watch,
)
from domain.core._internal.mole.models import Finding
from domain.core._internal.mole.probes import ProbeResult
from domain.core._internal.mole.report import merge
from domain.infrastructure._internal.health_flow import Severity


def _report(*messages: str, severity: Severity = Severity.WARN):
    findings = [
        Finding(
            source="http",
            check="api.health_score",
            severity=severity,
            score=60.0,
            message=msg,
        )
        for msg in messages
    ]
    return merge([ProbeResult(name="http", findings=findings)], targets={})


def _journal_lines(path):
    return [json.loads(raw) for raw in path.read_text().splitlines()]


# ──────────────────────────────────────────────────────────────────
# fingerprint
# ──────────────────────────────────────────────────────────────────


def test_fingerprint_ignores_payload_noise():
    """Same finding identity, jittering counters → same fingerprint.

    Findings carry live values (p95, health score, frame sizes) that
    change every sweep; re-alerting on them is alert fatigue, not signal.
    """
    one = fingerprint(_report("p95 12 ms"))
    jittered = fingerprint(_report("p95 999 ms"))
    assert one == jittered


def test_fingerprint_sensitive_to_identity_change():
    """A new finding, a removed one, or a band flip must change it."""
    base = fingerprint(_report("slow"))
    severity_flip = fingerprint(_report("slow", severity=Severity.CRITICAL))
    info_flip = fingerprint(_report("slow", severity=Severity.INFO))
    assert base != severity_flip
    assert base != info_flip


def test_default_context_has_loadavg_and_cpu():
    ctx = load_context()
    assert isinstance(ctx["loadavg"], list) and len(ctx["loadavg"]) == 3
    assert ctx["cpu_count"] == os.cpu_count()


# ──────────────────────────────────────────────────────────────────
# the watch loop
# ──────────────────────────────────────────────────────────────────


def test_watch_dedupes_events_and_journals(tmp_path):
    journal = tmp_path / "mole-journal.jsonl"
    # tick 2 is the same finding with different numbers (noise: must dedupe);
    # tick 3 flips the severity band (identity change: must alert).
    reports = [_report("slow"), _report("p95 999 ms"), _report("slow", severity=Severity.CRITICAL)]
    events = []
    sleeps = []

    exit_code = run_watch(
        interval_s=7.0,
        max_ticks=3,
        journal_path=str(journal),
        run=lambda **kwargs: reports.pop(0),
        sleep=sleeps.append,
        on_event=lambda line, report: events.append((line, report)),
        context=lambda: {"loadavg": [1.0, 1.0, 1.0], "cpu_count": 8},
    )

    lines = _journal_lines(journal)
    assert len(lines) == 3
    assert [line["tick"] for line in lines] == [1, 2, 3]
    assert sleeps == [7.0, 7.0]  # sleeps BETWEEN ticks only
    assert [line["changed"] for line in lines] == [True, False, True]
    assert len(events) == 2  # baseline + the real change; no repeats
    assert exit_code == 2  # last tick flipped critical → mole's exit 2


def test_watch_records_context_each_tick(tmp_path):
    journal = tmp_path / "mole.jsonl"
    run_watch(
        max_ticks=1,
        journal_path=str(journal),
        run=lambda **kwargs: _report("slow"),
        sleep=lambda seconds: None,
        context=lambda: {"loadavg": [1.5, 1.2, 1.0], "cpu_count": 8},
    )
    line = _journal_lines(journal)[0]
    assert line["context"]["cpu_count"] == 8
    assert line["context"]["loadavg"][0] == 1.5
    assert line["overall"] == str(Severity.WARN)


def test_watch_survives_run_failure(tmp_path):
    journal = tmp_path / "mole.jsonl"

    def boom(**kwargs):
        raise RuntimeError("probe stack exploded")

    exit_code = run_watch(
        max_ticks=2,
        journal_path=str(journal),
        run=boom,
        sleep=lambda seconds: None,
        on_event=lambda line, report: None,
        context=lambda: {},
    )
    lines = _journal_lines(journal)
    assert len(lines) == 2  # the loop never dies mid-run
    assert all("probe stack exploded" in line["error"] for line in lines)
    assert exit_code == 2  # error tick → critical exit code


def test_default_journal_path_env_override(monkeypatch):
    monkeypatch.setenv("SLO_MOLE_JOURNAL", "/tmp/custom-mole.jsonl")
    assert default_journal_path() == "/tmp/custom-mole.jsonl"
    monkeypatch.delenv("SLO_MOLE_JOURNAL")
    assert "slog-doctor" in default_journal_path()
