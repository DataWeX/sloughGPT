"""Tests for chargectl.systemd — unit discovery, ExecStart parsing, status fields."""

from __future__ import annotations

from pathlib import Path

import pytest
from chargectl import systemd
from chargectl.systemd import parse_exec_start, service_status

UNIT = """
[Unit]
Description=chargectl

[Service]
ExecStart=/usr/local/bin/chargectl daemon
Environment=CHARGECTL_STATE=/var/lib/chargectl/state.json

[Install]
WantedBy=multi-user.target
"""


# ── parse_exec_start ───────────────────────────────────────────────────────


def test_parse_exec_start_drops_arguments():
    assert parse_exec_start(UNIT) == "/usr/local/bin/chargectl"


def test_parse_exec_start_keeps_quoted_path_with_spaces():
    unit = 'ExecStart="/home/me/Default Project/.venv/bin/chargectl" daemon\n'
    assert parse_exec_start(unit) == "/home/me/Default Project/.venv/bin/chargectl"


def test_parse_exec_start_skips_comments_and_disabled_directives():
    unit = "# ExecStart=/old/bin\n-ExecStart=/disabled/bin arg\nExecStart=/real/bin arg\n"
    assert parse_exec_start(unit) == "/real/bin"


def test_parse_exec_start_missing_is_none():
    assert parse_exec_start("[Unit]\nDescription=none\n") is None
    assert parse_exec_start("ExecStart=\n") is None


# ── systemd_available ──────────────────────────────────────────────────────


def test_systemd_available_needs_both_pid1_and_systemctl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    run_dir = tmp_path / "run"
    monkeypatch.setattr(systemd, "RUN_DIR", run_dir)
    monkeypatch.setattr(systemd.shutil, "which", lambda _name: "/usr/bin/systemctl")

    assert systemd.systemd_available() is False  # systemctl present, not pid 1

    run_dir.mkdir()
    assert systemd.systemd_available() is True

    monkeypatch.setattr(systemd.shutil, "which", lambda _name: None)
    assert systemd.systemd_available() is False  # pid 1 is systemd, no systemctl


# ── service_status ─────────────────────────────────────────────────────────


def _forbidden_runner(_argv: list[str]) -> str | None:
    raise AssertionError("runner must not be called in this path")


def test_status_without_systemd_short_circuits():
    status = service_status(systemd=False, runner=_forbidden_runner)
    assert status.systemd is False
    assert status.unit_installed is False
    assert status.unit_path is None
    assert status.pid is None
    assert "no systemd" in status.reason


def test_status_reports_missing_unit_with_install_hint(tmp_path: Path):
    status = service_status(
        unit_dirs=[tmp_path],
        systemd=True,
        runner=_forbidden_runner,
    )
    assert status.systemd is True
    assert status.unit_installed is False
    assert status.scope is None
    assert "make charge-svc" in status.reason


def test_status_reads_enabled_active_pid_and_exec_start(tmp_path: Path):
    (tmp_path / "chargectl.service").write_text(UNIT)

    def runner(argv: list[str]) -> str | None:
        if "is-enabled" in argv:
            return "enabled"
        if "is-active" in argv:
            return "active"
        if "MainPID" in argv:
            return "4242"
        return None

    status = service_status(unit_dirs=[tmp_path], systemd=True, runner=runner)
    assert status.unit_installed is True
    assert status.enabled == "enabled"
    assert status.active == "active"
    assert status.running is True
    assert status.pid == 4242
    assert status.exec_start == "/usr/local/bin/chargectl"
    assert status.unit_path == str(tmp_path / "chargectl.service")
    assert status.scope == "system"
    assert "running" in status.reason


def test_status_inactive_unit_has_no_pid(tmp_path: Path):
    (tmp_path / "chargectl.service").write_text(UNIT)

    def runner(argv: list[str]) -> str | None:
        if "is-enabled" in argv:
            return "disabled"
        if "is-active" in argv:
            return "inactive"
        if "MainPID" in argv:
            return "0"
        return None

    status = service_status(unit_dirs=[tmp_path], systemd=True, runner=runner)
    assert status.running is False
    assert status.pid is None
    assert status.enabled == "disabled"
    assert "not running" in status.reason


def test_status_tolerates_runner_timeouts(tmp_path: Path):
    (tmp_path / "chargectl.service").write_text(UNIT)

    def runner(_argv: list[str]) -> str | None:
        return None  # every systemctl call failed or timed out

    status = service_status(unit_dirs=[tmp_path], systemd=True, runner=runner)
    assert status.unit_installed is True
    assert status.enabled is None
    assert status.active is None
    assert status.running is False


# ── daemon_state_path ────────────────────────────────────────────────────────


def test_state_path_prefers_the_app_default_when_it_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    default = tmp_path / ".config" / "chargectl" / "state.json"
    default.parent.mkdir(parents=True)
    default.write_text("{}")
    system = tmp_path / "var" / "lib" / "chargectl" / "state.json"
    monkeypatch.setattr(systemd, "default_state_path", lambda: default)
    monkeypatch.setattr(systemd, "SYSTEM_STATE_PATH", system)

    assert systemd.daemon_state_path() == default


def test_state_path_falls_back_to_the_system_unit_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    default = tmp_path / "home" / "state.json"  # user unit never ran
    system = tmp_path / "var" / "state.json"
    system.parent.mkdir(parents=True)
    system.write_text("{}")
    monkeypatch.setattr(systemd, "default_state_path", lambda: default)
    monkeypatch.setattr(systemd, "SYSTEM_STATE_PATH", system)

    assert systemd.daemon_state_path() == system  # system install still visible


def test_state_path_stays_on_the_default_when_nothing_is_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    default = tmp_path / "state.json"
    monkeypatch.setattr(systemd, "default_state_path", lambda: default)
    monkeypatch.setattr(systemd, "SYSTEM_STATE_PATH", tmp_path / "nope.json")

    assert systemd.daemon_state_path() == default
