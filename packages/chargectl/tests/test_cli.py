"""Tests for chargectl.cli — argument handling and output shape."""

from __future__ import annotations

from pathlib import Path

import pytest
from chargectl.cli import build_parser, main


def test_parser_defaults_to_status():
    args = build_parser().parse_args([])
    assert args.command is None


def test_parser_accepts_limit():
    args = build_parser().parse_args(["limit", "80"])
    assert args.command == "limit"
    assert args.percent == "80"


def test_status_prints_level(capsys: pytest.CaptureFixture[str]):
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "%" in out
    assert "state" in out


def test_advice_prints_action(capsys: pytest.CaptureFixture[str]):
    assert main(["advice"]) == 0
    out = capsys.readouterr().out
    assert "charge limit" in out


def test_limit_rejects_garbage(capsys: pytest.CaptureFixture[str]):
    assert main(["limit", "banana"]) == 2
    assert "invalid limit" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["off", "OFF", "clear", "100"])
def test_limit_off_aliases(value: str, capsys: pytest.CaptureFixture[str]):
    # never raises — returns 0 (applied) or 1 (unsupported), never a traceback
    assert main(["limit", value]) in (0, 1)
    capsys.readouterr()


# ── policy ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def policy_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("CHARGECTL_POLICY", str(tmp_path / "policy.json"))
    monkeypatch.setenv("CHARGECTL_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("CHARGECTL_SYSFS", str(tmp_path / "power_supply"))
    return tmp_path


def test_policy_show_defaults(capsys: pytest.CaptureFixture[str], policy_env: Path):
    assert main(["policy", "show"]) == 0
    out = capsys.readouterr().out
    assert "40-80" in out
    assert "policy off" in out


def test_policy_bare_invocation_shows(capsys: pytest.CaptureFixture[str], policy_env: Path):
    assert main(["policy"]) == 0
    assert "band" in capsys.readouterr().out


def test_policy_set_persists(capsys: pytest.CaptureFixture[str], policy_env: Path):
    assert main(["policy", "set", "--floor", "45", "--ceiling", "70", "--enable"]) == 0
    out = capsys.readouterr().out
    assert "45-70" in out and "enabled" in out
    assert (policy_env / "policy.json").is_file()

    assert main(["policy", "show"]) == 0
    assert "45-70" in capsys.readouterr().out


def test_policy_enable_disable_round_trip(capsys, policy_env: Path):
    assert main(["policy", "enable"]) == 0
    assert "enabled" in capsys.readouterr().out
    assert main(["policy", "disable"]) == 0
    assert "disabled" in capsys.readouterr().out


def test_policy_set_rejects_inverted_band(capsys: pytest.CaptureFixture[str], policy_env: Path):
    assert main(["policy", "set", "--floor", "90", "--ceiling", "50"]) == 2
    assert "invalid policy" in capsys.readouterr().out


def test_policy_parser_constrains_mode():
    args = build_parser().parse_args(["policy", "set", "--mode", "ceiling"])
    assert args.mode == "ceiling"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["policy", "set", "--mode", "turbo"])


# ── daemon ──────────────────────────────────────────────────────────────────


def test_parser_accepts_daemon_flags():
    args = build_parser().parse_args(["daemon", "--once", "--interval", "15"])
    assert args.command == "daemon"
    assert args.once is True
    assert args.interval == 15.0


def test_daemon_once_writes_state(capsys: pytest.CaptureFixture[str], policy_env: Path):
    assert main(["daemon", "--once"]) == 0
    state = (policy_env / "state.json").read_text()
    assert '"dry_run": true' in state  # CHARGECTL_SYSFS points at an empty dir
    capsys.readouterr()
