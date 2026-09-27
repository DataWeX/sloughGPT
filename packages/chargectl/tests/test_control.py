"""Tests for chargectl.control — capability probe and threshold writes."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from chargectl.control import clear_limit, probe, set_limit


def make_sysfs(tmp_path: Path, *, threshold: str | None = "100\n") -> Path:
    base = tmp_path / "power_supply"
    bat = base / "BAT0"
    bat.mkdir(parents=True)
    (bat / "capacity").write_text("55")
    (bat / "type").write_text("Battery")
    if threshold is not None:
        (bat / "charge_control_end_threshold").write_text(threshold)
    return base


def test_probe_unsupported_without_battery(tmp_path: Path):
    cap = probe(tmp_path / "nope")
    assert cap.supported is False
    assert cap.writable is False
    assert cap.path is None
    assert cap.reason


def test_probe_unsupported_without_threshold_file(tmp_path: Path):
    cap = probe(make_sysfs(tmp_path, threshold=None))
    assert cap.supported is False
    assert "charge_control_end_threshold" in cap.reason


def test_probe_supported_and_writable(tmp_path: Path):
    cap = probe(make_sysfs(tmp_path, threshold="80\n"))
    assert cap.supported is True
    assert cap.writable is True
    assert cap.current_limit == 80
    assert cap.path is not None and cap.path.endswith("charge_control_end_threshold")


def test_set_limit_writes_threshold(tmp_path: Path):
    base = make_sysfs(tmp_path, threshold="100\n")
    result = set_limit(80, sys_base=base)
    assert result.applied is True
    assert result.supported is True
    assert result.limit == 80
    written = (base / "BAT0" / "charge_control_end_threshold").read_text().strip()
    assert written == "80"


def test_set_limit_is_idempotent(tmp_path: Path):
    base = make_sysfs(tmp_path, threshold="100\n")
    first = set_limit(80, sys_base=base)
    second = set_limit(80, sys_base=base)
    assert first.applied and second.applied
    assert second.limit == 80


def test_clear_limit_writes_100(tmp_path: Path):
    base = make_sysfs(tmp_path, threshold="80\n")
    result = clear_limit(sys_base=base)
    assert result.applied is True
    assert result.limit == 100
    assert (base / "BAT0" / "charge_control_end_threshold").read_text().strip() == "100"


@pytest.mark.parametrize("bad", [0, 101, -5, 1000])
def test_set_limit_rejects_out_of_range(tmp_path: Path, bad: int):
    base = make_sysfs(tmp_path)
    result = set_limit(bad, sys_base=base)
    assert result.applied is False
    assert result.limit is None
    # nothing was written
    assert (base / "BAT0" / "charge_control_end_threshold").read_text().strip() == "100"


@pytest.mark.parametrize("bad", ["80", 80.0, None, True])
def test_set_limit_rejects_non_int(tmp_path: Path, bad):
    result = set_limit(bad, sys_base=make_sysfs(tmp_path))
    assert result.applied is False
    assert "integer" in result.reason


def test_set_limit_unsupported_returns_not_applied(tmp_path: Path):
    base = make_sysfs(tmp_path, threshold=None)
    result = set_limit(80, sys_base=base)
    assert result.applied is False
    assert result.supported is False
    assert result.reason


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write read-only files")
def test_set_limit_unwritable_reports_permission(tmp_path: Path):
    base = make_sysfs(tmp_path, threshold="100\n")
    path = base / "BAT0" / "charge_control_end_threshold"
    path.chmod(0o444)
    result = set_limit(80, sys_base=base)
    assert result.applied is False
    assert result.supported is True
    assert "root" in result.reason
    assert path.read_text().strip() == "100"


def test_nothing_raises_on_garbage_sysfs(tmp_path: Path):
    base = tmp_path / "power_supply"
    bat = base / "BAT0"
    bat.mkdir(parents=True)
    (bat / "type").write_text("Battery")
    (bat / "charge_control_end_threshold").write_text("not-a-number")
    cap = probe(base)
    assert cap.supported is True
    assert cap.current_limit is None
    result = set_limit(80, sys_base=base)
    assert result.applied is True  # write succeeded; readback is now valid


def test_control_results_are_serialisable(tmp_path: Path):
    base = make_sysfs(tmp_path)
    assert set_limit(80, sys_base=base).as_dict()["applied"] is True
    assert probe(base).as_dict()["supported"] is True
