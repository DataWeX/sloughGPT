"""Tests for chargectl.control — capability probe and threshold writes."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from chargectl import control as control_mod
from chargectl.control import clear_limit, probe, set_band, set_floor, set_limit


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


# ── floor / band ────────────────────────────────────────────────────────────


def make_band_sysfs(tmp_path: Path, *, end: str = "100\n", start: str | None = "1\n") -> Path:
    base = make_sysfs(tmp_path, threshold=end)
    if start is not None:
        (base / "BAT0" / "charge_control_start_threshold").write_text(start)
    return base


def test_probe_reports_floor_capability(tmp_path: Path):
    cap = probe(make_band_sysfs(tmp_path, end="80\n", start="40\n"))
    assert cap.supported is True
    assert cap.start_supported is True
    assert cap.current_floor == 40
    assert cap.start_path is not None and cap.start_path.endswith("charge_control_start_threshold")
    assert cap.as_dict()["current_floor"] == 40


def test_probe_without_start_node_reports_no_floor(tmp_path: Path):
    cap = probe(make_band_sysfs(tmp_path, start=None))
    assert cap.start_supported is False
    assert cap.current_floor is None
    assert cap.start_path is None


def test_set_floor_writes_start_threshold(tmp_path: Path):
    base = make_band_sysfs(tmp_path, start="1\n")
    result = set_floor(40, sys_base=base)
    assert result.applied is True and result.limit == 40
    assert (base / "BAT0" / "charge_control_start_threshold").read_text().strip() == "40"


def test_set_floor_without_start_node_reports_unsupported(tmp_path: Path):
    base = make_band_sysfs(tmp_path, start=None)
    result = set_floor(40, sys_base=base)
    assert result.applied is False and result.supported is False
    assert "charge_control_start_threshold" in result.reason


@pytest.mark.parametrize("bad", [0, 101, -5])
def test_set_floor_rejects_out_of_range(tmp_path: Path, bad: int):
    base = make_band_sysfs(tmp_path, start="1\n")
    assert set_floor(bad, sys_base=base).applied is False
    assert (base / "BAT0" / "charge_control_start_threshold").read_text().strip() == "1"


def test_set_band_writes_both_thresholds(tmp_path: Path):
    base = make_band_sysfs(tmp_path, end="100\n", start="1\n")
    result = set_band(40, 80, sys_base=base)
    assert result.applied is True
    assert result.limit == 80
    assert result.floor_limit == 40
    bat = base / "BAT0"
    assert bat.joinpath("charge_control_end_threshold").read_text().strip() == "80"
    assert bat.joinpath("charge_control_start_threshold").read_text().strip() == "40"
    assert "40-80" in result.reason


def test_set_band_rejects_inverted_range(tmp_path: Path):
    base = make_band_sysfs(tmp_path)
    result = set_band(80, 40, sys_base=base)
    assert result.applied is False
    assert "floor" in result.reason
    assert (base / "BAT0" / "charge_control_end_threshold").read_text().strip() == "100"


def test_set_band_still_caps_without_a_floor_node(tmp_path: Path):
    base = make_band_sysfs(tmp_path, start=None)
    result = set_band(40, 80, sys_base=base)
    assert result.applied is True
    assert result.limit == 80
    assert result.floor_limit is None
    assert "no floor held" in result.reason


def test_set_band_reports_permission_not_crash(tmp_path: Path):
    if os.geteuid() == 0:
        pytest.skip("root can write read-only files")
    base = make_band_sysfs(tmp_path)
    (base / "BAT0" / "charge_control_end_threshold").chmod(0o444)
    result = set_band(40, 80, sys_base=base)
    assert result.applied is False and "root" in result.reason


def test_incumbent_detection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    fake_tlp = tmp_path / "tlp.conf"
    fake_tlp.write_text("# tlp\n")
    monkeypatch.setattr(control_mod, "INCUMBENTS", {str(fake_tlp): "tlp"})
    assert probe(make_sysfs(tmp_path / "a")).incumbent == "tlp"

    monkeypatch.setattr(control_mod, "INCUMBENTS", {})
    assert probe(make_sysfs(tmp_path / "b")).incumbent is None


def test_garbage_sysfs_still_probes(tmp_path: Path):
    base = make_band_sysfs(tmp_path, start="not-a-number")
    cap = probe(base)
    assert cap.start_supported is True
    assert cap.current_floor is None
