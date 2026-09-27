"""Tests for chargectl.status — sysfs reads, fallback, and simulation purity."""

from __future__ import annotations

from pathlib import Path

import pytest
from chargectl.status import (
    BatteryReader,
    ChargeStatus,
    SimulatedBattery,
    read_status,
)


def make_sysfs(
    tmp_path: Path,
    *,
    level: int = 55,
    status: str = "Charging",
    ac: bool = True,
    prefix: str = "BAT0",
    **files: str,
) -> Path:
    base = tmp_path / "power_supply"
    bat = base / prefix
    bat.mkdir(parents=True, exist_ok=True)
    (bat / "capacity").write_text(str(level))
    (bat / "status").write_text(status)
    (bat / "type").write_text("Battery")
    (bat / "health").write_text("Good")
    (bat / "voltage_now").write_text("1234567")  # µV → 1234 mV
    sign = "-" if status.lower() == "discharging" else ""
    (bat / "current_now").write_text(f"{sign}1500000")  # µA → 1500 mA
    (bat / "charge_full").write_text("4000000")
    for name, value in files.items():
        (bat / name).write_text(value)
    if ac:
        acp = base / "AC0"
        acp.mkdir(exist_ok=True)
        (acp / "online").write_text("1")
    return base


def test_reads_sysfs_battery(tmp_path: Path):
    base = make_sysfs(tmp_path, level=55, status="Charging")
    st = read_status(base)
    assert st.source == "sysfs"
    assert st.level == 55
    assert st.is_charging is True
    assert st.is_plugged is True
    assert st.name == "BAT0"
    assert st.health == "Good"


def test_unit_conversion(tmp_path: Path):
    st = read_status(make_sysfs(tmp_path))
    assert st.voltage_mv == 1234
    assert st.current_ma == 1500


def test_discharging_not_plugged(tmp_path: Path):
    st = read_status(make_sysfs(tmp_path, level=30, status="Discharging", ac=False))
    assert st.is_charging is False
    assert st.is_plugged is False
    assert st.time_to_full_min is None
    assert st.time_to_empty_min is not None


def test_energy_ratio_fallback(tmp_path: Path):
    base = tmp_path / "power_supply"
    bat = base / "BAT1"
    bat.mkdir(parents=True)
    (bat / "type").write_text("Battery")
    (bat / "energy_now").write_text("25000000")
    (bat / "energy_full").write_text("50000000")
    (bat / "status").write_text("Full")
    st = read_status(base)
    assert st.level == 50
    assert st.source == "sysfs"
    assert st.name == "BAT1"


def test_missing_sysfs_falls_back_to_simulated(tmp_path: Path):
    st = read_status(tmp_path / "nope")
    assert st.source == "simulated"
    assert st.name == "SIM0"
    assert 0 <= st.level <= 100


def test_simulation_is_idempotent_for_same_instant(tmp_path: Path):
    reader = BatteryReader(sys_base=tmp_path / "nope")
    now = 1_700_000_000.0
    first = reader.read(now=now)
    second = reader.read(now=now)
    assert first == second, "reads must not mutate state"


def test_simulation_advances_with_time(tmp_path: Path):
    sim = SimulatedBattery(level=50, mode="charging", plugged=True, since=1000.0)
    early = sim.snapshot(now=1000.0)
    late = sim.snapshot(now=1060.0)  # 60s = 1 min at 4%/min → +4%
    assert early.level == 50
    assert late.level == 54
    assert late.is_charging is True
    assert late.time_to_full_min is not None
    # the anchor itself never moved
    assert sim.level == 50
    assert sim.since == 1000.0


def test_simulation_discharge_and_clamp(tmp_path: Path):
    sim = SimulatedBattery(level=2, mode="discharging", plugged=False, since=0.0)
    st = sim.snapshot(now=600.0)  # 10 minutes at 1%/min → clamped at 0
    assert st.level == 0
    assert st.is_charging is False


def test_reanchor_returns_new_instance():
    sim = SimulatedBattery(level=40, mode="discharging")
    moved = sim.at(80, "charging")
    assert sim.level == 40
    assert moved.level == 80
    assert moved.mode == "charging"


def test_level_bands():
    def band(level: int) -> str:
        return ChargeStatus(
            level=level,
            is_charging=False,
            is_plugged=False,
            health="Good",
            capacity=-1,
            voltage_mv=-1,
            current_ma=-1,
            time_to_full_min=None,
            time_to_empty_min=None,
            source="simulated",
        ).level_band

    assert band(10) == "low"
    assert band(50) == "ok"
    assert band(85) == "high"
    assert band(99) == "full"


def test_as_dict_shape():
    st = read_status(Path("/nonexistent"))
    data = st.as_dict()
    for key in ("level", "is_charging", "source", "level_band", "updated_at"):
        assert key in data


@pytest.mark.parametrize("status", ["Charging", "Full", "Not charging", "Discharging"])
def test_every_kernel_status_is_readable(tmp_path: Path, status: str):
    st = read_status(make_sysfs(tmp_path, status=status, ac=status != "Discharging"))
    assert st.source == "sysfs"
    assert st.level == 55


# ── health metrics ──────────────────────────────────────────────────────────


def test_reads_health_metrics(tmp_path: Path):
    base = make_sysfs(
        tmp_path,
        cycle_count="212",
        energy_full="45000000",
        energy_full_design="50000000",
    )
    st = read_status(base)
    assert st.cycle_count == 212
    assert st.energy_full == 45_000_000
    assert st.energy_full_design == 50_000_000
    assert st.health_percent == 90.0
    assert st.as_dict()["cycle_count"] == 212


def test_health_is_unknown_when_the_nodes_are_absent(tmp_path: Path):
    st = read_status(make_sysfs(tmp_path))
    assert st.cycle_count == -1
    assert st.energy_full == -1
    assert st.energy_full_design == -1
    assert st.health_percent == -1.0


def test_simulation_reports_health():
    st = read_status(Path("/nonexistent"))
    assert st.simulated is True
    assert st.cycle_count == 12
    assert st.energy_full_design == 50_000_000
    assert st.health_percent == 96.0
