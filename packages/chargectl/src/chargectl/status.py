"""status — measure battery charge from Linux sysfs, with a simulated fallback.

Reads are **pure**: no counter is bumped, no file is written, and the simulation is
anchored to wall-clock time rather than to a poll index. Two reads at the same instant
return the same value, so this is safe behind a cached ``GET`` handler.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_SYS_BASE = "/sys/class/power_supply"

CHARGING = "charging"
DISCHARGING = "discharging"
IDLE = "idle"


@dataclass(frozen=True)
class ChargeStatus:
    """Immutable snapshot of battery state."""

    level: int  # 0..100
    is_charging: bool
    is_plugged: bool
    health: str  # Good / Overheat / Cold / Unknown
    capacity: int  # design-vs-current (mAh) if known, else -1
    voltage_mv: int  # -1 if unknown
    current_ma: int  # -1 if unknown (positive = charging)
    time_to_full_min: int | None
    time_to_empty_min: int | None
    source: str  # "sysfs" | "simulated"
    name: str = "BAT0"
    updated_at: float = 0.0

    @property
    def simulated(self) -> bool:
        return self.source == "simulated"

    @property
    def level_band(self) -> str:
        """Coarse band for UI colouring: low / ok / high / full."""
        if self.level >= 95:
            return "full"
        if self.level < 20:
            return "low"
        if self.level > 80:
            return "high"
        return "ok"

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "is_charging": self.is_charging,
            "is_plugged": self.is_plugged,
            "health": self.health,
            "capacity": self.capacity,
            "voltage_mv": self.voltage_mv,
            "current_ma": self.current_ma,
            "time_to_full_min": self.time_to_full_min,
            "time_to_empty_min": self.time_to_empty_min,
            "source": self.source,
            "name": self.name,
            "level_band": self.level_band,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class SimulatedBattery:
    """Wall-clock-anchored fake battery for VMs, CI, and dev boxes.

    ``since`` is the timestamp at which ``level`` was true. Subsequent reads derive the
    current level from elapsed time — nothing is mutated by reading.
    """

    level: int = 67
    mode: str = IDLE
    plugged: bool = False
    since: float = field(default_factory=time.time)
    charge_rate: float = 4.0  # % per minute
    discharge_rate: float = 1.0  # % per minute

    def at(self, level: int, mode: str = IDLE, plugged: bool = False) -> SimulatedBattery:
        """Return a new battery re-anchored at ``level``/``mode`` from now."""
        return SimulatedBattery(
            level=max(0, min(100, int(level))),
            mode=mode,
            plugged=plugged or mode == CHARGING,
            since=time.time(),
            charge_rate=self.charge_rate,
            discharge_rate=self.discharge_rate,
        )

    def level_at(self, now: float) -> int:
        elapsed_min = max(0.0, (now - self.since) / 60.0)
        if self.mode == CHARGING:
            lvl = self.level + self.charge_rate * elapsed_min
        elif self.mode == DISCHARGING:
            lvl = self.level - self.discharge_rate * elapsed_min
        else:
            lvl = float(self.level)
        return max(0, min(100, int(round(lvl))))

    def snapshot(self, now: float | None = None) -> ChargeStatus:
        """Pure: derive status at ``now`` without changing any state."""
        ts = time.time() if now is None else now
        level = self.level_at(ts)
        charging = self.mode == CHARGING and level < 100
        plugged = self.plugged or charging
        elapsed_min = max(0.0, (ts - self.since) / 60.0)

        t_full: int | None = None
        t_empty: int | None = None
        if charging and self.charge_rate > 0:
            t_full = max(0, math.ceil((100 - self.level) / self.charge_rate - elapsed_min))
        elif self.mode == DISCHARGING and self.discharge_rate > 0 and level > 0:
            t_empty = max(0, math.ceil(level / self.discharge_rate))

        return ChargeStatus(
            level=level,
            is_charging=charging,
            is_plugged=plugged,
            health="Good",
            capacity=-1,
            voltage_mv=7400 if plugged else 7200,
            current_ma=1200 if charging else (-400 if not plugged else 0),
            time_to_full_min=t_full,
            time_to_empty_min=t_empty,
            source="simulated",
            name="SIM0",
            updated_at=ts,
        )


class BatteryReader:
    """Reads sysfs when a battery is present, otherwise the simulation."""

    __slots__ = ("_sys_base", "_sim")

    def __init__(
        self,
        sys_base: str | Path = DEFAULT_SYS_BASE,
        sim: SimulatedBattery | None = None,
    ) -> None:
        self._sys_base = Path(sys_base)
        self._sim = sim if sim is not None else SimulatedBattery()

    @property
    def sys_base(self) -> Path:
        return self._sys_base

    @property
    def sim(self) -> SimulatedBattery:
        return self._sim

    def read(self, now: float | None = None) -> ChargeStatus:
        """Return the current status. Safe to call from concurrent readers."""
        ts = time.time() if now is None else now
        status = _read_sysfs(self._sys_base, ts)
        if status is not None:
            return status
        return self._sim.snapshot(ts)


def read_status(
    sys_base: str | Path = DEFAULT_SYS_BASE,
    now: float | None = None,
    sim: SimulatedBattery | None = None,
) -> ChargeStatus:
    """One-shot convenience read. Equivalent to ``BatteryReader(...).read(now)``."""
    return BatteryReader(sys_base=sys_base, sim=sim).read(now=now)


# ── sysfs ────────────────────────────────────────────────────────────────────


def _read_sysfs(base: Path, now: float) -> ChargeStatus | None:
    try:
        bat = _find_battery(base)
        if bat is None:
            return None
        level = _read_int(bat / "capacity")
        if level is None:
            en_now = _read_int(bat / "energy_now")
            en_full = _read_int(bat / "energy_full")
            if en_now is not None and en_full and en_full > 0:
                level = int(en_now * 100 / en_full)
        if level is None:
            return None
        level = max(0, min(100, level))

        status_str = (_read_str(bat / "status") or "").strip().lower()
        is_charging = status_str == "charging"
        ac_online = _ac_online(base)
        is_plugged = ac_online or is_charging or status_str not in ("discharging", "not charging")

        cur = _read_int(bat / "current_now")
        if cur is not None and abs(cur) > 100_000:  # µA → mA
            cur //= 1000

        volt = _read_int(bat / "voltage_now")
        if volt is not None and volt > 100_000:  # µV → mV
            volt //= 1000

        cap = _read_int(bat / "charge_full")
        if cap is None:
            cap = _read_int(bat / "energy_full")
        if cap is None:
            cap = -1

        health = (_read_str(bat / "health") or "Unknown").strip()
        t_full, t_empty = _estimates(level, is_charging, cur, cap)

        return ChargeStatus(
            level=level,
            is_charging=is_charging,
            is_plugged=is_plugged,
            health=health,
            capacity=cap,
            voltage_mv=volt if volt is not None else -1,
            current_ma=cur if cur is not None else -1,
            time_to_full_min=t_full,
            time_to_empty_min=t_empty,
            source="sysfs",
            name=bat.name,
            updated_at=now,
        )
    except OSError:
        return None


def _estimates(
    level: int, is_charging: bool, current_ma: int | None, capacity: int
) -> tuple[int | None, int | None]:
    """Crude linear ETA. Units are not fully known (µAh vs µWh), so bounds are clamped."""
    t_full: int | None = None
    t_empty: int | None = None
    if is_charging and current_ma and current_ma > 0 and capacity > 0:
        remaining = 100 - level
        needed = int(capacity * remaining / 100) if capacity < 100_000 else remaining * 20
        t_full = int(needed / current_ma * 60)
        if t_full > 24 * 60:
            t_full = None
    elif not is_charging and current_ma and current_ma < 0 and level > 0:
        drain = -current_ma
        usable = capacity if 0 < capacity < 100_000 else 3000
        t_empty = int(level / 100 * usable / drain * 60)
        if t_empty > 24 * 60:
            t_empty = None
    return t_full, t_empty


def _find_battery(base: Path) -> Path | None:
    if not base.is_dir():
        return None
    entries = [p for p in base.iterdir() if p.is_dir()]
    for p in entries:
        if p.name.upper().startswith("BAT"):
            return p
    for p in entries:
        if (_read_str(p / "type") or "").strip().lower() == "battery":
            return p
    return None


def _ac_online(base: Path) -> bool:
    for p in base.iterdir():
        if not p.is_dir():
            continue
        if p.name.upper().startswith(("AC", "ADP")):
            if _read_int(p / "online") == 1:
                return True
    return False


def _read_int(path: Path) -> int | None:
    try:
        if path.is_file():
            return int(path.read_text().strip().split()[0])
    except (OSError, ValueError):
        return None
    return None


def _read_str(path: Path) -> str | None:
    try:
        if path.is_file():
            return path.read_text().strip()
    except OSError:
        return None
    return None
