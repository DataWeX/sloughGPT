"""
charge_manager — Linux charging/power-supply reader + optimizer.

Reads ``/sys/class/power_supply`` when available, falls back to a
simulated in-memory battery so the pane always renders (dev boxes,
VMs, CI). The optimizer suggests a charge limit to maximize longevity
(80% rule) and estimates time-to-full/empty.

Zero heavy deps. Intended for the shell pane binary ``chargectl``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChargeStatus:
    """Snapshot of current charging state."""

    level: int  # 0..100
    is_charging: bool
    is_plugged: bool
    health: str  # Good / Overheat / Cold / Unknown
    capacity: int  # design vs current (mAh) if known, else -1
    voltage_mv: int  # -1 if unknown
    current_ma: int  # -1 if unknown (positive = charging)
    time_to_full_min: int | None  # None when not charging or unknown
    time_to_empty_min: int | None  # None when charging or unknown
    source: str  # "sysfs" or "simulated"
    updated_at: float


class ChargeManager:
    """Polls Linux power-supply and exposes optimize hints.

    ``__slots__`` for low overhead — the TUI may hold a singleton.
    """

    __slots__ = ("_sim_level", "_sim_charging", "_sim_plugged", "_last", "_sys_base")

    SIM_STEP = 1  # % per poll when simulated and charging
    OPTIMAL_LIMIT = 80  # % — longevity sweet spot

    def __init__(self, sys_base: str | Path = "/sys/class/power_supply") -> None:
        self._sim_level: int = 67
        self._sim_charging: bool = False
        self._sim_plugged: bool = False
        self._last: ChargeStatus | None = None
        self._sys_base = Path(sys_base)

    # ── Public ────────────────────────────────────────────────────────

    def poll(self) -> ChargeStatus:
        """Read sysfs (or advance simulation) and return a snapshot."""
        st = self._read_sysfs()
        if st is None:
            st = self._advance_sim()
        self._last = st
        return st

    def status(self) -> ChargeStatus:
        """Last polled status (polls if never polled)."""
        if self._last is None:
            return self.poll()
        return self._last

    def optimize_hint(self, level: int | None = None) -> dict[str, Any]:
        """Heuristic charging advice.

        Returns ``{"limit": 80, "action": "...", "reason": "..."}``.
        """
        lvl = level if level is not None else self.status().level
        is_charging = self.status().is_charging
        if lvl >= 90 and is_charging:
            return {
                "limit": self.OPTIMAL_LIMIT,
                "action": "unplug",
                "reason": f"Battery at {lvl}% — unplug to preserve longevity (80% optimal).",
            }
        if lvl >= self.OPTIMAL_LIMIT and is_charging:
            return {
                "limit": self.OPTIMAL_LIMIT,
                "action": "cap_at_80",
                "reason": f"At {lvl}% and still charging — cap long-term charge to 80%.",
            }
        if lvl < 20 and not is_charging:
            return {
                "limit": 100,
                "action": "plug_in",
                "reason": f"Battery low ({lvl}%) — plug in to avoid deep discharge.",
            }
        return {
            "limit": self.OPTIMAL_LIMIT,
            "action": "maintain",
            "reason": f"Battery {lvl}% — { 'charging' if is_charging else 'on battery'} (optimal range 20–80%).",
        }

    def set_simulated(self, level: int, charging: bool, plugged: bool = True) -> ChargeStatus:
        """Force simulated state — useful for tests / demo."""
        self._sim_level = max(0, min(100, level))
        self._sim_charging = charging
        self._sim_plugged = plugged
        self._last = self._build_sim()
        return self._last

    # ── Sysfs ─────────────────────────────────────────────────────────

    def _read_sysfs(self) -> ChargeStatus | None:
        try:
            if not self._sys_base.is_dir():
                return None
            bats = [p for p in self._sys_base.iterdir() if p.name.startswith("BAT")]
            if not bats:
                return None
            bat = bats[0]
            acs = [p for p in self._sys_base.iterdir() if p.name.startswith("AC") or p.name.startswith("ADP")]
            ac_online = False
            if acs:
                for ac in acs:
                    v = self._read_int(ac / "online")
                    if v == 1:
                        ac_online = True
                        break
            level = self._read_int(bat / "capacity")
            if level is None:
                # fallback: energy_now / energy_full
                en_now = self._read_int(bat / "energy_now")
                en_full = self._read_int(bat / "energy_full")
                if en_now is not None and en_full and en_full > 0:
                    level = int(en_now * 100 / en_full)
            if level is None:
                return None
            level = max(0, min(100, level))
            status_str = self._read_str(bat / "status") or ""
            is_charging = status_str.strip().lower() == "charging"
            # Some kernels report "Charging" only when AC online; trust AC as plugged
            is_plugged = ac_online or is_charging or (status_str.strip().lower() not in ("discharging", "not charging"))
            # Health
            health = (self._read_str(bat / "health") or "Unknown").strip()
            # Voltage / current
            volt = self._read_int(bat / "voltage_now")
            if volt is not None and volt > 100000:  # µV → mV
                volt = volt // 1000
            else:
                volt = volt if volt is not None else -1
            cur = self._read_int(bat / "current_now")
            if cur is not None and abs(cur) > 100000:  # µA → mA
                cur = cur // 1000
            else:
                cur = cur if cur is not None else -1
            # Capacity
            cap = self._read_int(bat / "charge_full")
            if cap is None:
                cap = self._read_int(bat / "energy_full")
            if cap is None:
                cap = -1

            # Time estimates (crude: capacity / current)
            t_full = None
            t_empty = None
            if is_charging and cur and cur > 0 and cap > 0:
                remaining_pct = 100 - level
                # assume linear: full capacity corresponds to 100%
                needed_mah = int(cap * remaining_pct / 100) if cap < 100000 else remaining_pct * 20  # heuristic
                if cur > 0:
                    t_full = int(needed_mah / cur * 60)
                    if t_full > 24 * 60:
                        t_full = None
            elif not is_charging and cur and cur < 0 and level > 0:
                # discharging current negative
                drain = -cur
                if drain > 0:
                    # rough: level% corresponds to remaining capacity
                    t_empty = int(level / 100 * (cap if cap > 0 and cap < 100000 else 3000) / drain * 60)

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
                updated_at=time.time(),
            )
        except Exception:
            return None

    def _read_int(self, p: Path) -> int | None:
        try:
            if p.is_file():
                return int(p.read_text().strip().split()[0])
        except Exception:
            return None
        return None

    def _read_str(self, p: Path) -> str | None:
        try:
            if p.is_file():
                return p.read_text().strip()
        except Exception:
            return None
        return None

    # ── Simulated ─────────────────────────────────────────────────────

    def _advance_sim(self) -> ChargeStatus:
        # drift level when charging/discharging
        if self._sim_charging and self._sim_level < 100:
            self._sim_level = min(100, self._sim_level + 1)
        elif not self._sim_charging and not self._sim_plugged and self._sim_level > 0:
            # slow self-discharge
            if int(time.time()) % 3 == 0 and self._sim_level > 0:
                self._sim_level = max(0, self._sim_level - 1)
        return self._build_sim()

    def _build_sim(self) -> ChargeStatus:
        return ChargeStatus(
            level=self._sim_level,
            is_charging=self._sim_charging,
            is_plugged=self._sim_plugged or self._sim_charging,
            health="Good",
            capacity=-1,
            voltage_mv=7400 if self._sim_plugged else 7200,
            current_ma=1200 if self._sim_charging else -400,
            time_to_full_min=max(0, (100 - self._sim_level) * 2) if self._sim_charging else None,
            time_to_empty_min=max(0, self._sim_level * 3) if not self._sim_charging else None,
            source="simulated",
            updated_at=time.time(),
        )


# Singleton for shell reuse
_manager: ChargeManager | None = None


def get_charge_manager() -> ChargeManager:
    global _manager
    if _manager is None:
        _manager = ChargeManager()
    return _manager
