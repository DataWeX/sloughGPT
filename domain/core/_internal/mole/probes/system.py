"""System/host probe — thermal, load, uptime (the environmental seam).

Reads the local machine only: no network, no writes, no live-stack
dependency — side-effect-free by construction (sysfs / proc reads and
``os.getloadavg``).

Sensors:

    thermal   every ``/sys/class/thermal/thermal_zone*/`` — ``type`` plus
              ``temp`` (millidegrees → Celsius); the AVERAGE and MAX across
              zones are both reported, severity comes from the max. The
              ``x86_pkg_temp`` zone — the sensor ``scripts/
              benchmark_latency.py`` gates on — is called out in the
              detail, so a mole finding and a latency run agree on why a
              machine is slow. No readable zone (non-Linux / CI / missing
              sysfs) → an explicit "unavailable" INFO finding, never an
              exception.
    load      ``os.getloadavg()`` 1/5/15-minute averages vs
              ``os.cpu_count()``; load1 > ncpu is therefore always a busy
              (warn) finding.
    uptime    ``/proc/uptime`` seconds — under 30 min means the box
              recently rebooted and caches/thermals are unsettled (warn);
              long uptime is surfaced as informational context for
              thermal-throttle reasoning, never a failure.

Port of 55ba68693 (doctor namespace) onto the rebranded Mole package.
Thresholds (card a101ee3a): thermal warn >= 85 C / critical >= 95 C on the
max zone; load warn > 0.7x cores / critical > 1.5x; uptime warn < 30 min,
long-uptime info at >= 7 days.

Severity follows the shared ``models.py`` contract — INFO findings carry
score 90, OK 100, WARN 50–79, CRITICAL < 50 (``band()`` agrees with every
non-INFO score). The probe never raises: a failing reader degrades to an
explicit INFO finding.
"""

from __future__ import annotations

import os
from pathlib import Path

from domain.core._internal.mole.models import Finding
from domain.infrastructure._internal.health_flow import Severity

from . import ProbeResult

THERMAL_WARN_C = 85.0
THERMAL_CRIT_C = 95.0
LOAD_WARN_RATIO = 0.7
LOAD_CRIT_RATIO = 1.5
UPTIME_WARN_S = 30 * 60
LONG_UPTIME_S = 7 * 24 * 3600

THERMAL_BASE = Path("/sys/class/thermal")
UPTIME_PATH = Path("/proc/uptime")


def _read_thermal_zones() -> list[tuple[str, float]]:
    """All readable thermal zones as ``(type, celsius)``; ``[]`` when none.

    Read-only sysfs access: a missing ``/sys/class/thermal`` (non-Linux /
    CI) yields an empty list and an unreadable zone is skipped — this
    reader never raises.
    """
    zones: list[tuple[str, float]] = []
    try:
        entries = sorted(THERMAL_BASE.glob("thermal_zone*"))
    except OSError:
        return zones
    for zone in entries:
        try:
            zone_type = (zone / "type").read_text().strip()
            temp_c = int((zone / "temp").read_text()) / 1000
        except (OSError, ValueError):
            continue
        zones.append((zone_type, round(temp_c, 1)))
    return zones


def _read_loadavgs() -> tuple[float, float, float]:
    """1/5/15-minute load averages (raises OSError where unsupported)."""
    return os.getloadavg()


def _core_count() -> int | None:
    return os.cpu_count()


def _read_uptime_s() -> float | None:
    """Seconds since boot, or None when /proc/uptime is unreadable."""
    try:
        return float(UPTIME_PATH.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _finding(check: str, severity: Severity, score: float, message: str, detail: str) -> Finding:
    return Finding(
        source="system",
        check=check,
        severity=severity,
        score=score,
        message=message,
        detail=detail,
        component="system",
    )


def _thermal_finding() -> Finding:
    """Average + max across thermal zones; severity from the max."""
    try:
        zones = _read_thermal_zones()
    except OSError:
        zones = []
    if not zones:
        return _finding(
            "system.thermal",
            Severity.INFO,
            90.0,
            "Thermal zones unavailable",
            "no readable /sys/class/thermal/thermal_zone* — thermal state "
            "unknown (non-Linux host or CI)",
        )
    temps = [celsius for _, celsius in zones]
    avg = round(sum(temps) / len(temps), 1)
    top_type, top_c = max(zones, key=lambda zone: zone[1])
    pkg = next((celsius for name, celsius in zones if name == "x86_pkg_temp"), None)
    detail = (
        f"zones={len(zones)} avg={avg}C max={top_c}C (worst: {top_type})"
        + (f" pkg_temp={pkg}C" if pkg is not None else "")
        + f" warn >= {THERMAL_WARN_C:.0f}C critical >= {THERMAL_CRIT_C:.0f}C"
    )
    if top_c >= THERMAL_CRIT_C:
        return _finding(
            "system.thermal",
            Severity.CRITICAL,
            40.0,
            f"Max zone at {top_c} C — throttling territory",
            detail,
        )
    if top_c >= THERMAL_WARN_C:
        return _finding(
            "system.thermal",
            Severity.WARN,
            70.0,
            f"Max zone at {top_c} C — hot (avg {avg} C)",
            detail,
        )
    return _finding(
        "system.thermal",
        Severity.OK,
        100.0,
        f"Thermal nominal — max {top_c} C, avg {avg} C",
        detail,
    )


def _load_finding() -> Finding:
    """Load averages vs core count; load1 > ncpu is a busy (warn) finding."""
    try:
        load1, load5, load15 = _read_loadavgs()
        cores = _core_count()
    except OSError:
        return _finding(
            "system.load",
            Severity.INFO,
            90.0,
            "Load average unavailable",
            "os.getloadavg()/os.cpu_count() unreadable on this host",
        )
    if not cores:
        return _finding(
            "system.load",
            Severity.INFO,
            90.0,
            "Load average unavailable",
            "cpu_count unreadable — load cannot be judged against cores",
        )
    ratio = load1 / cores
    detail = (
        f"load1={load1:.2f} load5={load5:.2f} load15={load15:.2f} cores={cores} "
        f"ratio={ratio:.2f}x (warn > {LOAD_WARN_RATIO}x, critical > {LOAD_CRIT_RATIO}x)"
    )
    if ratio > LOAD_CRIT_RATIO:
        return _finding(
            "system.load",
            Severity.CRITICAL,
            40.0,
            f"Load {ratio:.1f}x core count — system saturated",
            detail,
        )
    if ratio > LOAD_WARN_RATIO:
        return _finding(
            "system.load",
            Severity.WARN,
            65.0,
            f"Load {ratio:.1f}x core count — contended",
            detail,
        )
    return _finding("system.load", Severity.OK, 100.0, "Load nominal", detail)


def _uptime_finding() -> Finding:
    """Recent reboot = warn; long uptime = informational context, never a failure."""
    try:
        uptime = _read_uptime_s()
    except OSError:
        uptime = None
    if uptime is None:
        return _finding(
            "system.uptime",
            Severity.INFO,
            90.0,
            "Uptime unavailable",
            "/proc/uptime unreadable",
        )
    if uptime < UPTIME_WARN_S:
        return _finding(
            "system.uptime",
            Severity.WARN,
            60.0,
            f"Up only {int(uptime // 60)} min — recently rebooted",
            f"uptime={uptime:.0f}s warn < {UPTIME_WARN_S}s — caches and "
            "thermals may not be settled; treat fresh measurements carefully",
        )
    if uptime >= LONG_UPTIME_S:
        return _finding(
            "system.uptime",
            Severity.INFO,
            90.0,
            f"Up {uptime / 86400:.1f} days — long uptime",
            f"uptime={uptime:.0f}s — informational context, not a failure: no "
            "recent reboot to reset thermals; sustained-uptime thermal drift "
            "is possible context for throttle reasoning",
        )
    return _finding(
        "system.uptime",
        Severity.OK,
        100.0,
        f"Up {uptime / 3600:.1f} h",
        f"uptime={uptime:.0f}s",
    )


def run_probe() -> ProbeResult:
    """Observe the host; read-only, never raises — failures degrade to INFO."""
    findings = [_thermal_finding(), _load_finding(), _uptime_finding()]
    return ProbeResult(name="system", ok=True, findings=findings)
