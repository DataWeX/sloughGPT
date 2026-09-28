"""control — capability-probed writes of the kernel's charge thresholds.

Prevents overcharge by capping charge at a chosen percentage (the 80% longevity rule)
and, where the kernel allows it, holding a floor as well:

* ``<BAT>/charge_control_end_threshold`` — stop charging at N% (widely available),
* ``<BAT>/charge_control_start_threshold`` — resume charging at N% (ThinkPad, ASUS,
  some framework kernels). With both set the pack simply rests between them, which is
  where the longevity band comes from.

Both files are often absent (VM, container, many chassis) and both need root. Nothing
here raises: every entry point returns a structured result so a caller can report
"unsupported"/"permission denied" instead of crashing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .status import DEFAULT_SYS_BASE

END_THRESHOLD = "charge_control_end_threshold"
START_THRESHOLD = "charge_control_start_threshold"
MIN_LIMIT = 1
MAX_LIMIT = 100
CLEARED_LIMIT = 100  # writing 100 disables the cap on every mainline driver

# Daemons that also write charge thresholds — writing over them is a two-daemons-fight.
INCUMBENTS: dict[str, str] = {
    "/etc/tlp.conf": "tlp",
    "/etc/tlp.d": "tlp",
    "/etc/default/tlp": "tlp",
}


@dataclass(frozen=True)
class Capability:
    """Whether this machine can cap charge, hold a floor, and who else might be writing."""

    supported: bool
    writable: bool
    path: str | None
    current_limit: int | None
    reason: str
    start_supported: bool = False
    start_path: str | None = None
    current_floor: int | None = None
    incumbent: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "supported": self.supported,
            "writable": self.writable,
            "path": self.path,
            "current_limit": self.current_limit,
            "reason": self.reason,
            "start_supported": self.start_supported,
            "start_path": self.start_path,
            "current_floor": self.current_floor,
            "incumbent": self.incumbent,
        }


@dataclass(frozen=True)
class ControlResult:
    """Outcome of a threshold write. ``applied`` is the only success signal."""

    applied: bool
    supported: bool
    limit: int | None
    reason: str
    path: str | None = None
    floor_limit: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "applied": self.applied,
            "supported": self.supported,
            "limit": self.limit,
            "reason": self.reason,
            "path": self.path,
            "floor_limit": self.floor_limit,
        }


def probe(sys_base: str | Path = DEFAULT_SYS_BASE) -> Capability:
    """Report whether a charge threshold can be written here. Never raises."""
    incumbent = _detect_incumbent()
    try:
        base = Path(sys_base)
        bat = _battery_dir(base)
        if bat is None:
            return Capability(
                False, False, None, None, "no battery device in sysfs", incumbent=incumbent
            )
        path = bat / END_THRESHOLD
        start_path = bat / START_THRESHOLD
        floor = _read_int(start_path) if start_path.is_file() else None
        start_supported = start_path.is_file()
        if not path.is_file():
            return Capability(
                False,
                False,
                str(path),
                None,
                f"kernel does not expose {END_THRESHOLD}",
                start_supported=start_supported,
                start_path=str(start_path) if start_supported else None,
                current_floor=floor,
                incumbent=incumbent,
            )
        current = _read_int(path)
        writable = os.access(path, os.W_OK)
        if not writable:
            return Capability(
                True,
                False,
                str(path),
                current,
                "threshold is read-only here — writing requires root",
                start_supported=start_supported,
                start_path=str(start_path) if start_supported else None,
                current_floor=floor,
                incumbent=incumbent,
            )
        return Capability(
            True,
            True,
            str(path),
            current,
            "ready",
            start_supported=start_supported,
            start_path=str(start_path) if start_supported else None,
            current_floor=floor,
            incumbent=incumbent,
        )
    except OSError as exc:
        return Capability(
            False, False, None, None, f"{type(exc).__name__}: {exc}", incumbent=incumbent
        )


def set_limit(percent: int, sys_base: str | Path = DEFAULT_SYS_BASE) -> ControlResult:
    """Cap charge at ``percent`` (1–100). Never raises."""
    if isinstance(percent, bool) or not isinstance(percent, int):
        return ControlResult(False, False, None, "limit must be an integer percent")
    if not MIN_LIMIT <= percent <= MAX_LIMIT:
        return ControlResult(
            False, False, None, f"limit must be between {MIN_LIMIT} and {MAX_LIMIT}"
        )

    cap = probe(sys_base)
    if not cap.supported:
        return ControlResult(False, False, None, cap.reason, cap.path)
    if not cap.writable:
        return ControlResult(False, True, cap.current_limit, cap.reason, cap.path)
    return _write(Path(str(cap.path)), percent)


def set_floor(percent: int, sys_base: str | Path = DEFAULT_SYS_BASE) -> ControlResult:
    """Make the pack resume charging at ``percent``. Never raises.

    Needs ``charge_control_start_threshold``; on kernels without it the result reports
    ``supported=False`` rather than pretending a floor exists.
    """
    if isinstance(percent, bool) or not isinstance(percent, int):
        return ControlResult(False, False, None, "floor must be an integer percent")
    if not MIN_LIMIT <= percent <= MAX_LIMIT:
        return ControlResult(
            False, False, None, f"floor must be between {MIN_LIMIT} and {MAX_LIMIT}"
        )

    cap = probe(sys_base)
    if not cap.start_supported or cap.start_path is None:
        return ControlResult(False, False, None, f"kernel does not expose {START_THRESHOLD}")
    if not cap.writable:
        return ControlResult(False, True, cap.current_floor, cap.reason, cap.start_path)
    return _write(Path(cap.start_path), percent)


def set_band(floor: int, ceiling: int, sys_base: str | Path = DEFAULT_SYS_BASE) -> ControlResult:
    """Hold the pack between ``floor`` and ``ceiling`` percent. Never raises.

    Writes the ceiling first (lowering the ceiling can never be rejected for an
    inverted range), then the floor. A machine with only an end threshold still gets
    the ceiling — ``floor_limit`` then reports ``None``.
    """
    for name, value in (("floor", floor), ("ceiling", ceiling)):
        if isinstance(value, bool) or not isinstance(value, int):
            return ControlResult(False, False, None, f"{name} must be an integer percent")
        if not MIN_LIMIT <= value <= MAX_LIMIT:
            return ControlResult(
                False, False, None, f"{name} must be between {MIN_LIMIT} and {MAX_LIMIT}"
            )
    if floor >= ceiling:
        return ControlResult(
            False, False, None, f"floor ({floor}%) must be below ceiling ({ceiling}%)"
        )

    cap = probe(sys_base)
    if not cap.supported:
        return ControlResult(False, False, None, cap.reason, cap.path)
    if not cap.writable:
        return ControlResult(False, True, cap.current_limit, cap.reason, cap.path)

    end_result = _write(Path(str(cap.path)), ceiling)
    if not end_result.applied:
        return end_result
    if not cap.start_supported or cap.start_path is None:
        return ControlResult(
            True,
            True,
            ceiling,
            f"charge capped at {ceiling}% — kernel has no start threshold, no floor held",
            end_result.path,
            None,
        )

    floor_result = _write(Path(cap.start_path), floor)
    if not floor_result.applied:
        return ControlResult(
            True,
            True,
            ceiling,
            f"ceiling {ceiling}% applied, floor failed: {floor_result.reason}",
            end_result.path,
            floor_result.limit,
        )
    return ControlResult(
        True,
        True,
        ceiling,
        f"charge held between {floor}-{ceiling}%",
        end_result.path,
        floor,
    )


def clear_limit(sys_base: str | Path = DEFAULT_SYS_BASE) -> ControlResult:
    """Lift the charge cap (writes 100). Never raises."""
    return set_limit(CLEARED_LIMIT, sys_base=sys_base)


def _write(path: Path, percent: int) -> ControlResult:
    try:
        path.write_text(f"{percent}\n")
    except OSError as exc:
        return ControlResult(False, True, None, f"{type(exc).__name__}: {exc}", str(path))

    readback = _read_int(path)
    if readback != percent:
        return ControlResult(
            False,
            True,
            readback,
            f"kernel clamped the threshold to {readback}",
            str(path),
        )
    return ControlResult(True, True, percent, f"charge threshold set to {percent}%", str(path))


def _detect_incumbent() -> str | None:
    for raw, name in INCUMBENTS.items():
        try:
            if Path(raw).exists():
                return name
        except OSError:  # pragma: no cover - unreadable path
            continue
    return None


def _battery_dir(base: Path) -> Path | None:
    if not base.is_dir():
        return None
    entries = [p for p in base.iterdir() if p.is_dir()]
    for p in entries:
        if p.name.upper().startswith("BAT"):
            return p
    for p in entries:
        type_path = p / "type"
        try:
            if type_path.is_file() and type_path.read_text().strip().lower() == "battery":
                return p
        except OSError:
            continue
    return None


def _read_int(path: Path) -> int | None:
    try:
        if path.is_file():
            return int(path.read_text().strip().split()[0])
    except (OSError, ValueError):
        return None
    return None
