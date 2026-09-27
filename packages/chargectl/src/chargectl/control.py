"""control — capability-probed write of the kernel's charge stop threshold.

Prevents overcharge by capping charge at a chosen percentage (the 80% longevity rule).
The kernel exposes this as ``<BAT>/charge_control_end_threshold`` (percent), but:

* the file often does not exist (VM, container, many chassis),
* writing it requires root.

Nothing here raises. Every entry point returns a structured result so a caller can
report "unsupported"/"permission denied" instead of crashing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .status import DEFAULT_SYS_BASE

END_THRESHOLD = "charge_control_end_threshold"
MIN_LIMIT = 1
MAX_LIMIT = 100
CLEARED_LIMIT = 100  # writing 100 disables the cap on every mainline driver


@dataclass(frozen=True)
class Capability:
    """Whether this machine can cap charge, and where."""

    supported: bool
    writable: bool
    path: str | None
    current_limit: int | None
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "supported": self.supported,
            "writable": self.writable,
            "path": self.path,
            "current_limit": self.current_limit,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ControlResult:
    """Outcome of a threshold write. ``applied`` is the only success signal."""

    applied: bool
    supported: bool
    limit: int | None
    reason: str
    path: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "applied": self.applied,
            "supported": self.supported,
            "limit": self.limit,
            "reason": self.reason,
            "path": self.path,
        }


def probe(sys_base: str | Path = DEFAULT_SYS_BASE) -> Capability:
    """Report whether a charge threshold can be written here. Never raises."""
    try:
        base = Path(sys_base)
        bat = _battery_dir(base)
        if bat is None:
            return Capability(False, False, None, None, "no battery device in sysfs")
        path = bat / END_THRESHOLD
        if not path.is_file():
            return Capability(
                False,
                False,
                str(path),
                None,
                f"kernel does not expose {END_THRESHOLD}",
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
            )
        return Capability(True, True, str(path), current, "ready")
    except OSError as exc:
        return Capability(False, False, None, None, f"{type(exc).__name__}: {exc}")


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

    path = Path(str(cap.path))
    try:
        path.write_text(f"{percent}\n")
    except OSError as exc:
        return ControlResult(
            False, True, cap.current_limit, f"{type(exc).__name__}: {exc}", str(path)
        )

    readback = _read_int(path)
    if readback != percent:
        return ControlResult(
            False,
            True,
            readback,
            f"kernel clamped the threshold to {readback}",
            str(path),
        )
    return ControlResult(True, True, percent, f"charge capped at {percent}%", str(path))


def clear_limit(sys_base: str | Path = DEFAULT_SYS_BASE) -> ControlResult:
    """Lift the charge cap (writes 100). Never raises."""
    return set_limit(CLEARED_LIMIT, sys_base=sys_base)


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
