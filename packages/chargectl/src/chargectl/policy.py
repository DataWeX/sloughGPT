"""policy — what the battery *should* do, decided from inputs alone.

Three rules:

* **pure** — :func:`decide` takes a status, a capability and the thresholds that are
  currently in force, and returns the next write (or ``None``). No I/O, no clock, no
  globals, so the whole band matrix is testable without a battery.
* **drift is the trigger** — the kernel keeps a charge threshold until something resets
  it (reboot, AC re-plug, suspend, a vendor driver). So the policy is re-asserted whenever
  the value in force differs from the policy, not only when the level crosses a boundary.
  The hysteresis itself lives in hardware: ``charge_control_start_threshold`` +
  ``charge_control_end_threshold`` make the pack hold between floor and ceiling.
* **off by default** — a policy that is disabled clears only what this daemon wrote
  (``owned=True``), so turning management off never clobbers a cap a human set by hand.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .status import ChargeStatus

BAND = "band"
CEILING = "ceiling"
MODES = (BAND, CEILING)

MIN_PERCENT = 1
MAX_PERCENT = 100
DEFAULT_FLOOR = 40
DEFAULT_CEILING = 80
DEFAULT_INTERVAL = 60.0

SET_CEILING = "set_ceiling"
SET_FLOOR = "set_floor"
CLEAR = "clear"

POLICY_ENV = "CHARGECTL_POLICY"
STATE_ENV = "CHARGECTL_STATE"


@dataclass(frozen=True)
class Directive:
    """The single write the daemon should perform now (``value=None`` means lift)."""

    action: str
    value: int | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.action, "value": self.value, "reason": self.reason}


@dataclass(frozen=True)
class Policy:
    """Longevity band for one machine. ``enabled=False`` is the safe default."""

    enabled: bool = False
    floor: int = DEFAULT_FLOOR
    ceiling: int = DEFAULT_CEILING
    mode: str = BAND
    interval_seconds: float = DEFAULT_INTERVAL

    @property
    def band(self) -> str:
        return f"{self.floor}-{self.ceiling}"

    def normalized(self) -> tuple[Policy, str | None]:
        """Clamp into 1–100 and keep floor below ceiling. Returns ``(policy, error)``."""
        floor = max(MIN_PERCENT, min(MAX_PERCENT, int(self.floor)))
        ceiling = max(MIN_PERCENT, min(MAX_PERCENT, int(self.ceiling)))
        error = None
        if floor >= ceiling:
            error = f"floor ({floor}%) must be below ceiling ({ceiling}%)"
            if self.floor < self.ceiling:
                floor = min(floor, ceiling - 1)
            else:
                ceiling = min(MAX_PERCENT, floor + 1)
        return (
            replace(
                self,
                floor=floor,
                ceiling=ceiling,
                mode=self.mode if self.mode in MODES else BAND,
            ),
            error,
        )

    @property
    def controls_floor(self) -> bool:
        return self.mode == BAND

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "floor": self.floor,
            "ceiling": self.ceiling,
            "mode": self.mode,
            "interval_seconds": self.interval_seconds,
            "band": self.band,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Policy:
        """Build from untrusted JSON. Anything invalid falls back to the default."""
        if not isinstance(raw, dict):
            return cls()
        interval = raw.get("interval_seconds", DEFAULT_INTERVAL)
        try:
            interval = float(interval)
        except (TypeError, ValueError):
            interval = DEFAULT_INTERVAL
        mode = raw.get("mode", BAND)
        policy = cls(
            enabled=bool(raw.get("enabled", False)),
            floor=_clamp_int(raw.get("floor"), DEFAULT_FLOOR),
            ceiling=_clamp_int(raw.get("ceiling"), DEFAULT_CEILING),
            mode=mode if mode in MODES else BAND,
            interval_seconds=interval if interval >= 1.0 else DEFAULT_INTERVAL,
        )
        return policy.normalized()[0]


def normalize(
    floor: int | None = None,
    ceiling: int | None = None,
    mode: str | None = None,
    enabled: bool | None = None,
    interval_seconds: float | None = None,
    base: Policy | None = None,
) -> tuple[Policy, str | None]:
    """Validate a proposed policy. Returns ``(policy, error)``; ``error`` is ``None`` when ok."""
    policy = base or Policy()
    if floor is not None:
        if isinstance(floor, bool) or not isinstance(floor, int):
            return policy, "floor must be an integer percent"
        policy = replace(policy, floor=floor)
    if ceiling is not None:
        if isinstance(ceiling, bool) or not isinstance(ceiling, int):
            return policy, "ceiling must be an integer percent"
        policy = replace(policy, ceiling=ceiling)
    if mode is not None:
        if mode not in MODES:
            return policy, f"mode must be one of {', '.join(MODES)}"
        policy = replace(policy, mode=mode)
    if enabled is not None:
        policy = replace(policy, enabled=bool(enabled))
    if interval_seconds is not None:
        try:
            value = float(interval_seconds)
        except (TypeError, ValueError):
            return policy, "interval_seconds must be a number"
        if value < 1.0:
            return policy, "interval_seconds must be >= 1"
        policy = replace(policy, interval_seconds=value)
    return policy.normalized()


def decide(
    status: ChargeStatus,
    capability: Any,
    policy: Policy,
    *,
    current_limit: int | None = None,
    current_floor: int | None = None,
    owned: bool = False,
) -> Directive | None:
    """Return the next write, or ``None`` when nothing needs doing.

    ``capability`` is a :class:`~chargectl.control.Capability` (duck-typed to keep this
    module free of control imports). ``current_*`` are the thresholds in force right now.
    ``owned`` says whether *we* wrote ``current_limit`` — it is what makes "disable"
    safe: a cap a human set by hand is left alone.
    """
    if not policy.enabled:
        if owned and current_limit not in (None, MAX_PERCENT):
            return Directive(
                CLEAR,
                MAX_PERCENT,
                f"policy disabled — lifting the {current_limit}% cap this daemon set",
            )
        return None

    if not getattr(capability, "supported", False):
        return None

    if current_limit != policy.ceiling:
        was = "unreadable" if current_limit is None else f"{current_limit}%"
        return Directive(
            SET_CEILING,
            policy.ceiling,
            f"charge cap is {was} — re-asserting {policy.ceiling}% "
            f"({'band' if policy.controls_floor else 'ceiling'} mode)",
        )

    if policy.controls_floor and getattr(capability, "start_supported", False):
        if current_floor != policy.floor:
            was = "unset" if current_floor is None else f"{current_floor}%"
            return Directive(
                SET_FLOOR,
                policy.floor,
                f"charge floor is {was} — holding the pack at {policy.floor}-{policy.ceiling}%",
            )

    return None


def explain(policy: Policy, capability: Any) -> str:
    """One-line human summary of what the policy can actually do here."""
    if not policy.enabled:
        return "policy off — charge thresholds untouched"
    if not getattr(capability, "supported", False):
        return "policy on, but this machine exposes no charge threshold"
    if policy.controls_floor and not getattr(capability, "start_supported", False):
        return f"policy on — ceiling only ({policy.ceiling}%), kernel has no start threshold"
    return f"policy on — holding {policy.band}%"


def default_policy_path() -> Path:
    env = os.environ.get(POLICY_ENV)
    if env:
        return Path(env)
    return Path.home() / ".config" / "chargectl" / "policy.json"


def default_state_path() -> Path:
    env = os.environ.get(STATE_ENV)
    if env:
        return Path(env)
    return Path.home() / ".config" / "chargectl" / "state.json"


def load_policy(path: str | Path | None = None) -> tuple[Policy, str | None]:
    """Read the policy file. Missing file → default policy, no error."""
    target = Path(path) if path is not None else default_policy_path()
    try:
        if not target.is_file():
            return Policy(), None
        return Policy.from_dict(json.loads(target.read_text() or "{}")), None
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return Policy(), f"{type(exc).__name__}: {exc}"


def save_policy(policy: Policy, path: str | Path | None = None) -> Path:
    """Write the policy atomically (tmp file + ``os.replace``). Never raises."""
    target = Path(path) if path is not None else default_policy_path()
    _atomic_write(target, json.dumps(policy.as_dict(), indent=2) + "\n")
    return target


def _atomic_write(target: Path, payload: str) -> None:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".chargectl-", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, target)
    except OSError:
        # Best effort: a read-only root filesystem must not kill the daemon.
        return


def _clamp_int(value: Any, fallback: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(MIN_PERCENT, min(MAX_PERCENT, number))
