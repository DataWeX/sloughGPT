"""chargectl — measure charge, cap overcharge, manage a longevity band.

Zero dependencies, Linux sysfs, simulated fallback. See README.md.

Layout:

* :mod:`chargectl.status` — measure (pure reads)
* :mod:`chargectl.control` — one-shot threshold writes (never raises)
* :mod:`chargectl.policy` — pure rules: status + capability → next write
* :mod:`chargectl.daemon` — the loop that keeps thresholds from decaying
* :mod:`chargectl.advice` — static longevity advice for humans
"""

from __future__ import annotations

from .advice import OPTIMAL_LIMIT, optimize_hint
from .control import (
    Capability,
    ControlResult,
    clear_limit,
    probe,
    set_band,
    set_floor,
    set_limit,
)
from .daemon import read_state, run_daemon, tick, write_state
from .policy import (
    Directive,
    Policy,
    decide,
    default_policy_path,
    default_state_path,
    explain,
    load_policy,
    normalize,
    save_policy,
)
from .status import (
    DEFAULT_SYS_BASE,
    BatteryReader,
    ChargeStatus,
    SimulatedBattery,
    default_sys_base,
    read_status,
)
from .systemd import ServiceStatus, service_status, systemd_available

__version__ = "0.2.0"

__all__ = [
    "BatteryReader",
    "Capability",
    "ChargeStatus",
    "ControlResult",
    "DEFAULT_SYS_BASE",
    "Directive",
    "OPTIMAL_LIMIT",
    "Policy",
    "ServiceStatus",
    "SimulatedBattery",
    "clear_limit",
    "decide",
    "default_policy_path",
    "default_state_path",
    "default_sys_base",
    "explain",
    "load_policy",
    "normalize",
    "optimize_hint",
    "probe",
    "read_state",
    "read_status",
    "run_daemon",
    "save_policy",
    "service_status",
    "set_band",
    "set_floor",
    "set_limit",
    "systemd_available",
    "tick",
    "write_state",
    "__version__",
]
