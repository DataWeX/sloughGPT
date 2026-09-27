"""chargectl — measure charge, cap overcharge, advise on longevity.

Zero dependencies, Linux sysfs, simulated fallback. See README.md.
"""

from __future__ import annotations

from .advice import OPTIMAL_LIMIT, optimize_hint
from .control import (
    Capability,
    ControlResult,
    clear_limit,
    probe,
    set_limit,
)
from .status import (
    DEFAULT_SYS_BASE,
    BatteryReader,
    ChargeStatus,
    SimulatedBattery,
    read_status,
)

__version__ = "0.1.0"

__all__ = [
    "BatteryReader",
    "Capability",
    "ChargeStatus",
    "ControlResult",
    "DEFAULT_SYS_BASE",
    "OPTIMAL_LIMIT",
    "SimulatedBattery",
    "clear_limit",
    "optimize_hint",
    "probe",
    "read_status",
    "set_limit",
    "__version__",
]
