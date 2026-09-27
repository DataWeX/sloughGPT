# chargectl

Zero-dependency Linux battery charge reader, overcharge control, and longevity advice.

Three concerns, three modules, no shared mutable state:

| Module              | Job                                                                                  | Side effects                                    |
| ------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------- |
| `chargectl.status`  | **Measure** — read `/sys/class/power_supply`, fall back to a time-derived simulation | none (reads are idempotent)                     |
| `chargectl.control` | **Control** — capability-probed write of `charge_control_end_threshold`              | writes sysfs only when supported _and_ writable |
| `chargectl.advice`  | **Advise** — 80% longevity heuristic                                                 | none                                            |

## Why the split

Reading is always allowed; writing a charge threshold needs root and is not exposed by
every kernel/VM/chassis. So control returns a `ControlResult(applied, supported, limit,
reason, path)` instead of raising — callers report "not supported" rather than crashing.

The simulation is **anchored to wall-clock time**, not to a poll counter, so two reads at
the same instant return the same value. A `GET` handler can cache it safely.

## Usage

```python
from chargectl import read_status, probe, set_limit, clear_limit, optimize_hint

status = read_status()  # ChargeStatus, source="sysfs" or "simulated"
cap = probe()  # Capability(supported=..., writable=..., current_limit=...)
res = set_limit(80)  # ControlResult(applied=True/False, reason=...)
clear_limit()  # set_limit(100)
optimize_hint(status)  # {"limit": 80, "action": "...", "reason": "..."}
```

CLI:

```
chargectl status          # level, charging state, voltage, current, ETA
chargectl advice          # what to do to maximise longevity
chargectl probe           # can this machine's charge be capped?
chargectl limit 80        # cap charge at 80% (needs root)
chargectl limit off       # lift the cap
```

## Tests

```
python3 -m pytest packages/chargectl/tests -q
```
