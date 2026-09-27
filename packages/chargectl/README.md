# chargectl

Zero-dependency Linux battery charge reader, overcharge control, and a longevity band
that is actually _kept_ — measure, control, decide, enforce.

| Module              | Job                                                                                  | Side effects                                    |
| ------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------- |
| `chargectl.status`  | **Measure** — read `/sys/class/power_supply`, fall back to a time-derived simulation | none (reads are idempotent)                     |
| `chargectl.control` | **Control** — capability-probed write of end/start thresholds                        | writes sysfs only when supported _and_ writable |
| `chargectl.policy`  | **Decide** — pure rules: status + capability → the next write                        | none                                            |
| `chargectl.daemon`  | **Enforce** — re-assert the policy on a timer, record what it did                    | reads/writes policy + state files, writes sysfs |
| `chargectl.advice`  | **Advise** — 80% longevity heuristic                                                 | none                                            |

## Why the split

Reading is always allowed; writing a charge threshold needs root and is not exposed by
every kernel/VM/chassis. So control returns a `ControlResult(applied, supported, limit,
reason, path)` instead of raising — callers report "not supported" rather than crashing.

A threshold is also **state that decays**: reboot, AC re-plug, suspend and vendor drivers
reset it. One-shot writes are not management, so the policy is re-asserted whenever the
value in force differs from the policy — the hysteresis itself lives in hardware
(`charge_control_start_threshold` + `charge_control_end_threshold` hold the pack between
floor and ceiling).

The simulation is **anchored to wall-clock time**, not to a poll counter, so two reads at
the same instant return the same value. A `GET` handler can cache it safely.

## Policy

Off by default — enabling it is a decision, not a side effect.

```json
{
  "enabled": true,
  "floor": 40,
  "ceiling": 80,
  "mode": "band",
  "interval_seconds": 60
}
```

`mode: "band"` holds 40–80% where the kernel exposes a start threshold; otherwise the
ceiling applies alone and the state file says so. Disabling clears **only what the daemon
wrote** (`owned`), so a cap a human set by hand survives.

```bash
chargectl policy show
chargectl policy set --floor 40 --ceiling 80 --enable
chargectl policy disable
```

## Daemon

```bash
chargectl daemon              # loop until SIGTERM (systemd's job)
chargectl daemon --once       # one tick: read → decide → write → record
```

Each tick writes `~/.config/chargectl/state.json` (override with `CHARGECTL_STATE`):
policy, current thresholds, the directive chosen, the write result, and `owned`. The API
serves it, so the monitoring page can show _managed_ rather than guessing.

On a VM or container with no charge node the tick is a **dry run**: it reports what it
would have written and exits 0 — a dev box must not look broken, and systemd must not
restart-loop it.

### systemd

```bash
sudo install -m 644 packages/chargectl/systemd/chargectl.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now chargectl
```

## Usage

```python
from chargectl import (
    read_status,
    probe,
    set_limit,
    set_band,
    optimize_hint,
    Policy,
    decide,
    tick,
    load_policy,
)

status = read_status()  # ChargeStatus, source="sysfs" or "simulated"
cap = probe()  # Capability(supported=..., start_supported=..., incumbent=...)
set_limit(80)  # ControlResult(applied=True/False, reason=...)
set_band(40, 80)  # floor + ceiling where the kernel has both

policy = Policy(enabled=True, floor=40, ceiling=80)
directive = decide(status, cap, policy, current_limit=cap.current_limit)
# Directive(action="set_ceiling", value=80, reason="charge cap is 100% — ...") | None
```

CLI:

```
chargectl status          # level, charging state, voltage, current, ETA, health
chargectl advice          # what to do to maximise longevity
chargectl probe           # cap + floor + incumbent (TLP) on this machine
chargectl limit 80        # cap charge at 80% (needs root)
chargectl limit off       # lift the cap
chargectl policy show     # policy file, band, what it can do here
chargectl daemon --once   # a single enforcement tick
```

## In the app

- `GET /system/battery` — status, capability, advice, policy, daemon state.
- `POST /system/battery/limit?percent=` — one-shot cap.
- `PUT /system/battery/policy?enabled=&floor=&ceiling=` — update the policy file.

## Tests

```
python3 -m pytest packages/chargectl/tests -q
```
