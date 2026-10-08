"""
Device-node benchmark: /dev dispatch overhead and boot registration cost.

Covers the unified AI device node (/dev/ai) and its VFS projection — the seam
this change added — plus the registry it joined:

1. create_default_devices()   boot-time registry construction (AI + hardware)
2. vfs_set_devices            VFS /dev mount rebuild from the registry
3. vfs_listdir_dev            `ls /dev`
4. ioctl_info                 per-request dispatch overhead (kernel ioctl seam)
5. vfs_read_dev_ai            `cat /dev/ai`  (entry -> ioctl -> unwrap)
6. vfs_write_dev_ai           `echo <prompt> > /dev/ai` (auto op dispatch)
7. vfs_write_embed_op         `echo "embed: <text>" > /dev/ai`

Every path is offline: generate/embed take injected functions, `info` is a
static card, and nothing here calls the API.

Usage:
    python scripts/benchmark_devices.py           # run and compare to baseline
    python scripts/benchmark_devices.py --update  # update baseline
    python scripts/benchmark_devices.py --ci      # exit non-zero if regression >20%

The gate is load-aware: a canary workload (pure Python, identical in every
worktree) anchors machine speed, and the recorded ``loadavg`` decides whether
the comparison is meaningful at all. A shared build box at load 11 makes every
metric ~2.5x slower regardless of code, so comparing those numbers to an idle
baseline would report phantom regressions.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASELINE_FILE = REPO_ROOT / "data" / "benchmark_devices_baseline.json"
REGRESSION_THRESHOLD_PCT = 20.0
#: loadavg ratio beyond which a baseline comparison is not meaningful
LOAD_MISMATCH_RATIO = 1.5

# (name, iterations, batches) — heavier setups get fewer iterations.
# `canary` is last on purpose: it measures the machine, not the code.
CASES = (
    ("create_default_devices", 100, 3),
    ("vfs_set_devices", 500, 5),
    ("vfs_listdir_dev", 50_000, 5),
    ("ioctl_info", 50_000, 5),
    ("vfs_read_dev_ai", 50_000, 5),
    ("vfs_write_dev_ai", 20_000, 5),
    ("vfs_write_embed_op", 20_000, 5),
    ("canary", 20_000, 5),
)


def _loadavg() -> float:
    try:
        return round(float(Path("/proc/loadavg").read_text().split()[0]), 2)
    except (OSError, ValueError, IndexError):  # pragma: no cover - non-Linux
        return 0.0


def _measure(fn, iterations: int, batches: int) -> dict:
    """Fastest batch wins: min-of-N keeps scheduler noise out of micro timings."""
    best_per_op = None
    for _ in range(batches):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            fn()
        per_op = (time.perf_counter_ns() - start) / iterations
        if best_per_op is None or per_op < best_per_op:
            best_per_op = per_op
    per_op_us = best_per_op / 1000.0
    return {
        "per_op_us": round(per_op_us, 4),
        "ops_per_s": round(1_000_000 / per_op_us, 1) if per_op_us else 0.0,
        "iterations": iterations,
        "batches": batches,
    }


def build_targets() -> dict:
    """Wire up the objects under test — no network, injected AI functions."""
    from domain.shell._internal.addons.filesystem import VFS
    from domain.shell._internal.devices import AIDeviceDriver, create_default_devices
    from domain.shell._internal.kernel_devices import DeviceManager

    mgr = create_default_devices()
    vfs = VFS()
    vfs.set_devices(mgr)

    live = DeviceManager()
    live.register(
        AIDeviceDriver(
            "ai",
            generate_fn=lambda prompt: prompt,
            embed_fn=lambda text: [0.0, 0.1, 0.2],
        )
    )
    live.alias("llm", "ai")
    live.alias("embedding", "ai")
    live_vfs = VFS()
    live_vfs.set_devices(live)

    ai = mgr.get("ai")
    table = {f"key{i}": i for i in range(32)}

    def canary() -> int:
        # Pure Python — identical in every worktree, so it tracks machine speed.
        total = 0
        for i in range(32):
            total += table[f"key{i}"]
        return total

    return {
        "create_default_devices": create_default_devices,
        "vfs_set_devices": lambda: vfs.set_devices(mgr),
        "vfs_listdir_dev": lambda: vfs.listdir("/dev"),
        "ioctl_info": lambda: ai.ioctl("info"),
        "vfs_read_dev_ai": lambda: live_vfs.read("/dev/ai"),
        "vfs_write_dev_ai": lambda: live_vfs.write("/dev/ai", "hello"),
        "vfs_write_embed_op": lambda: live_vfs.write("/dev/ai", "embed: hi"),
        "canary": canary,
    }


def run() -> dict:
    targets = build_targets()
    return {name: _measure(targets[name], iters, batches) for name, iters, batches in CASES}


def load_baseline() -> dict:
    if BASELINE_FILE.exists():
        with open(BASELINE_FILE) as f:
            return json.load(f)
    return {}


def save_baseline(data: dict) -> None:
    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[BASELINE] Saved to {BASELINE_FILE}")


def main() -> None:
    args = set(sys.argv[1:])
    update = "--update" in args
    ci_mode = "--ci" in args

    print("[BENCH] Measuring /dev dispatch overhead (offline, injected AI fns)...")
    result = run()
    result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    result["python"] = sys.version.split()[0]
    result["loadavg"] = _loadavg()

    print(f"  {'metric':<24} {'us/op':>10} {'ops/s':>12}")
    for name, metrics in result.items():
        if not isinstance(metrics, dict):
            continue
        print(f"  {name:<24} {metrics['per_op_us']:>10.3f} {metrics['ops_per_s']:>12,.0f}")
    print(f"  loadavg: {result['loadavg']}")

    if update:
        save_baseline(result)
        print("[BENCH] Baseline updated")
        return

    baseline = load_baseline()
    if not baseline:
        print("[BENCH] No baseline found. Run with --update to create one.")
        return

    # A shared box at load 11 slows every metric ~2.5x — comparing that to an
    # idle baseline reports phantom regressions, so refuse the comparison
    # instead of failing the gate on machine noise.
    base_load = float(baseline.get("loadavg") or 1.0)
    now_load = float(result.get("loadavg") or 0.0) or base_load
    load_ratio = max(now_load, 1.0) / max(base_load, 1.0)
    canary_base = (
        baseline.get("canary", {}).get("per_op_us")
        if isinstance(baseline.get("canary"), dict)
        else None
    )
    canary_now = result["canary"]["per_op_us"]

    print(f"  {'metric':<24} {'baseline':>10} {'now':>10} {'delta':>9}")
    regressions = []
    for name, metrics in result.items():
        if not isinstance(metrics, dict) or name not in baseline:
            continue
        base_us = baseline[name]["per_op_us"]
        now_us = metrics["per_op_us"]
        if not base_us:
            continue
        change = ((now_us - base_us) / base_us) * 100
        print(f"  {name:<24} {base_us:>10.3f} {now_us:>10.3f} {change:>+8.1f}%")
        if change > REGRESSION_THRESHOLD_PCT:
            regressions.append((name, change))

    if canary_base:
        canary_change = ((canary_now - canary_base) / canary_base) * 100
        print(
            f"  machine speed (canary): baseline {canary_base:.3f} us, "
            f"now {canary_now:.3f} us ({canary_change:+.1f}%)"
        )

    if load_ratio > LOAD_MISMATCH_RATIO or load_ratio < 1 / LOAD_MISMATCH_RATIO:
        # Raw deltas are printed above for information; never gate on them.
        print(
            f"[SKIP] loadavg {now_load} vs baseline {base_load} (x{load_ratio:.1f}) — "
            "timings are not comparable; re-run on an idle machine to gate."
        )
        return

    if regressions:
        for name, change in regressions:
            print(
                f"[REGRESSION] {name}: {change:.1f}% slower than baseline (>{REGRESSION_THRESHOLD_PCT:.0f}%)"
            )
        if ci_mode:
            sys.exit(1)
    else:
        print(f"[OK] Within acceptable range (+-{REGRESSION_THRESHOLD_PCT:.0f}%)")


if __name__ == "__main__":
    main()
