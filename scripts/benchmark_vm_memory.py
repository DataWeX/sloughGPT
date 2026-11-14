#!/usr/bin/env python
"""Benchmark: x86 VM memory-size scaling — current vs 2× (construct/throughput/footprint).

The VM's only resource knob is ``memory_size`` — it is a single-CPU
interpreter, there are no vCPUs to double. Doubling it must show up in
exactly two places, and NOT in a third:

  construct ms   ``bytearray(memory_size)`` zeroing at construction — O(size),
                 expect LINEAR (2× size ≈ 2× construct cost)
  run ms         interpreter throughput (ALU + stack/memory workloads) — the
                 dispatch loop does not care how big ``_mem`` is, expect FLAT.
                 A slope here would be a surprise worth its own card.
  footprint      tracemalloc live bytes for one instance — expect the delta
                 between 2× and 1× to equal the added buffer size (2× config
                 = 2× RAM; peak would double-count bytearray's transient
                 mid-allocation buffer, so live/current is measured)

Sweeps (0.5×/1×/2×/4× of each layer's default):

  X86CPU            default 1 MiB  (API ``POST /vm/run`` memory_size default)
  X86VirtualSystem  default 4 MiB  (engine default; API max 16 MiB)

INFORMATIONAL: timing never gates (host variance). The structural checks do —
identical step counts across sizes (same program = same work), footprint
linearity, spawn success at every size.

Run:
    PYTHONNOUSERSITE=1 ./scripts/python scripts/benchmark_vm_memory.py \\
        [--repeat 20] [--construct-repeat 5]

History is appended to ``scripts/benchmark_vm_memory.json``
(``benchmark_results.py record --kind`` has no VM category — same call as the
files-import and downcraft compression benchmarks).
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domain.shell._internal.vm import (  # noqa: E402
    X86CPU,
    X86Assembler,
    X86VirtualSystem,
)

HISTORY = ROOT / "scripts" / "benchmark_vm_memory.json"
MiB = 1024 * 1024

# Workloads — verified byte-exact and step-deterministic on both the CPU and
# virtual-system paths (2001 / 1001 steps at every memory size).
#
# NOT the `times`-based forms from tests/core-py/test_vm_benchmark.py: that
# suite is assertion-free timing and hides a real assembler defect —
# `times N <insn-with-operands>` emits one 0x00 byte per repetition instead
# of code (nop works; mov/add/sub/push silently become garbage). Label loops
# assemble correctly, so the benchmark uses those.
#
# ALU: 2001 dispatch-bound instructions (dec+jnz loop; no memory traffic
# beyond instruction fetch).
ALU_SOURCE = (
    "[BITS 32]\n[ORG 0x100000]\n"
    "mov ecx, 1000\n.loop:\ndec ecx\njnz .loop\nhlt"
)
# MEM: 1001 instructions with balanced push/pop pairs — real stack writes at
# the top of memory (250 iterations × push+pop+dec+jnz).
MEM_SOURCE = (
    "[BITS 32]\n[ORG 0x100000]\n"
    "mov ecx, 250\n.loop:\npush eax\npop ebx\ndec ecx\njnz .loop\nhlt"
)

MAX_STEPS = 100_000

# Sweeps: 0.5×/1×/2×/4× of each layer's default (default at index 1, 2× at 2).
CPU_SIZES = [MiB // 2, MiB, 2 * MiB, 4 * MiB]
SYS_SIZES = [2 * MiB, 4 * MiB, 8 * MiB, 16 * MiB]
CPU_TAGS = {MiB // 2: "0.5x", MiB: "1x default", 2 * MiB: "2x", 4 * MiB: "4x"}
SYS_TAGS = {
    2 * MiB: "0.5x",
    4 * MiB: "1x default",
    8 * MiB: "2x",
    16 * MiB: "4x",
}


def _fmt_size(size: int, tags: dict[int, str]) -> str:
    return f"{size / MiB:g} MiB ({tags[size]})"


def timed_construct(factory, repeat: int) -> float:
    """Median construction time in ms over ``repeat`` fresh instances."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter_ns()
        obj = factory()
        times.append((time.perf_counter_ns() - t0) / 1e6)
        del obj
    return statistics.median(times)


def live_footprint(factory) -> int:
    """Tracemalloc *live* bytes for one construction (timing pass separate).

    Uses ``current``, not ``peak``: ``bytearray(n)`` transiently holds two
    buffers mid-allocation, so peak reports ~2×n while live = n + C — the
    number that is actually resident after construction.
    """
    gc.collect()
    tracemalloc.start()
    try:
        obj = factory()
        current, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        del obj
    return current


def sweep_cpu(iterations: int, crepeat: int) -> list[dict]:
    asm = X86Assembler()
    binaries = {
        "alu": asm.assemble(ALU_SOURCE),
        "mem": asm.assemble(MEM_SOURCE),
    }
    rows = []
    for size in CPU_SIZES:
        row = {
            "size": size,
            "construct_ms": round(
                timed_construct(lambda: X86CPU(memory_size=size), crepeat), 4
            ),
            "footprint_b": live_footprint(lambda: X86CPU(memory_size=size)),
        }
        for name, binary in binaries.items():
            samples, steps = [], 0
            for _ in range(iterations):
                cpu = X86CPU(memory_size=size)
                cpu.load(binary)
                t0 = time.perf_counter_ns()
                steps = cpu.run(max_steps=MAX_STEPS)
                samples.append((time.perf_counter_ns() - t0) / 1e6)
                del cpu
            row[f"{name}_ms"] = round(statistics.median(samples), 4)
            row[f"{name}_steps"] = steps
        rows.append(row)
    return rows


def sweep_system(iterations: int, crepeat: int) -> list[dict]:
    rows = []
    for size in SYS_SIZES:
        row = {
            "size": size,
            "construct_ms": round(
                timed_construct(lambda: X86VirtualSystem(memory_size=size), crepeat),
                4,
            ),
            "footprint_b": live_footprint(
                lambda: X86VirtualSystem(memory_size=size)
            ),
            "spawn_ok": True,
        }
        spawn_ms, run_ms, steps = [], [], 0
        for _ in range(iterations):
            vs = X86VirtualSystem(memory_size=size)
            t0 = time.perf_counter_ns()
            pid = vs.spawn("bench", ALU_SOURCE)
            spawn_ms.append((time.perf_counter_ns() - t0) / 1e6)
            if pid is None:
                row["spawn_ok"] = False
                break
            vs.scheduler.start(vs.cpu)
            current = vs.scheduler.current
            if current is None:
                row["spawn_ok"] = False
                break
            current.restore_to_cpu(vs.cpu)
            t0 = time.perf_counter_ns()
            steps = vs.cpu.run(max_steps=MAX_STEPS)
            run_ms.append((time.perf_counter_ns() - t0) / 1e6)
            del vs
        row["spawn_ms"] = round(statistics.median(spawn_ms), 4) if spawn_ms else None
        row["run_ms"] = round(statistics.median(run_ms), 4) if run_ms else None
        row["steps"] = steps
        rows.append(row)
    return rows


def print_table(title: str, rows: list[dict], tags: dict[int, str]) -> None:
    print(f"\n{title}")
    if "alu_ms" in rows[0]:
        header = (
            f"{'size':<18} {'construct':>10} {'alu ms/run':>11} "
            f"{'mem ms/run':>11} {'footprint':>10}"
        )
        print(header)
        print("-" * len(header))
        for r in rows:
            fp = f"{r['footprint_b'] / MiB:.2f} MiB"
            print(
                f"{_fmt_size(r['size'], tags):<18} {r['construct_ms']:>10.3f} "
                f"{r['alu_ms']:>11.3f} {r['mem_ms']:>11.3f} {fp:>10}"
            )
    else:
        header = (
            f"{'size':<18} {'construct':>10} {'spawn':>8} {'run ms':>8} "
            f"{'steps':>7} {'footprint':>10}"
        )
        print(header)
        print("-" * len(header))
        for r in rows:
            fp = f"{r['footprint_b'] / MiB:.2f} MiB"
            spawn = "-" if r["spawn_ms"] is None else f"{r['spawn_ms']:.3f}"
            run = "-" if r["run_ms"] is None else f"{r['run_ms']:.3f}"
            print(
                f"{_fmt_size(r['size'], tags):<18} {r['construct_ms']:>10.3f} "
                f"{spawn:>8} {run:>8} {r['steps']:>7} {fp:>10}"
            )


def delta_block(rows: list[dict], keys: list[str]) -> str:
    """current (index 1) vs 2× (index 2): percent deltas, informational."""
    base, two = rows[1], rows[2]
    parts = []
    for key in keys:
        b, t = base[key], two[key]
        if b in (None, 0) or t is None:
            parts.append(f"{key} -")
        else:
            parts.append(f"{key} {100 * (t / b - 1):+.1f}%")
    fp_delta = (two["footprint_b"] - base["footprint_b"]) / MiB
    parts.append(f"footprint +{fp_delta:.2f} MiB (expected +{base['size'] / MiB:g})")
    return "current vs 2×: " + " | ".join(parts)


def check_linearity(
    rows: list[dict], name: str, checks: list[tuple[str, bool, str]]
) -> None:
    """fp(2×) − fp(1×) must equal the added buffer within 10%."""
    base, two = rows[1], rows[2]
    delta = two["footprint_b"] - base["footprint_b"]
    expect = base["size"]
    ok = abs(delta - expect) <= 0.10 * expect
    checks.append(
        (
            f"{name} footprint doubles with memory",
            ok,
            f"delta={delta / MiB:.2f} MiB expected={expect / MiB:.2f} MiB",
        )
    )


def git_short() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeat", type=int, default=20, help="run iterations (median)")
    ap.add_argument(
        "--construct-repeat", type=int, default=5, help="construct iterations (median)"
    )
    args = ap.parse_args()

    checks: list[tuple[str, bool, str]] = []

    print(
        "vm memory benchmark — "
        f"{args.repeat} run iters (median), {args.construct_repeat} construct iters"
    )

    cpu_rows = sweep_cpu(args.repeat, args.construct_repeat)
    print_table("X86CPU (default 1 MiB)", cpu_rows, CPU_TAGS)
    print("  " + delta_block(cpu_rows, ["construct_ms", "alu_ms", "mem_ms"]))

    sys_rows = sweep_system(args.repeat, args.construct_repeat)
    print_table("X86VirtualSystem (default 4 MiB)", sys_rows, SYS_TAGS)
    print("  " + delta_block(sys_rows, ["construct_ms", "run_ms"]))

    # ── Structural checks (deterministic — timing never gates) ─────────────
    for key, label in (("alu_steps", "cpu alu"), ("mem_steps", "cpu mem")):
        vals = {r[key] for r in cpu_rows}
        checks.append(
            (
                f"{label} steps identical across sizes",
                len(vals) == 1,
                f"steps={sorted(vals)}",
            )
        )
    sys_steps = {r["steps"] for r in sys_rows}
    checks.append(
        (
            "system steps identical across sizes",
            len(sys_steps) == 1,
            f"steps={sorted(sys_steps)}",
        )
    )
    checks.append(
        (
            "system spawn ok at every size",
            all(r["spawn_ok"] for r in sys_rows),
            f"spawn_ok={[r['spawn_ok'] for r in sys_rows]}",
        )
    )
    check_linearity(cpu_rows, "cpu", checks)
    check_linearity(sys_rows, "system", checks)

    failed = [c for c in checks if not c[1]]
    print(f"\nchecks: {len(checks) - len(failed)}/{len(checks)} ok")
    for name, _, detail in failed:
        print(f"  FAIL {name}: {detail}")

    record = {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git": git_short(),
        "repeat": args.repeat,
        "construct_repeat": args.construct_repeat,
        "cpu": cpu_rows,
        "system": sys_rows,
        "delta": {
            "cpu": delta_block(cpu_rows, ["construct_ms", "alu_ms", "mem_ms"]),
            "system": delta_block(sys_rows, ["construct_ms", "run_ms"]),
        },
        "checks_ok": not failed,
    }
    history = []
    if HISTORY.exists():
        try:
            history = json.loads(HISTORY.read_text())
        except (OSError, ValueError):
            history = []
    history.append(record)
    HISTORY.write_text(json.dumps(history[-100:], indent=2) + "\n")
    print(f"history: {HISTORY.relative_to(ROOT)} (record #{len(history)})")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
