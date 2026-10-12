"""
VM life benchmark: execution cost of the built-in Conway's Game of Life
(`vm run life` / `vmrun life` / POST /vm/run with builtin=life).

What it measures (and why these numbers):
    assemble        — LIFE_ASM source -> machine code: encoder cost + size.
    batch           — headless journey, no keys: spawn -> prompt idle timeout
                      -> seed 240 -> 4 generations -> hlt.  Wall time (mean of
                      --runs samples) + total CPU steps (amortized/gen is
                      total/4 — dominated by the 15000-poll prompt idle
                      timeout, ~70% of all steps, not by the rule sweep).
                      All step counts are deterministic (fixed xorshift seed).
    step_frame      — one paused `s` press: steps from key consumption to
                      gen+1 = key_step tail + step_gen rule sweep (1200 cells
                      x 8 neighbors + copy) + detection tail.  Constant by
                      construction — cell work does not depend on live count.
    first_seed_steps — Enter consumption -> board ready at the prompt, per
                       live count n.  Grows with n: seeding places cells by
                       rejection sampling (rand % 1200 until an empty cell is
                       found), so cost rises as the board fills; 1200 takes
                       the full-board fast path instead.
    reseed_steps    — `r` press consumption -> gen back to 0, per live count.
                       Pure next_rand + seed_board cost (rejection sampling)
                       at the current n — the N-dependent curve without the
                       prompt/clear-screen overhead.

Key presses are measured in two segments (consume / after-consume) so frame
alignment never pollutes the phase being measured.

Usage:
    python scripts/benchmark_vm_life.py                 # table
    python scripts/benchmark_vm_life.py --runs 5        # more wall samples
    python scripts/benchmark_vm_life.py --json out.json

Record + compare:
    python scripts/benchmark_vm_life.py --json /tmp/vmlife.json
    python scripts/benchmark_results.py record --kind vm_life --json-file /tmp/vmlife.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: domain.*

from domain.shell._internal.vm import X86Assembler, X86VirtualSystem  # noqa: E402
from domain.shell._internal.vm_permissions import Role  # noqa: E402
from domain.shell._internal.vm_programs import LIFE_ASM  # noqa: E402

PROMPT_END = 60_000  # steps to reach the prompt on a fresh spawn
DEFAULT_LIVES = 240
BATCH_GENS = 4
SEED_POINTS = (1, 240, 600, 1199, 1200)
STEP_BUDGET = 4_000_000


# ── harness (mirrors tests/core-py/test_vm_life_program.py) ──────────────────

def _spawn() -> X86VirtualSystem:
    vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
    pid = vs.spawn("life", LIFE_ASM)
    if pid is None:
        raise RuntimeError("spawn failed")
    vs._syscall._rbac.assign(pid, Role.USER)
    vs.scheduler.start(vs.cpu)
    vs.scheduler.current.restore_to_cpu(vs.cpu)
    return vs


def _dword(vs: X86VirtualSystem, addr: int) -> int:
    m = vs.cpu._mem
    return m[addr] | (m[addr + 1] << 8) | (m[addr + 2] << 16) | (m[addr + 3] << 24)


def _labels() -> dict[str, int]:
    asm = X86Assembler()
    asm.assemble(LIFE_ASM, org=0x100000)
    return dict(asm._labels)


def _step(vs: X86VirtualSystem, n: int = 1) -> None:
    for _ in range(n):
        if not vs.cpu.step():
            raise RuntimeError("CPU halted unexpectedly")


def _drain(vs: X86VirtualSystem, budget: int = 400_000) -> None:
    """Step until [0x400] is consumed, then settle 2000 steps (echo/state)."""
    limit = vs.cpu._step_count + budget
    while vs.cpu._mem[0x400] != 0 and vs.cpu._step_count < limit:
        if not vs.cpu.step():
            raise RuntimeError("CPU halted unexpectedly")
    settle = vs.cpu._step_count + 2_000
    while vs.cpu._step_count < settle:
        if not vs.cpu.step():
            raise RuntimeError("CPU halted unexpectedly")


def _feed(vs: X86VirtualSystem, keys: str, budget: int = 400_000) -> None:
    for ch in keys:
        vs.cpu.push_key(ch)
        vs.cpu.transfer_key()
        _drain(vs, budget)


def _press_segmented(
    vs: X86VirtualSystem, ch: str, predicate, budget: int = STEP_BUDGET
) -> tuple[int, int]:
    """Press `ch`; return (consume_steps, after_consume_steps).

    Segment 1 ends the moment the program reads the key byte out of [0x400];
    segment 2 runs from there until `predicate` holds.  Segment 2 is therefore
    aligned to the program's key-handler entry — no frame-draw noise.
    """
    vs.cpu.push_key(ch)
    vs.cpu.transfer_key()
    c0 = vs.cpu._step_count
    limit = c0 + budget
    while vs.cpu._mem[0x400] != 0:
        if vs.cpu._step_count >= limit:
            raise RuntimeError("key never consumed")
        if not vs.cpu.step():
            raise RuntimeError("CPU halted unexpectedly")
    consume = vs.cpu._step_count - c0
    a0 = vs.cpu._step_count
    while not predicate():
        if vs.cpu._step_count >= limit:
            raise RuntimeError("predicate not reached")
        if not vs.cpu.step():
            raise RuntimeError("CPU halted unexpectedly")
    return consume, vs.cpu._step_count - a0


# ── benchmarks ───────────────────────────────────────────────────────────────

def bench_assemble() -> dict:
    t0 = time.perf_counter()
    code = X86Assembler().assemble(LIFE_ASM, org=0x100000)
    ms = (time.perf_counter() - t0) * 1000
    return {
        "assemble_ms": round(ms, 2),
        "source_bytes": len(LIFE_ASM),
        "code_bytes": len(code),
    }


def bench_batch(runs: int) -> dict:
    """Spawn -> halt with no keyboard input (the headless CLI/API journey)."""
    walls: list[float] = []
    steps: int | None = None
    halted = False
    for _ in range(runs):
        vs = _spawn()
        t0 = time.perf_counter()
        n = vs.cpu.run(max_steps=STEP_BUDGET)
        walls.append((time.perf_counter() - t0) * 1000)
        if steps is None:
            steps = n
            halted = n < STEP_BUDGET
        elif n != steps:
            print(f"[WARN] batch steps nondeterministic: {n} != {steps}")
    return {
        "batch_wall_ms": round(statistics.mean(walls), 2),
        "batch_wall_ms_min": round(min(walls), 2),
        "batch_steps": steps,
        "batch_halted": halted,
        "batch_steps_per_gen": round(steps / BATCH_GENS) if halted else None,
    }


def bench_seed_curve(labels: dict[str, int]) -> dict:
    """first_seed_steps + reseed_steps across live counts (rejection sampling)."""
    first: dict[str, int] = {}
    reseed: dict[str, int] = {}
    for n in SEED_POINTS:
        vs = _spawn()
        # prompt: type the number (plain feed — not under measurement)
        _step(vs, PROMPT_END)
        _feed(vs, str(n))
        # Enter -> board ready, measured from key consumption
        _, after = _press_segmented(
            vs,
            "\r",
            lambda: _dword(vs, labels["lives"]) == n
            and _dword(vs, labels["livecount"]) == n
            and _dword(vs, labels["gen"]) == 0,
        )
        first[str(n)] = after
        # Setup for the reseed measurement: the program auto-resumed after
        # the prompt, so pause it, then run one generation — reseed then has
        # real state to reset (gen 1 -> 0 marks seed_board's completion).
        _feed(vs, " ")  # pause (running 1 -> 0)
        _feed(vs, "s")  # one generation while paused
        _, after_r = _press_segmented(
            vs, "r", lambda: _dword(vs, labels["gen"]) == 0
        )
        reseed[str(n)] = after_r
    return {"first_seed_steps": first, "reseed_steps": reseed}


def bench_step_frame(labels: dict[str, int]) -> dict:
    """One paused `s` press: key-handler tail + step_gen rule sweep."""
    vs = _spawn()
    _step(vs, PROMPT_END)
    _feed(vs, str(DEFAULT_LIVES))
    _press_segmented(vs, "\r", lambda: _dword(vs, labels["livecount"]) == DEFAULT_LIVES)
    _press_segmented(vs, " ", lambda: _dword(vs, labels["running"]) == 0)
    _, after = _press_segmented(
        vs, "s", lambda: _dword(vs, labels["gen"]) >= 1
    )
    return {"step_frame_steps": after}


# ── entry ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark the life VM builtin")
    ap.add_argument("--runs", type=int, default=3, help="wall-time samples")
    ap.add_argument("--json", metavar="PATH", help="write metrics JSON")
    args = ap.parse_args()

    labels = _labels()
    metrics: dict = {}
    metrics.update(bench_assemble())
    metrics.update(bench_batch(args.runs))
    metrics.update(bench_step_frame(labels))
    metrics.update(bench_seed_curve(labels))

    print("=== VM life benchmark (scripts/benchmark_vm_life.py) ===")
    print(
        f"  assemble         : {metrics['assemble_ms']:>8.2f} ms   "
        f"code {metrics['code_bytes']} B (src {metrics['source_bytes']} B)"
    )
    print(
        f"  batch end-to-end : {metrics['batch_wall_ms']:>8.2f} ms   "
        f"{metrics['batch_steps']} steps halted={metrics['batch_halted']}"
    )
    if metrics.get("batch_steps_per_gen"):
        print(
            f"  amortized/gen    : {metrics['batch_steps_per_gen']:>8} steps "
            f"(batch total / 4 — dominated by the 15000-poll prompt timeout)"
        )
    print(f"  step frame (s)   : {metrics['step_frame_steps']:>8} steps "
          f"(key tail + rule sweep)")
    print("  first seed (Enter -> board ready):")
    for n, s in metrics["first_seed_steps"].items():
        print(f"    n={n:>4}: {s:>8} steps")
    print("  reseed (r press -> gen=0):")
    for n, s in metrics["reseed_steps"].items():
        print(f"    n={n:>4}: {s:>8} steps")

    if args.json:
        payload = {"kind": "vm_life", "metrics": metrics}
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
