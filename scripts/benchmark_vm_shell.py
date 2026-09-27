"""Benchmark: own-VM shell REPL kernel throughput and command latency.

Measures the X86VirtualSystem shell kernel (vm_builtins) driven directly —
no HTTP layer — so regressions in assembly roundtrips, syscall dispatch, or
the FlatFS file path show up as latency/throughput changes.

Metrics:
  - startup: kernel boot to first prompt (steps + ms)
  - roundtrip: keystroke feed -> command output complete, per command class
    (help / ls / cat / write / cp / echo / uname)
  - throughput: commands per second at saturation (batched feed)

Usage:
  .venv/bin/python scripts/benchmark_vm_shell.py [--iterations 50] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "apps" / "api" / "server"))


def _make_shell():
    from vm_builtins import get_builtin

    from domain.shell._internal.vm import X86VirtualSystem
    from domain.shell._internal.vm_permissions import Role

    vs = X86VirtualSystem(memory_size=0x100000)
    vs._fs.write("bench", b"x" * 64)
    pid = vs.spawn("bench_user", get_builtin("shell"))
    if pid is None:
        raise RuntimeError("failed to spawn shell")
    vs._syscall._rbac.assign(pid, Role.USER)
    vs.scheduler.start(vs.cpu)
    vs.scheduler.current.restore_to_cpu(vs.cpu)

    written: list[bytes] = []
    original = vs._syscall._sys_write

    def capture(fd, addr, count):
        if fd in (1, 2):
            written.append(bytes(vs.cpu._read8(addr + i) for i in range(count)))
            return count
        return original(fd, addr, count)

    vs._syscall._sys_write = capture
    return vs, written


def _feed(vs, text: str) -> None:
    for ch in text:
        vs.cpu.push_key(ch)


def _run_until_prompt(vs, written: list[bytes], max_steps: int = 200_000) -> float:
    """Step until the captured output ends with a fresh prompt. Returns steps used."""
    return _run_until_prompts(vs, written, prompts=1, max_steps=max_steps)


def _run_until_prompts(vs, written: list[bytes], prompts: int, max_steps: int = 200_000) -> float:
    """Step until `prompts` prompt markers have appeared in output. Returns steps used."""
    target = b"\nsloughvm> "
    start_len = sum(len(b) for b in written)
    last_len = start_len
    steps = 0
    while steps < max_steps:
        vs.cpu.transfer_key()
        if not vs.cpu.step():
            break
        steps += 1
        total = sum(len(b) for b in written)
        if total != last_len:
            last_len = total
            joined = b"".join(written)
            if joined.count(target) >= prompts:
                break
    return steps


def _roundtrip(vs, written: list[bytes], cmd: str, iterations: int) -> list[float]:
    samples: list[float] = []
    for _ in range(iterations):
        written.clear()
        t0 = time.perf_counter()
        _feed(vs, cmd)
        _run_until_prompt(vs, written)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    results: dict[str, object] = {}

    # Boot to first prompt
    vs, written = _make_shell()
    t0 = time.perf_counter()
    boot_steps = _run_until_prompt(vs, written)
    results["boot_ms"] = (time.perf_counter() - t0) * 1000.0
    results["boot_steps"] = boot_steps

    commands = {
        "help": "help\n",
        "ls": "ls\n",
        "cat": "cat bench\n",
        "echo": "echo hello bench\n",
        "write": "write bench2 hello-world\n",
        "cp": "cp bench bench3\n",
        "uname": "uname\n",
    }
    for name, cmd in commands.items():
        samples = _roundtrip(vs, written, cmd, args.iterations)
        results[f"{name}_ms_p50"] = statistics.median(samples)
        results[f"{name}_ms_p95"] = sorted(samples)[max(0, int(len(samples) * 0.95) - 1)]
        results[f"{name}_ms_mean"] = statistics.fmean(samples)

    # Saturation throughput: N commands back-to-back, feed all, count completions
    n = 100
    written.clear()
    t0 = time.perf_counter()
    for i in range(n):
        _feed(vs, f"echo tick{i}\n")
    _run_until_prompts(vs, written, prompts=n, max_steps=4_000_000)
    elapsed = time.perf_counter() - t0
    results["throughput_cmds_per_s"] = n / elapsed
    results["throughput_seconds"] = elapsed
    results["throughput_prompts_seen"] = b"".join(written).count(b"\nsloughvm> ")

    print("── vm shell kernel benchmark ──")
    print(f"  boot:            {results['boot_ms']:.1f} ms ({boot_steps} steps)")
    for name in commands:
        print(
            f"  {name:<8} p50={results[f'{name}_ms_p50']:7.2f} ms"
            f"  p95={results[f'{name}_ms_p95']:7.2f} ms"
            f"  mean={results[f'{name}_ms_mean']:7.2f} ms"
        )
    print(f"  throughput:      {results['throughput_cmds_per_s']:.1f} cmds/s ({n} batched)")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2))
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
