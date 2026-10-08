"""Benchmark: cave_game voxel render cost on the x86 VM.

Measures the Minecraft-iteration-1 runner (vm_programs.CAVE_GAME_ASM)
driven directly through X86VirtualSystem — no HTTP layer — so regressions
in the assembler, CPU step loop, or the frame's raycast math show up as
steps/frame and fps changes.

Metrics:
  - assemble: source -> machine code (ms, bytes)
  - frame: steps + wall ms per rendered frame (frame counter label polled)
  - fps: frames/sec at saturation (batch render, no input)
  - batch_exit (--full): full headless run to idle-HLT (CLI `vm run` parity)

Usage:
  python scripts/benchmark_vm_cave_game.py [--frames 5] [--json out.json]
  python scripts/benchmark_vm_cave_game.py --full   # + batch-to-halt wall
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "apps" / "api" / "server"))


def _hud(mem, addr: int = 0xB8F00, cells: int = 80) -> str:
    raw = bytes(mem[addr : addr + cells * 2])
    return "".join(chr(raw[i * 2]) if 32 <= raw[i * 2] < 127 else " " for i in range(cells))


def run_bench(frames: int, full: bool) -> dict:
    from domain.shell._internal.vm import X86Assembler, X86VirtualSystem
    from domain.shell._internal.vm_permissions import Role
    from domain.shell._internal.vm_programs import CAVE_GAME_ASM

    t0 = time.perf_counter()
    asm = X86Assembler()
    code = asm.assemble(CAVE_GAME_ASM, org=0x100000)
    assemble_ms = (time.perf_counter() - t0) * 1000.0
    labels = asm._labels
    frame_addr = labels["frame"]

    vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
    pid = vs.spawn("cave_game", CAVE_GAME_ASM)
    assert pid is not None, "spawn failed"
    vs._syscall._rbac.assign(pid, Role.USER)
    vs.scheduler.start(vs.cpu)
    vs.scheduler.current.restore_to_cpu(vs.cpu)

    # frames: run until the in-program frame counter advances, K times
    frame_steps: list[int] = []
    frame_ms: list[float] = []
    for _ in range(frames):
        start_steps = vs.cpu._step_count
        t0 = time.perf_counter()
        guard = 0
        seen = int.from_bytes(bytes(vs.cpu._mem[frame_addr : frame_addr + 4]), "little")
        while guard < 4_000_000:
            vs.cpu.run(max_steps=vs.cpu._step_count + 2_000)
            guard = vs.cpu._step_count - start_steps
            now = int.from_bytes(bytes(vs.cpu._mem[frame_addr : frame_addr + 4]), "little")
            if now != seen:
                break
        frame_ms.append((time.perf_counter() - t0) * 1000.0)
        frame_steps.append(vs.cpu._step_count - start_steps)

    hud = _hud(vs.cpu._mem)
    assert "CRAFT" in hud, f"HUD missing after {frames} frames: {hud!r}"

    result = {
        "assemble_ms": round(assemble_ms, 2),
        "code_bytes": len(code),
        "frames": frames,
        "steps_per_frame": frame_steps,
        "ms_per_frame": [round(m, 1) for m in frame_ms],
        "ms_per_frame_mean": round(sum(frame_ms) / len(frame_ms), 1),
        "fps": round(1000.0 / (sum(frame_ms) / len(frame_ms)), 2),
        "hud_ok": True,
    }

    if full:
        # CLI parity: bare VMEngine, unlimited run, idle-HLT terminates it
        from domain.shell._internal.vm_engine import VMEngine

        eng = VMEngine(memory_size=0x400000)
        eng.load_source(CAVE_GAME_ASM, org=0x1000)
        t0 = time.perf_counter()
        trace = eng.run()
        result["batch_exit_reason"] = trace.exit_reason
        result["batch_exit_s"] = round(time.perf_counter() - t0, 1)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="cave_game render benchmark")
    parser.add_argument("--frames", type=int, default=5, help="frames to time")
    parser.add_argument("--full", action="store_true", help="also run batch-to-HLT")
    parser.add_argument("--json", type=Path, default=None, help="write results json")
    args = parser.parse_args()

    result = run_bench(args.frames, args.full)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
