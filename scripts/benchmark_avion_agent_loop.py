#!/usr/bin/env python3
"""Benchmark — avion agent-loop throughput after the tightening pass.

What it measures (steps/sec of ``Agent.run`` with a fake model+backend,
i.e. loop overhead only — no browser, no model inference):

1. **bare loop** — bounded awaits, no transcript, no callbacks;
2. **+ JSONL transcript** — per-step write + flush (durability cost);
3. **+ 3 callbacks** — observer fan-out with failure containment;
4. **+ screenshot every step** — perception capture + base64 per step.
5. **+ JSONL transcript (group-commit)** — same records, flush every 1000
   steps: isolates serialization cost from the per-step syscall.

Every scenario runs the tightened loop: each await is bounded by
``min(step_timeout, remaining deadline)``, which is the cost driver this
benchmark exists to watch.

Usage::

    .venv/bin/python scripts/benchmark_avion_agent_loop.py [--steps 2000]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "avion" / "src"))

from avion.ai import Action, ActionType, Agent, AgentConfig  # noqa: E402


class WaitModel:
    """Answers every decision with a zero-cost WAIT (no backend calls)."""

    model_name = "wait"
    supports_vision = False

    async def predict_action(self, **kwargs):
        return Action(ActionType.WAIT, {"ms": 0})


class BareBackend:
    """Backend with everything the loop touches, all instantaneous."""

    name = "bare"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def screenshot(self, path=None) -> bytes:
        return b"\x89PNG-fake-frame"


def _run_scenario(
    label: str,
    steps: int,
    *,
    transcript: bool,
    callbacks: int,
    shots: bool,
    transcript_flush: int = 1,
) -> None:
    async def scenario() -> None:
        cfg = AgentConfig(
            max_steps=steps,
            action_delay_ms=0,
            no_progress_limit=0,  # identical WAITs every step
            screenshot_on_each_step=shots,
            save_trajectories=transcript,
            transcript_flush_steps=transcript_flush,
            output_dir=tempfile.mkdtemp(prefix="avion_loop_bench_") if transcript else "",
        )
        agent = Agent(model=WaitModel(), config=cfg)
        await agent.start(backend=BareBackend())
        for i in range(callbacks):
            agent.on_step(lambda _s: None)
        start = time.perf_counter()
        result = await agent.run("benchmark")
        elapsed = time.perf_counter() - start
        await agent.stop()
        assert result.steps_taken == steps, f"short run: {result.steps_taken}/{steps}"
        rate = steps / elapsed
        per_step_us = (elapsed / steps) * 1e6
        print(
            f"  {label:<44} {rate:>10,.0f} steps/s   {per_step_us:>8.2f} us/step"
        )

    asyncio.run(scenario())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000, help="steps per scenario")
    args = parser.parse_args()

    print(f"avion agent-loop benchmark — {args.steps:,} steps per scenario")
    _run_scenario("bare loop (bounded awaits only)", args.steps, transcript=False, callbacks=0, shots=False)
    _run_scenario("+ JSONL transcript (write+flush/step)", args.steps, transcript=True, callbacks=0, shots=False)
    _run_scenario("+ 3 on_step callbacks", args.steps, transcript=False, callbacks=3, shots=False)
    _run_scenario("+ screenshot every step", args.steps, transcript=False, callbacks=0, shots=True)
    _run_scenario(
        "+ JSONL transcript (group-commit, flush/1000)",
        args.steps,
        transcript=True,
        callbacks=0,
        shots=False,
        transcript_flush=1000,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
