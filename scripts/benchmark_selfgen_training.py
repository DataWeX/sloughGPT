#!/usr/bin/env python3
"""
Self-gen flywheel benchmark — identical tiny training runs, flywheel ON vs OFF.

Measures what the self-gen loop costs and what it produces:
  1. overhead_pct   — steps/sec delta (ON vs OFF); generation steals train time.
  2. rounds         — in-training generate->gate rounds executed when ON.
  3. acceptance     — gated writes / round (dedup+quality at the write boundary).
  4. loss           — final loss ON vs OFF (self-training must not rot the model).

Gold Standard (pass/fail):
  - rounds executed:      > 0 when ON (flywheel actually ran)
  - acceptance per round: >= 1 on average (the loop feeds the trainer, else it is theater)
  - final loss ON:        <= max(OFF * 1.5, OFF + 0.25) (no catastrophic regression)
  - overhead:             reported; gate at <= 300% wall-time of OFF (sanity bound)

Usage:
    ./scripts/python scripts/benchmark_selfgen_training.py
    ./scripts/python scripts/benchmark_selfgen_training.py --steps 60 --json-out out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "core-py"))

os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("SLO_OPENBLAS_NUM_THREADS", "1"))
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("SLO_OMP_NUM_THREADS", "1"))

import numpy as np

GOLD = {
    "min_rounds": 1,
    "min_acceptance_per_round": 1.0,
    "loss_ratio_max": 1.5,
    "loss_abs_slack": 0.25,
    "wall_ratio_max": 4.0,
}


def _dataset() -> tuple[str, str]:
    """Resolve the benchmark dataset (tinyshakespeare when present)."""
    from domain.training._internal.cache_tags import get_cache_root

    candidates = [
        Path(__file__).resolve().parents[1] / "data" / "datasets" / "tinyshakespeare" / "input.txt",
        Path(__file__).resolve().parents[1] / "data" / "tinyshakespeare" / "input.txt",
        get_cache_root() / "tinyshakespeare" / "corpus.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return str(p), f"{os.path.getsize(p) / 1024:.1f} KB"
    # synthetic fallback — keep both runs on identical data either way
    tmp = Path(tempfile.mkstemp(prefix="selfgen_bench_", suffix=".txt")[1])
    tmp.write_text("abcdefghij" * 2048, encoding="utf-8")
    return str(tmp), "synthetic"


def _run(data_path: str, flywheel: bool, steps: int, seed: int, corpus_base: Path) -> dict:
    """One identical training run; flywheel toggles only selfgen_interval."""
    from domain.training._internal.train_pipeline import SloughGPTTrainer, TrainerConfig

    np.random.seed(seed)
    cfg = TrainerConfig(
        n_embed=64,
        n_layer=1,
        n_head=4,
        block_size=32,
        batch_size=16,
        epochs=10_000,  # steps are bounded by max_steps
        learning_rate=2e-3,
        warmup_steps=10,
        weight_decay=0.0,
        min_data_quality=0.0,
        max_toxicity_rate=1.0,
        max_steps=steps,
        log_interval=steps + 1,  # quiet
        eval_interval=steps + 1,
        checkpoint_interval=100_000,
        checkpoint_dir=str(corpus_base.parent / ("ckpt_on" if flywheel else "ckpt_off")),
        scheduler_type="constant",
        # flywheel: one round every ~steps/4 so a short run still exercises it
        selfgen_interval=max(1, steps // 4) if flywheel else 0,
        selfgen_samples=4,
        selfgen_max_tokens=48,
        selfgen_min_quality=1.0,
        selfgen_local_base=str(corpus_base),
    )
    trainer = SloughGPTTrainer(data_path=data_path, config=cfg)

    start = time.perf_counter()
    result = trainer.train()
    wall = time.perf_counter() - start

    producer = trainer._selfgen_producer
    stats = producer.stats() if producer is not None else {}
    attempts = (
        stats.get("written", 0)
        + stats.get("rejected_quality", 0)
        + stats.get("rejected_dup", 0)
        + stats.get("rejected_short", 0)
    )
    rounds = int(stats.get("rounds", 0))
    written = int(stats.get("written", 0))

    # loss trajectory: eval the trained model on a held slice for comparability
    try:
        eval_metrics = trainer.evaluate(num_batches=5)
        final_eval = float(eval_metrics["eval_loss"])
    except Exception:
        final_eval = float(result.final_loss or 0.0)

    return {
        "flywheel": flywheel,
        "wall_s": round(wall, 2),
        "steps_per_sec": round(result.global_step / wall, 2) if wall else 0.0,
        "global_step": result.global_step,
        "final_loss": round(final_eval, 4),
        "rounds": rounds,
        "written": written,
        "attempts": attempts,
        "acceptance_per_round": round(written / rounds, 2) if rounds else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=48, help="training steps per run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", default="", help="write results JSON here")
    args = parser.parse_args()

    from domain.training._internal.selfgen import resolve_corpus_path

    data_path, size = _dataset()
    print(f"Dataset: {data_path} ({size})")
    print(f"Steps: {args.steps}  seed: {args.seed}")

    with tempfile.TemporaryDirectory(prefix="selfgen_bench_") as tmp:
        base = Path(tmp) / "feed"
        # sanity: corpus path resolution stays inside the temp base
        assert str(resolve_corpus_path("selfgen", local_base=base)).startswith(tmp)

        print("\n--- run 1/2: flywheel OFF ---")
        off = _run(data_path, flywheel=False, steps=args.steps, seed=args.seed, corpus_base=base)
        print(
            f"  wall={off['wall_s']}s  {off['steps_per_sec']} steps/s  "
            f"final_loss={off['final_loss']}"
        )

        print("\n--- run 2/2: flywheel ON ---")
        on = _run(data_path, flywheel=True, steps=args.steps, seed=args.seed, corpus_base=base)
        print(
            f"  wall={on['wall_s']}s  {on['steps_per_sec']} steps/s  "
            f"final_loss={on['final_loss']}  rounds={on['rounds']}  "
            f"accept/round={on['acceptance_per_round']}"
        )

    # ── verdicts ──────────────────────────────────────────────────────────
    verdicts: list[tuple[str, bool, str]] = []

    verdicts.append(
        (
            "rounds",
            on["rounds"] >= GOLD["min_rounds"],
            f"{on['rounds']} >= {GOLD['min_rounds']}",
        )
    )
    verdicts.append(
        (
            "acceptance",
            on["acceptance_per_round"] >= GOLD["min_acceptance_per_round"],
            f"{on['acceptance_per_round']}/round >= {GOLD['min_acceptance_per_round']}",
        )
    )
    loss_budget = max(
        off["final_loss"] * GOLD["loss_ratio_max"], off["final_loss"] + GOLD["loss_abs_slack"]
    )
    verdicts.append(
        (
            "loss",
            on["final_loss"] <= loss_budget,
            f"{on['final_loss']} <= budget {loss_budget:.4f} (off={off['final_loss']})",
        )
    )
    wall_budget = off["wall_s"] * GOLD["wall_ratio_max"]
    verdicts.append(
        (
            "overhead",
            on["wall_s"] <= wall_budget,
            f"wall {on['wall_s']}s <= {wall_budget:.1f}s ({on['wall_s'] / off['wall_s']:.2f}x off)"
            if off["wall_s"]
            else "n/a",
        )
    )

    overhead_pct = (
        round(100.0 * (on["steps_per_sec"] - off["steps_per_sec"]) / off["steps_per_sec"], 1)
        if off["steps_per_sec"]
        else 0.0
    )
    print("\n" + "=" * 70)
    print("  metric            OFF         ON          verdict")
    print("  ----------------  ----------  ----------  --------")
    rows = [
        ("wall_s", off["wall_s"], on["wall_s"]),
        ("steps/s", off["steps_per_sec"], on["steps_per_sec"]),
        ("final_loss", off["final_loss"], on["final_loss"]),
        ("rounds", 0, on["rounds"]),
        ("accept/round", 0, on["acceptance_per_round"]),
    ]
    for name, v_off, v_on in rows:
        print(f"  {name:<16}  {v_off!s:<10}  {v_on!s:<10}")
    print(f"\n  steps/s delta: {overhead_pct:+.1f}% (negative = generation overhead)")
    print()
    ok = True
    for name, passed, detail in verdicts:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<10} {detail}")
        ok = ok and passed
    print("=" * 70)
    print(f"Verdict: {'PASS' if ok else 'FAIL'}")

    if args.json_out:
        payload = {
            "off": off,
            "on": on,
            "steps_per_sec_delta_pct": overhead_pct,
            "verdicts": [{"name": n, "pass": p, "detail": d} for n, p, d in verdicts],
            "pass": ok,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Results written to {args.json_out}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
