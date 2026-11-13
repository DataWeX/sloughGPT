#!/usr/bin/env python3
"""Track B long-run pretraining supervisor (ROADMAP goal 18).

Unattended from-scratch run over ``SloughGPTTrainer`` (the one loop):

- **resume-latest** — every attempt starts with ``resume=True`` when a
  checkpoint exists, so a crash continues instead of restarting
- **durable metrics** — every ``on_progress`` dict is appended to
  ``<checkpoint_dir>/metrics.jsonl`` via ``LongRunRecorder`` (tail-able after
  a crash)
- **bench hooks** — optional periodic eval appended as ``{"kind": "eval"}``
  rows; ``--json`` emits a ``benchmark_results.py record --kind training``
  compatible summary
- **restart budget** — exceptions are logged and retried up to
  ``--max-restarts`` with a short backoff (months-long jobs die often)

No fourth training loop: delegates to ``SloughGPTTrainer.train``.

Run:
    python scripts/train_longrun.py --smoke
    python scripts/train_longrun.py --data datasets/... --max-steps 100000
    python scripts/train_longrun.py ... --json /tmp/longrun.json
    python scripts/benchmark_results.py record --kind training --json-file /tmp/longrun.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "core-py"))

from domain.training._internal.long_run import LongRunRecorder
from domain.training._internal.train_pipeline import CheckpointManager, SloughGPTTrainer


def _build_trainer(args: argparse.Namespace) -> SloughGPTTrainer:
    return SloughGPTTrainer(
        data_path=args.data,
        n_embed=args.n_embed,
        n_layer=args.n_layer,
        n_head=args.n_head,
        block_size=args.block_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_steps=args.max_steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        device="cpu",
        soul_name=args.soul_name,
    )


def _smoke_corpus(path: Path) -> None:
    cycle = "abcdefghijkl"
    path.write_text("".join(cycle[i % len(cycle)] for i in range(2201)), encoding="ascii")


def run_supervisor(args: argparse.Namespace) -> dict:
    workdir = Path(args.checkpoint_dir)
    workdir.mkdir(parents=True, exist_ok=True)
    metrics_path = workdir / "metrics.jsonl"
    recorder = LongRunRecorder(metrics_path)
    ckpt_mgr = CheckpointManager(args.checkpoint_dir)

    t0 = time.time()
    attempt = 0
    last_error: str | None = None
    result: dict = {}
    evals = 0

    while attempt <= args.max_restarts:
        attempt += 1
        has_ckpt = ckpt_mgr.latest_valid_path() is not None
        resume = has_ckpt or args.resume
        trainer = _build_trainer(args)
        start_step = trainer.global_step

        def on_progress(info: dict, _rec=recorder, _attempt=attempt) -> None:
            row = dict(info)
            row["attempt"] = _attempt
            _rec.record(**row)
            loss = row.get("train_loss")
            step = row.get("global_step")
            if loss is not None and step is not None and args.bench_every > 0:
                if step % args.bench_every == 0:
                    try:
                        ev = trainer.evaluate()
                        _rec.record(
                            kind="eval",
                            step=int(step),
                            eval_loss=float(ev["eval_loss"]),
                            eval_ppl=float(ev["eval_ppl"]),
                            attempt=_attempt,
                        )
                    except Exception as exc:  # bench hook must not kill the run
                        _rec.record(kind="eval_error", step=int(step), error=str(exc))

        print(
            f"[longrun] attempt={attempt} resume={resume} "
            f"start_step={start_step} max_steps={args.max_steps} "
            f"ckpt={args.checkpoint_dir}",
            flush=True,
        )
        try:
            raw = trainer.train(resume=resume, on_progress=on_progress)
            result = {
                "success": bool(raw.get("success", True)),
                "global_step": int(raw.get("global_step", 0)),
                "final_loss": raw.get("final_loss"),
                "best_eval_loss": raw.get("best_eval_loss"),
                "checkpoint": raw.get("checkpoint_name") or raw.get("model_path"),
            }
            last_error = None
            # Done when budget reached (or trainer finished without max_steps)
            if args.max_steps is None or result["global_step"] >= args.max_steps:
                break
            # Shortfall without exception: treat as another resume segment
            print(
                f"[longrun] segment ended at step {result['global_step']}"
                f" < {args.max_steps}, resuming…",
                flush=True,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"[longrun] attempt {attempt} failed: {last_error}", flush=True)
            traceback.print_exc()
            if attempt > args.max_restarts:
                break
            time.sleep(min(2.0 * attempt, args.backoff_max_s))

    elapsed = time.time() - t0
    rows = recorder.read_all()
    train_losses = [
        r["train_loss"]
        for r in rows
        if r.get("train_loss") is not None and r.get("kind") != "eval"
    ]
    eval_rows = [r for r in rows if r.get("kind") == "eval"]
    evals = len(eval_rows)
    initial_loss = train_losses[0] if train_losses else None
    final_loss = train_losses[-1] if train_losses else result.get("final_loss")
    steps = int(result.get("global_step") or 0)

    summary = {
        "config": "longrun",
        "attempts": attempt,
        "resumed": bool(args.resume or rows),
        "total_steps": steps,
        "max_steps": args.max_steps,
        "elapsed_s": round(elapsed, 2),
        "steps_per_sec": round(steps / max(elapsed, 0.001), 2),
        "initial_loss": round(float(initial_loss), 4) if initial_loss is not None else None,
        "final_loss": round(float(final_loss), 4) if final_loss is not None else None,
        "metric_rows": len(rows),
        "eval_hooks": evals,
        "metrics_path": str(metrics_path),
        "checkpoint_dir": args.checkpoint_dir,
        "last_error": last_error,
        "success": last_error is None and steps > 0,
    }

    print(f"\n[longrun] summary: {json.dumps(summary, indent=2)}", flush=True)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        # benchmark_results.py --kind training expects a list of metric dicts
        # (same shape as benchmark_slonet_training.py) — wrap as one record.
        payload = [
            {
                "config": summary["config"],
                "params": 0,
                "params_readable": "n/a",
                "epochs": args.epochs,
                "total_steps": summary["total_steps"],
                "elapsed_s": summary["elapsed_s"],
                "steps_per_sec": summary["steps_per_sec"],
                "initial_loss": summary["initial_loss"],
                "final_loss": summary["final_loss"],
                "convergence_ratio": (
                    round(summary["initial_loss"] / max(summary["final_loss"], 1e-8), 2)
                    if summary["initial_loss"] and summary["final_loss"]
                    else 0.0
                ),
                "gate_converged": bool(summary["success"]),
                "perplexity": None,
                "peak_memory_mb": 0.0,
                "longrun_attempts": summary["attempts"],
                "longrun_eval_hooks": summary["eval_hooks"],
                "metric_rows": summary["metric_rows"],
            }
        ]
        out.write_text(json.dumps(payload, indent=2))
        print(f"[longrun] wrote {out}", flush=True)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--data", default=None, help="Training corpus path")
    parser.add_argument("--checkpoint-dir", default="models/longrun")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-embed", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--soul-name", default="sloughgpt-longrun")
    parser.add_argument("--max-restarts", type=int, default=5, help="Crash retries")
    parser.add_argument("--backoff-max-s", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true", help="Force resume=True")
    parser.add_argument(
        "--bench-every",
        type=int,
        default=0,
        help="Every N steps run evaluate() and append kind=eval to metrics.jsonl (0=off)",
    )
    parser.add_argument("--json", default=None, help="Write training-kind results JSON")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Hermetic tiny run (generated cycle corpus, 24 steps, tmp checkpoints)",
    )
    args = parser.parse_args()

    if args.smoke:
        base = Path("/tmp/longrun-smoke")
        base.mkdir(parents=True, exist_ok=True)
        corpus = base / "corpus.txt"
        _smoke_corpus(corpus)
        # Smoke supplies hermetic defaults; leave explicit CLI overrides alone
        # (e.g. --smoke --max-steps 30 continues an existing smoke checkpoint).
        if args.data is None:
            args.data = str(corpus)
        if args.checkpoint_dir == "models/longrun":
            args.checkpoint_dir = str(base / "ckpts")
        if args.max_steps == 1000:
            args.max_steps = 24
        if args.n_embed == 64:
            args.n_embed = 32
        if args.n_layer == 2:
            args.n_layer = 1
        if args.n_head == 4:
            args.n_head = 2
        if args.block_size == 64:
            args.block_size = 16
        if args.batch_size == 16:
            args.batch_size = 8
        if args.epochs == 10_000:
            args.epochs = 200
        if args.warmup_steps == 50:
            args.warmup_steps = 5
        if args.checkpoint_interval == 50:
            args.checkpoint_interval = 6
        if args.eval_interval == 100:
            args.eval_interval = 12
        if args.bench_every == 0:
            args.bench_every = 8
        if args.max_restarts == 5:
            args.max_restarts = 1
        if args.json is None:
            args.json = str(base / "results.json")

    if not args.data:
        parser.error("--data is required (or use --smoke)")

    summary = run_supervisor(args)
    return 0 if summary.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
