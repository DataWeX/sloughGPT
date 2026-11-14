#!/usr/bin/env python3
"""Sequential stage training over sealed manifests (CLI host).

Runs ``domain.training._internal.sequential.run_sequential_stages`` — the
same adapter the server's training-job runtime uses — as a fallback for
jobs started outside the browser. Prints live loss, the final probe
standings matrix, and the trained-on chain.

Example:
    ./scripts/python scripts/train_sequential.py \\
        --manifests data/crunched/sys_seed/v1/manifest.json \\
                    data/crunched/github_mini/v1/manifest.json \\
        --steps-per-stage 300
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--manifests",
        nargs="+",
        required=True,
        help="Sealed manifest.json paths, in training order",
    )
    parser.add_argument("--steps-per-stage", type=int, default=1000)
    parser.add_argument("--probe-batches", type=int, default=5)
    parser.add_argument(
        "--checkpoint-dir",
        default="runs/sequential/checkpoints",
        help="Shared checkpoint dir — the continuity channel between stages",
    )
    parser.add_argument("--n-embed", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10, help="Per-stage fallback budget source")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default=None, help="cpu|cuda|mps|auto (default: trainer auto)")
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from domain.training._internal.sequential import run_sequential_stages

    trainer_kwargs: dict = {
        "n_embed": args.n_embed,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "checkpoint_dir": args.checkpoint_dir,
    }
    if args.device:
        trainer_kwargs["device"] = args.device

    last_logged_step = -1

    def on_progress(info: dict) -> None:
        nonlocal last_logged_step
        step = int(info.get("global_step", 0))
        if step // max(1, args.log_interval) != last_logged_step // max(1, args.log_interval):
            last_logged_step = step
            loss = info.get("train_loss")
            loss_s = f"{loss:.4f}" if isinstance(loss, (int, float)) else "?"
            print(
                f"[step {step}] loss={loss_s} epoch={info.get('epoch')}",
                file=sys.stderr,
                flush=True,
            )

    result = run_sequential_stages(
        args.manifests,
        trainer_kwargs=trainer_kwargs,
        steps_per_stage=args.steps_per_stage,
        probe_batches=args.probe_batches,
        on_progress=on_progress,
    )
    standings = result["standings"]
    if not standings:
        print("no stage ran (cancelled?)", file=sys.stderr)
        return 1

    refs: list[str] = []
    for entry in standings:
        for ref in entry["probes"]:
            if ref not in refs:
                refs.append(ref)

    print("\nprobe standings (loss; up = forgot, down = held/improved)")
    header = f"{'stage':<28} {'step':>7} " + " ".join(f"{r:>14}" for r in refs)
    print(header)
    print("-" * len(header))
    for entry in standings:
        cells = " ".join(
            f"{entry['probes'].get(r):>14}" if entry["probes"].get(r) is not None else f"{'-':>14}"
            for r in refs
        )
        print(f"{entry['stage'] + 1}: {entry['dataset']:<25} {entry['global_step']:>7} {cells}")

    print("\ntrained-on chain:")
    for link in result["trained_on"]:
        print(f"  stage {link.get('stage')}: {link.get('ref')} @ step {link.get('global_step')}")

    return 0 if len(standings) == len(args.manifests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
