#!/usr/bin/env python3
"""Native auto-train adapter — JSON contract with AutoTrainer (goal 12).

Runs the one loop (``SloughGPTTrainer``) on a conversation-pair text file
and prints a single JSON result line matching the old ``hf_train.py`` contract:

    {"success": true, "loss": <float>, "steps": <int>, ...}

Usage:
    python scripts/train_auto_native.py --data pairs.txt --output models/auto-training/auto_123
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "core-py"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to training text file")
    parser.add_argument("--output", required=True, help="Output directory for checkpoint")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--n-embed", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    args = parser.parse_args()

    t0 = time.time()
    try:
        from domain.training._internal.train_pipeline import SloughGPTTrainer, TrainerConfig

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg = TrainerConfig(
            vocab_size=0,  # derived from corpus
            n_embed=args.n_embed,
            n_layer=args.n_layer,
            n_head=args.n_head,
            block_size=args.block_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            warmup_steps=1,
            checkpoint_dir=str(output_dir),
            checkpoint_interval=10_000,
            log_interval=100,
            eval_interval=10_000,
            min_data_quality=0.0,
            max_toxicity_rate=1.0,
            scheduler_type="constant",
        )

        trainer = SloughGPTTrainer(
            data_path=args.data,
            config=cfg,
            soul_name="auto-native",
        )
        result = trainer.train()

        final_path = output_dir / "auto"
        trainer.save(str(final_path), include_optimizer_state=False, is_final=True)

        loss = result.get("final_loss")
        if loss is None:
            loss = trainer._last_train_loss
        if loss is None:
            loss = 0.0

        payload = {
            "success": True,
            "loss": float(loss),
            "steps": int(result.get("global_step", 0)),
            "elapsed_s": round(time.time() - t0, 1),
            "model_path": str(final_path) + ".soul",
            "phase": "COMPLETE",
            "status": "complete",
        }
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
        return 0
    except Exception as e:  # noqa: BLE001 — CLI must emit JSON on any failure
        payload = {
            "success": False,
            "error": str(e),
            "loss": 0.0,
            "steps": 0,
            "elapsed_s": round(time.time() - t0, 1),
            "phase": "COMPLETE",
            "status": "error",
        }
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
