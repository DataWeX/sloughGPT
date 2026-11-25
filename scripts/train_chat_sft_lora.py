#!/usr/bin/env python3
"""Step-3 LoRA run: chat-sft seed through the BatchArray spine (card 20261101_052145).

The first production call of the adapter door: ``REGISTRY.get("chat-sft").build()``
-> ``SloughGPTTrainer`` (BatchArray data door, ADR-002 §2/§4) -> LoRA r=8
(frozen base, adapter-only updates) -> loss curve + final ``.soul`` checkpoint.

Usage (project env, repo root):

    ./scripts/python scripts/train_chat_sft_lora.py --steps 60 --precision fp16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=Path,
        default=repo_root / "tests" / "fixtures" / "chat_sft_seed.jsonl",
        help="trajectory JSONL the chat-sft adapter builds from",
    )
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="fp16", choices=("auto", "fp32", "fp16"))
    parser.add_argument("--out", type=Path, default=repo_root / "checkpoints" / "chat_sft_lora")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-embed", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--soul-name", default="chat-sft-lora")
    args = parser.parse_args()

    from domain.slolib._internal.gpu import set_accelerator_precision
    from domain.training._internal.adapter_registry import REGISTRY, boot
    from domain.training._internal.train_pipeline import SloughGPTTrainer, TrainerConfig

    specs = boot()
    print(f"adapters booted: {', '.join(s.name for s in specs)}")

    array = REGISTRY.get("chat-sft").build(args.seed)
    chars = sum(len(item) for item in array)
    print(f"chat-sft seed: {len(array)} train records, {chars} chars")

    config = TrainerConfig(
        vocab_size=0,  # corpus-derived (the seed's char vocab) — never raw 256
        n_embed=args.n_embed,
        n_layer=args.n_layer,
        n_head=args.n_head,
        block_size=args.block_size,
        dropout=0.1,
        batch_size=args.batch_size,
        max_steps=args.steps,
        learning_rate=args.lr,
        warmup_steps=10,
        scheduler_type="cosine",
        min_lr=1e-5,
        checkpoint_dir=str(args.out),
        checkpoint_interval=args.steps + 1,  # the runner saves explicitly at the end
        eval_interval=max(args.steps // 2, 1),
        log_interval=args.log_interval,
        use_lora=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=args.device,
        max_checkpoints=2,
    )

    started = time.time()
    trainer = SloughGPTTrainer(array, config=config, soul_name=args.soul_name)
    precision = set_accelerator_precision(args.precision)
    lora_params = sum(p.numel() for name, p in trainer.model.named_parameters() if "lora_" in name)
    total_params = sum(p.numel() for p in trainer.model.parameters())
    print(
        f"device={trainer.device} precision={precision} vocab={trainer.vocab_size} "
        f"params={total_params:,} (lora {lora_params:,})"
    )

    curve: list[dict] = []

    def on_progress(info: dict) -> None:
        # on_progress emits train_loss/global_step (not loss/step)
        step, loss = info.get("global_step"), info.get("train_loss")
        if loss is None:
            return
        curve.append({"step": step, "loss": float(loss), "eval_loss": info.get("eval_loss")})
        if step % args.log_interval == 0:
            eval_str = f" eval={info['eval_loss']:.4f}" if info.get("eval_loss") else ""
            print(f"  step={step:4d}  loss={loss:.4f}{eval_str}")

    result = trainer.train(on_progress=on_progress)
    elapsed = time.time() - started

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "model"
    trainer.save(str(checkpoint))

    final_loss = result.get("train_loss") or result.get("best_eval_loss")
    summary = {
        "seed": str(args.seed),
        "records": len(array),
        "chars": chars,
        "device": trainer.device,
        "precision": precision,
        "steps": result.get("total_steps"),
        "params": total_params,
        "lora_params": lora_params,
        "lora_rank": args.lora_rank,
        "initial_loss": curve[0]["loss"] if curve else None,
        "final_loss": final_loss,
        "elapsed_sec": round(elapsed, 2),
        "checkpoint": str(checkpoint) + ".soul",
        "curve": curve,
    }
    (args.out / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    printable = {k: v for k, v in summary.items() if k != "curve"}
    print(json.dumps(printable, indent=2))
    print(f"loss curve: {len(curve)} points -> {args.out / 'run.json'}")

    if not final_loss or not (final_loss > 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
