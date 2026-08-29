#!/usr/bin/env python3
"""Benchmark SloNet-native training on tinyshakespeare.

Measures: loss convergence, steps/sec, peak memory, final perplexity.
No PyTorch required — pure NumPy via SloNet.
"""

import sys, os, time, tracemalloc, json
from pathlib import Path

# Ensure imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "core-py"))

import numpy as np

def main():
    from domains.training.train_pipeline import SloughGPTTrainer, TrainerConfig
    data_path = str(Path(__file__).resolve().parents[1] / "data" / "tinyshakespeare" / "input.txt")
    print(f"Dataset: {data_path}")
    print(f"Data size: {os.path.getsize(data_path) / 1024:.1f} KB")

    configs = [
        {"name": "tiny",  "n_embed": 32,  "n_layer": 1, "n_head": 2, "block_size": 32,  "batch_size": 16, "epochs": 3, "learning_rate": 3e-3, "max_steps": 50},
        {"name": "small", "n_embed": 64,  "n_layer": 2, "n_head": 4, "block_size": 64,  "batch_size": 32, "epochs": 3, "learning_rate": 1e-3, "max_steps": 80},
        {"name": "medium","n_embed": 128, "n_layer": 4, "n_head": 4, "block_size": 128, "batch_size": 32, "epochs": 3, "learning_rate": 8e-4, "max_steps": 100},
    ]

    results = []
    for cfg in configs:
        name = cfg.pop("name")
        print(f"\n{'='*60}")
        print(f"  Config: {name}  ({cfg['n_embed']}d / {cfg['n_layer']}L / {cfg['n_head']}H)")
        print(f"{'='*60}")

        config = TrainerConfig(
            vocab_size=0,  # auto-detect
            checkpoint_dir="/tmp/slobench",
            checkpoint_interval=9999,
            max_checkpoints=1,
            log_interval=5,
            eval_interval=10,
            **cfg,
        )

        tracemalloc.start()
        t0 = time.time()

        progress_log = []
        def on_progress(info):
            step = info.get("global_step", 0)
            loss = info.get("train_loss")
            el = info.get("eval_loss")
            lr = info.get("learning_rate", 0)
            pct = info.get("progress_percent", 0)
            if loss is not None:
                progress_log.append({"step": step, "loss": float(loss), "eval_loss": float(el) if el else None})
                if step % 5 == 0 or info.get("done"):
                    eval_str = f" eval={el:.4f}" if el else ""
                    print(f"  step={step:4d}  loss={loss:.4f}{eval_str}  lr={lr:.2e}  ({pct}%)")

        trainer = SloughGPTTrainer(
            data_path=data_path,
            config=config,
            soul_name=f"bench-{name}",
        )

        result = trainer.train(on_progress=on_progress)
        elapsed = time.time() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        n_params = sum(p.numel() for p in trainer.model.parameters())
        final_loss = result.get("train_loss") or result.get("best_eval_loss")
        steps = result.get("total_steps", 0)
        loss_values = [p["loss"] for p in progress_log if p["loss"] is not None]

        # Convergence: loss reduction ratio
        if len(loss_values) >= 2:
            convergence = loss_values[0] / max(loss_values[-1], 1e-8)
        else:
            convergence = 0.0

        # Perplexity from final loss
        perplexity = np.exp(final_loss) if final_loss and final_loss < 10 else float("inf")

        r = {
            "config": name,
            "params": n_params,
            "params_readable": f"{n_params/1e3:.1f}K" if n_params < 1e6 else f"{n_params/1e6:.2f}M",
            "epochs": cfg.get("epochs", 3),
            "total_steps": steps,
            "elapsed_s": round(elapsed, 2),
            "steps_per_sec": round(steps / max(elapsed, 0.001), 2),
            "initial_loss": round(loss_values[0], 4) if loss_values else None,
            "final_loss": round(final_loss, 4) if final_loss else None,
            "convergence_ratio": round(convergence, 2),
            "perplexity": round(perplexity, 2) if perplexity < 1e6 else "inf",
            "peak_memory_mb": round(peak_mem / 1024 / 1024, 1),
            "loss_curve": [{"step": p["step"], "loss": p["loss"]} for p in progress_log if p["loss"] is not None],
        }
        results.append(r)

        print(f"\n  Result: {r['params_readable']} params | {r['elapsed_s']}s | "
              f"{r['steps_per_sec']} steps/s | loss {r['initial_loss']} → {r['final_loss']} | "
              f"ppl {r['perplexity']} | {r['peak_memory_mb']} MB")

    # Summary table
    print(f"\n{'='*72}")
    print(f"  {'Config':<8} {'Params':>8} {'Time':>7} {'s/s':>7} {'Loss→':>8} {'PPL':>8} {'Mem':>7}")
    print(f"  {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*7}")
    for r in results:
        print(f"  {r['config']:<8} {r['params_readable']:>8} {r['elapsed_s']:>6.1f}s "
              f"{r['steps_per_sec']:>7.2f} {r['final_loss'] or 0:>8.4f} "
              f"{r['perplexity']:>8} {r['peak_memory_mb']:>6.1f}M")
    print(f"{'='*72}")

    # Save results
    out = Path("/tmp/slobench/results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")

if __name__ == "__main__":
    main()
