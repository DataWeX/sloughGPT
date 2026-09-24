#!/usr/bin/env python3
"""Benchmark LoRA fine-tuning quality + speed on a compiled .slnc model.

Builds a tiny deterministic GPT-2-style .slnc model + tokenizer at runtime,
then LoRA fine-tunes it (pure NumPy via SloNet) and records:

- convergence: loss trajectory + reduction ratio (gate run reaches near-zero)
- speed: steps/sec, elapsed
- footprint: peak memory, adapter file size, LoRA param count
- progress: on_progress callback events (validates the router contract)

Run:
    python scripts/benchmark_hf_lora.py --json /tmp/lorabench-results.json
    python scripts/benchmark_results.py record --kind training --json-file /tmp/lorabench-results.json
"""

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

# Ensure imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "core-py"))

import numpy as np


def _build_model(dir_path: Path, n_embed: int, n_layer: int, vocab: int) -> Path:
    """Compile a tiny deterministic GPT-2-style .slnc model."""
    from domain.infrastructure._internal.slnc.compiler import SLNCCompiler

    config = {
        "n_layer": n_layer,
        "n_embd": n_embed,
        "n_head": 2 if n_embed == 32 else 4,
        "n_positions": 128,
        "vocab_size": vocab,
        "n_inner": n_embed * 2,
        "model_type": "gpt2",
        "max_position_embeddings": 128,
    }
    rng = np.random.RandomState(0)
    weight = lambda *shape: rng.randn(*shape).astype(np.float32) * 0.05

    weights = {
        "wte.weight": weight(vocab, n_embed),
        "wpe.weight": weight(128, n_embed),
        "ln_f.weight": np.ones(n_embed, np.float32),
        "ln_f.bias": np.zeros(n_embed, np.float32),
    }
    for i in range(n_layer):
        prefix = f"h.{i}."
        tensors = [
            ("ln_1.weight", (n_embed,)),
            ("ln_1.bias", (n_embed,)),
            ("attn.c_attn.weight", (n_embed, 3 * n_embed)),
            ("attn.c_attn.bias", (3 * n_embed,)),
            ("attn.c_proj.weight", (n_embed, n_embed)),
            ("attn.c_proj.bias", (n_embed,)),
            ("ln_2.weight", (n_embed,)),
            ("ln_2.bias", (n_embed,)),
            ("mlp.c_fc.weight", (n_embed, n_embed * 2)),
            ("mlp.c_fc.bias", (n_embed * 2,)),
            ("mlp.c_proj.weight", (n_embed * 2, n_embed)),
            ("mlp.c_proj.bias", (n_embed,)),
        ]
        for name, shape in tensors:
            weights[prefix + name] = weight(*shape)

    slnc_path = dir_path / f"model{n_embed}e{n_layer}l.slnc"
    SLNCCompiler().compile_from_dict(config, weights, str(slnc_path))
    return slnc_path


def _write_tokenizer(dir_path: Path, vocab: int) -> None:
    """Minimal byte-level BPE tokenizer.json for the compiled model."""
    tokenizer = {
        "model": {
            "type": "BPE",
            "vocab": {chr(i): i for i in range(128)},
            "merges": [],
            "eos_token_id": 0,
        },
        "pre_tokenizer": {"type": "ByteLevel"},
        "decoder": {"type": "ByteLevel"},
    }
    (dir_path / "tokenizer.json").write_text(json.dumps(tokenizer))


def _gate_corpus(path: Path) -> None:
    """Deterministic 10-cycle corpus — next char is a pure function of the previous."""
    cycle = "abcdefghijkl"
    corpus = "".join(cycle[i % len(cycle)] for i in range(513))
    path.write_text(corpus, encoding="ascii")


def _run_config(name: str, workdir: Path, cfg: dict) -> dict:
    from domain.training._internal.hf_lora_finetune import HFLoraConfig, HFLoraTrainer

    print(f"\n{'=' * 60}")
    print(f"  Config: {name}  ({cfg['n_embed']}d / {cfg['n_layer']}L)"
          f"  steps~{cfg['max_steps']}  target <{cfg['loss_target']})")
    print(f"{'=' * 60}")

    model_path = _build_model(workdir, cfg["n_embed"], cfg["n_layer"], cfg["vocab"])
    _write_tokenizer(workdir, cfg["vocab"])
    data_path = workdir / f"{name}-data.txt"
    _gate_corpus(data_path)

    lora_cfg = HFLoraConfig(
        model_path=str(model_path),
        data_path=str(data_path),
        rank=cfg["rank"],
        alpha=cfg["alpha"],
        target_modules=cfg["target_modules"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        block_size=cfg["block_size"],
        learning_rate=cfg["learning_rate"],
        output_dir=str(workdir),
        adapter_name=f"lora_{name}",
        log_interval=cfg["log_interval"],
    )

    progress = []

    def on_progress(info: dict) -> None:
        progress.append(dict(info))

    tracemalloc.start()
    t0 = time.time()
    result = HFLoraTrainer(lora_cfg).train(on_progress=on_progress)
    elapsed = time.time() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    steps = result.global_step or len(progress)
    losses = [p["loss"] for p in progress if p.get("loss") is not None]
    initial_loss = losses[0] if losses else 0.0
    final_loss = result.final_loss or (losses[-1] if losses else 0.0)

    converged = bool(
        initial_loss > 0
        and final_loss is not None
        and np.isfinite(final_loss)
        and final_loss > 0
        and final_loss < initial_loss * cfg["loss_ratio"]
        and final_loss < cfg["loss_target"]
    )

    adapter_path = workdir / f"lora_{name}.npz"
    adapter_bytes = adapter_path.stat().st_size if adapter_path.exists() else 0

    record = {
        "config": name,
        "n_embed": cfg["n_embed"],
        "n_layer": cfg["n_layer"],
        "rank": cfg["rank"],
        "epochs": cfg["epochs"],
        "total_steps": steps,
        "elapsed_s": round(elapsed, 2),
        "steps_per_sec": round(steps / max(elapsed, 0.001), 2),
        "initial_loss": round(initial_loss, 4),
        "final_loss": round(final_loss, 4),
        "convergence_ratio": round(initial_loss / max(final_loss, 1e-8), 2),
        "converged": converged,
        "gate_converged": converged if name == "gate" else None,
        "progress_events": len(progress),
        "lora_params": result.metrics.get("n_lora_params", 0) if result.metrics else 0,
        "adapter_bytes": adapter_bytes,
        "peak_memory_mb": round(peak_mem / 1024 / 1024, 1),
        "loss_curve": [{"step": p.get("step", 0), "loss": p["loss"]} for p in progress if p.get("loss") is not None],
    }

    print(
        f"\n  Result: {record['total_steps']} steps | {record['elapsed_s']}s | "
        f"{record['steps_per_sec']} steps/s | loss {record['initial_loss']} -> {record['final_loss']} | "
        f"conv {record['convergence_ratio']}x | {record['peak_memory_mb']} MB | "
        f"adapter {record['adapter_bytes']} B"
    )
    status = "PASS" if converged else "FAIL"
    print(f"  [GATE goal-17] {status}  final={final_loss:.4f} < {cfg['loss_target']} && "
          f"< {cfg['loss_ratio']}x of initial={initial_loss:.4f}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--json", default="/tmp/lorabench-results.json",
                        help="Output path for JSON results")
    args = parser.parse_args()

    workdir = Path("/tmp/lorabench-g17")
    workdir.mkdir(parents=True, exist_ok=True)

    configs = [
        {
            "name": "gate",
            "n_embed": 32,
            "n_layer": 1,
            "vocab": 128,
            "rank": 4,
            "alpha": 8.0,
            "target_modules": ["W_q", "W_k", "W_v", "W_o"],
            "epochs": 1,
            "batch_size": 8,
            "block_size": 32,
            "learning_rate": 1e-2,
            "max_steps": 60,
            "log_interval": 5,
            "loss_ratio": 0.95,
            "loss_target": 5.5,
        },
        {
            "name": "tiny",
            "n_embed": 32,
            "n_layer": 1,
            "vocab": 128,
            "rank": 4,
            "alpha": 8.0,
            "target_modules": ["W_q", "W_k", "W_v", "W_o"],
            "epochs": 1,
            "batch_size": 8,
            "block_size": 48,
            "learning_rate": 1e-3,
            "max_steps": 80,
            "log_interval": 10,
            "loss_ratio": 1.0,
            "loss_target": 100.0,
        },
    ]

    results = []
    for cfg in configs:
        name = cfg.pop("name")
        results.append(_run_config(name, workdir, cfg))
        cfg["name"] = name

    print(f"\n{'=' * 72}")
    print(f"  {'Config':<8} {'Steps':>6} {'Time':>7} {'s/s':>7} {'Loss->':>8} {'Conv':>6} {'Mem':>7}")
    print(f"  {'-' * 8} {'-' * 6} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 6} {'-' * 7}")
    for r in results:
        conv = "PASS" if r["converged"] else "FAIL"
        print(f"  {r['config']:<8} {r['total_steps']:>6} {r['elapsed_s']:>6.1f}s "
              f"{r['steps_per_sec']:>7.2f} {r['initial_loss']:>7.2f}->{r['final_loss']:<7.4f} "
              f"{conv:>6} {r['peak_memory_mb']:>6.1f}M")
    print(f"{'=' * 72}")

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
