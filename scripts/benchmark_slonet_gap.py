"""SloNet gap benchmark — "can it power X today?" (card 20261103_121700).

Answers ONE question with numbers before any build: at a stated model size S,
what is SloNet's inference decode throughput, training throughput, and eval
score — on CPU — against a stated target's SLO?

The target defaults to the v86 browser VM (an EMULATED x86): its throughput
budget is a fraction of bare metal, so the default SLO is deliberately low
and explicit. Composition only — this script must NOT grow a second training
loop: training goes through SloughGPTTrainer (train_pipeline.py), eval
through lm_eval_char.evaluate_soul_char_lm.

Usage:
    ./scripts/python scripts/benchmark_slonet_gap.py
    ./scripts/python scripts/benchmark_slonet_gap.py --size small --steps 50
    ./scripts/python scripts/benchmark_slonet_gap.py --results /tmp/slobench/gap.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import argparse
import json
import os
import time

# Before numpy: OpenBLAS reads these once at library load; the repo default
# is a single thread per op (SLO_OPENBLAS_NUM_THREADS=1).
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("SLO_OPENBLAS_NUM_THREADS", "1"))
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("SLO_OMP_NUM_THREADS", "1"))

import numpy as np

from domain.training._internal.train_pipeline import SloughGPTTrainer, TrainerConfig

# ── SLO constants (the "powering" target, stated — never implied) ────────────

#: The target this benchmark's verdict is stated for: the in-browser v86 VM.
DEFAULT_TARGET = "v86-browser-vm"

#: Minimum acceptable decode throughput for interactive use on the v86 VM.
#: v86 emulates x86 in the browser — the budget is a fraction of bare metal,
#: so this is a FLOOR for "usable", not a bare-metal chat speed.
DEFAULT_DECODE_SLO_TPS = 3.0

#: Model sizes: name -> (n_embed, n_layer, n_head, block_size). Mirrors the
#: training benchmark's config ladder so numbers stay comparable.
SIZES = {
    "tiny": (32, 1, 2, 32),
    "gate": (64, 1, 4, 32),
    "small": (64, 2, 4, 64),
    "medium": (128, 4, 4, 128),
}


# ── model ────────────────────────────────────────────────────────────────────


def build_model(vocab_size: int, n_embed: int, n_layer: int, n_head: int, block_size: int):
    """A SloTransformer at size S — the same class inference and eval use."""
    from domain.training._internal.slonet import SloTransformer

    return SloTransformer(
        vocab_size=vocab_size,
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block_size,
        dropout=0.0,
        _lazy=True,
    )


# ── measurement ──────────────────────────────────────────────────────────────


def measure_inference(model, prompt, new_tokens: int, repeats: int = 1) -> dict:
    """Decode + prefill throughput for `repeats` generate() calls.

    Token count is prescribed (repeats x new_tokens), never sampled — the
    benchmark measures the ENGINE, not the sampling outcome.
    """
    prompt = list(prompt)
    model.clear_kv_cache()

    # Warm-up: the first generate() on a _lazy model pays one-time init
    # (measured 2026-11-11: 856ms cold vs 12ms warmed = 69x). Timing it
    # reported init cost as prefill throughput (16-24 tok/s vs 649 decode —
    # a 40x "inversion" that was entirely artifact, since prefill is a
    # single parallel forward and must be FASTER per token than autoregressive
    # decode). One discarded call, then measure the steady state.
    model.generate(np.array([prompt], dtype=np.int64), max_new_tokens=1)
    model.clear_kv_cache()

    # Prefill: one warm forward over the prompt.
    t0 = time.perf_counter()
    model.generate(np.array([prompt], dtype=np.int64), max_new_tokens=1)
    prefill_s = max(time.perf_counter() - t0, 1e-9)

    t0 = time.perf_counter()
    for _ in range(repeats):
        model.clear_kv_cache()
        model.generate(np.array([prompt], dtype=np.int64), max_new_tokens=new_tokens)
    elapsed = max(time.perf_counter() - t0, 1e-9)

    tokens = new_tokens * repeats
    return {
        "tokens": tokens,
        "elapsed_s": round(elapsed, 4),
        "decode_tps": round(tokens / elapsed, 2),
        "prefill_tps": round(len(prompt) / prefill_s, 2),
    }


def measure_training(
    data_path: str,
    checkpoint_dir: str,
    max_steps: int = 50,
    n_embed: int = 64,
    n_layer: int = 2,
    n_head: int = 4,
    block_size: int = 64,
    batch_size: int = 32,
    seed: int = 0,
) -> dict:
    """Training throughput via SloughGPTTrainer — ONE loop, never a new one."""
    config = TrainerConfig(
        vocab_size=0,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=9999,
        max_checkpoints=1,
        log_interval=10_000,
        eval_interval=10_000,
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block_size,
        batch_size=batch_size,
        epochs=3,
        learning_rate=1e-3,
        max_steps=max_steps,
        min_data_quality=0.0,
        max_toxicity_rate=1.0,
    )
    np.random.seed(seed)
    trainer = SloughGPTTrainer(data_path=data_path, config=config, soul_name="gap-bench")
    t0 = time.perf_counter()
    result = trainer.train()
    elapsed = max(time.perf_counter() - t0, 1e-9)
    steps = int(result.get("total_steps", 0) or 0)
    tokens = steps * batch_size * block_size
    # TrainResult field order: final_loss (real) > train_loss > best_eval_loss
    # (best_eval_loss stays inf when no eval interval fired — never report it).
    final_loss = result.get("final_loss") or result.get("train_loss") or float("nan")
    if not np.isfinite(final_loss):
        final_loss = float("nan")
    return {
        "steps": steps,
        "tokens": tokens,
        "elapsed_s": round(elapsed, 4),
        "tokens_per_s": round(tokens / elapsed, 2),
        "final_loss": round(float(final_loss), 4),
    }


# ── corpus ───────────────────────────────────────────────────────────────────


def resolve_corpus(repo_root, cache_root=None) -> tuple["Path", str]:
    """Pick the training/eval corpus: OUR tokens first, never a silent swap.

    Order: own ``docs/**/*.md`` (this project's own prose — 'learn from our
    tokens') → external tinyshakespeare from the cache root → synthetic
    alphabet cycle (last resort, labeled). Returns (path, source-label);
    own-docs/tinyshakespeare corpora are materialised under ``/tmp/slobench-gap``
    with a deterministic (sorted) merge so scores stay reproducible.
    """
    repo_root = Path(repo_root)
    out_dir = Path("/tmp/slobench-gap")
    out_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted((repo_root / "docs").rglob("*.md"))
    if md_files:
        merged = out_dir / "own-docs-corpus.txt"
        merged.write_text(
            "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in md_files),
            encoding="utf-8",
        )
        return merged, "own-docs"

    if cache_root is None:
        from domain.training._internal.cache_tags import get_cache_root

        cache_root = get_cache_root()
    shakespeare = Path(cache_root) / "tinyshakespeare" / "corpus.jsonl"
    if shakespeare.exists():
        import json as _json

        texts = []
        for line in shakespeare.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                texts.append(_json.loads(line).get("text", ""))
            except _json.JSONDecodeError:
                continue
        merged = out_dir / "tinyshakespeare-corpus.txt"
        merged.write_text("\n".join(texts), encoding="utf-8")
        return merged, "tinyshakespeare"

    synthetic = out_dir / "synthetic-corpus.txt"
    cycle = "abcdefghij"
    synthetic.write_text("".join(cycle[i % len(cycle)] for i in range(20000)), encoding="utf-8")
    return synthetic, "synthetic"


# ── report ───────────────────────────────────────────────────────────────────


def load_target_calibration(paths: list[str]) -> dict | None:
    """Read v86 factor report(s) (scripts/benchmark_v86_factor.py output).

    Representative = first non-x87 entry (the guest has SSE2 and an
    in-guest decode compiles for it); x87 is carried as the floor case
    (i386 default build — worst plausible emulation). With no paths,
    nothing to calibrate with: None.
    """
    if not paths:
        return None
    entries = []
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "factor" not in data:
            raise ValueError(f"factor report {p} has no 'factor'")
        entries.append(
            {
                "path": str(p),
                # Reports written before the fp_mode field existed were run
                # with the benchmark's default compile mode: x87.
                "fp_mode": data.get("fp_mode", "x87"),
                "factor": data["factor"],
                "v86_decode_tps": data.get("v86_decode_tps"),
                "assumption": data.get("assumption"),
            }
        )
    floor = next((e for e in entries if e["fp_mode"] == "x87"), None)
    rep = next((e for e in entries if e["fp_mode"] != "x87"), entries[0])
    cal = dict(rep)
    cal["source"] = [e["path"] for e in entries]
    cal["floor"] = (
        {"fp_mode": floor["fp_mode"], "factor": floor["factor"]}
        if floor is not None and floor is not rep
        else None
    )
    return cal


def build_report(
    inference: dict,
    training: dict,
    eval: dict,
    model: dict,
    target: str = DEFAULT_TARGET,
    decode_slo: float = DEFAULT_DECODE_SLO_TPS,
    corpus_source: str = "unknown",
    target_calibration: dict | None = None,
) -> dict:
    """Assemble the gap report and render a verdict against the stated SLO.

    With a measured calibration (card 20261103_121721 C3) the verdict is
    stated ON the target machine — decode/factor — never on bare metal;
    without one, the verdict keeps the historical bare-metal comparison
    and says so via verdict_basis.
    """
    report = {
        "target": target,
        "decode_slo_tps": decode_slo,
        "verdict": "meets_slo" if inference["decode_tps"] >= decode_slo else "below_slo",
        "verdict_basis": "bare_metal_proxy",
        "model": model,
        "corpus": {"source": corpus_source},
        "inference": dict(inference),
        "training": training,
        "eval": eval,
    }
    if target_calibration is not None:
        factor = target_calibration["factor"]
        on_target = inference["decode_tps"] / factor
        report["inference"]["decode_tps_on_target"] = on_target
        report["verdict"] = "meets_slo" if on_target >= decode_slo else "below_slo"
        report["verdict_basis"] = "calibrated_target"
        report["target_calibration"] = target_calibration
        floor = target_calibration.get("floor")
        if floor:
            on_floor = inference["decode_tps"] / floor["factor"]
            report["inference"]["decode_tps_on_target_floor"] = on_floor
            report["floor_case_verdict"] = "meets_slo" if on_floor >= decode_slo else "below_slo"
    return report


def format_report(report: dict) -> str:
    """Human summary — numbers first, raw JSON one write away (never dumped)."""
    m = report["model"]
    i = report["inference"]
    t = report["training"]
    e = report["eval"]
    lines = [
        "=" * 72,
        f"  SloNet gap benchmark — target: {report['target']}",
        "=" * 72,
        f"  Model:   {m['config']}  ({m['params']:,} params)",
        f"  Corpus:  {report['corpus']['source']}",
        f"  Infer:   decode {i['decode_tps']} tok/s  prefill {i['prefill_tps']} tok/s"
        f"  ({i['tokens']} tokens in {i['elapsed_s']}s)",
        f"  Train:   {t['tokens_per_s']} tok/s  ({t['steps']} steps, final loss {t['final_loss']})",
        f"  Eval:    loss {e['mean_loss']}  ppl {e['perplexity']}"
        f"  ({e['num_token_positions']} positions)",
        f"  SLO:     {report['decode_slo_tps']} decode tok/s for {report['target']}",
    ]
    if "decode_tps_on_target" in i:
        cal = report["target_calibration"]
        lines.append(
            f"  Target:  bare metal {i['decode_tps']} -> ~{i['decode_tps_on_target']:.2f}"
            f" tok/s on {report['target']} (x{cal['factor']:.1f} {cal['fp_mode']})"
        )
        if "decode_tps_on_target_floor" in i:
            lines.append(
                f"  Floor:   x87-only build -> ~{i['decode_tps_on_target_floor']:.2f}"
                f" tok/s ({report['floor_case_verdict']})"
            )
    lines += [
        f"  VERDICT: {report['verdict'].upper()}"
        f" [{report.get('verdict_basis', 'bare_metal_proxy')}]",
        "=" * 72,
    ]
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="SloNet gap benchmark (can it power X today?)")
    parser.add_argument("--size", default="small", choices=sorted(SIZES), help="model size S")
    parser.add_argument("--steps", type=int, default=50, help="training steps for the report")
    parser.add_argument("--new-tokens", type=int, default=32, help="decode tokens per generate()")
    parser.add_argument("--repeats", type=int, default=3, help="generate() repeats for timing")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="SLO target name")
    parser.add_argument(
        "--seed", type=int, default=0, help="numpy RNG seed for init + batching (multi-seed sweeps)"
    )
    parser.add_argument(
        "--decode-slo", type=float, default=DEFAULT_DECODE_SLO_TPS, help="min decode tok/s"
    )
    parser.add_argument(
        "--results", default="/tmp/slobench/gap_report.json", help="report JSON path"
    )
    parser.add_argument(
        "--factor-report",
        action="append",
        default=None,
        metavar="JSON",
        help="v86 factor report (repeatable): non-x87 = representative estimate, x87 = floor case",
    )
    args = parser.parse_args()

    n_embed, n_layer, n_head, block_size = SIZES[args.size]

    # Corpus: OUR docs first ('learn from our tokens'), then tinyshakespeare,
    # then synthetic — the source is recorded in the report, never silent.
    repo = Path(__file__).resolve().parents[1]
    corpus, corpus_source = resolve_corpus(repo)
    print(f"Corpus: {corpus_source} ({corpus})")

    # Eval split: last 10% of the corpus, held out from training.
    text = Path(corpus).read_text(encoding="utf-8")
    split = int(len(text) * 0.9)
    train_file = Path("/tmp/slobench-gap-train.txt")
    eval_file = Path("/tmp/slobench-gap-eval.txt")
    train_file.write_text(text[:split], encoding="utf-8")
    eval_file.write_text(text[split:], encoding="utf-8")

    model = build_model(
        vocab_size=256, n_embed=n_embed, n_layer=n_layer, n_head=n_head, block_size=block_size
    )
    n_params = sum(int(p.numel()) for p in model.parameters() if p is not None)
    model_cfg = f"{n_embed}d/{n_layer}L/{n_head}H/b{block_size}"

    print(f"Model: {model_cfg} ({n_params:,} params) — inference measurement...")
    inference = measure_inference(
        model, prompt=list(range(16)), new_tokens=args.new_tokens, repeats=args.repeats
    )
    print(f"  decode {inference['decode_tps']} tok/s")

    print(f"Training measurement ({args.steps} steps via SloughGPTTrainer)...")
    training = measure_training(
        data_path=str(train_file),
        checkpoint_dir="/tmp/slobench-gap",
        max_steps=args.steps,
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block_size,
        seed=args.seed,
    )
    print(f"  {training['tokens_per_s']} tok/s (final loss {training['final_loss']})")

    # Eval: score the just-trained checkpoint on the held-out split.
    from domain.training._internal.lm_eval_char import evaluate_soul_char_lm

    checkpoint = max(
        Path("/tmp/slobench-gap").glob("*.soul"), key=lambda p: p.stat().st_mtime, default=None
    )
    if checkpoint is None:
        eval_result = {
            "mean_loss": float("nan"),
            "perplexity": float("nan"),
            "num_token_positions": 0,
        }
        print("  (no checkpoint found — eval section empty)")
    else:
        eval_result = evaluate_soul_char_lm(str(checkpoint), str(eval_file), max_chars=20000)
        print(f"  eval loss {eval_result['mean_loss']}  ppl {eval_result['perplexity']}")

    report = build_report(
        inference=inference,
        training=training,
        eval={
            "mean_loss": round(float(eval_result["mean_loss"]), 4),
            "perplexity": round(float(eval_result["perplexity"]), 2),
            "num_token_positions": int(eval_result["num_token_positions"]),
        },
        model={"params": n_params, "config": model_cfg, "size": args.size},
        target=args.target,
        decode_slo=args.decode_slo,
        corpus_source=corpus_source,
        target_calibration=load_target_calibration(args.factor_report),
    )

    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(format_report(report))
    print(f"Report saved to {out}")


if __name__ == "__main__":
    main()
