"""
Model Comparison Benchmark — Side-by-side quality & speed comparison

Compares multiple models (checkpoints or HF models) on the same prompts:
  - Latency (mean, p50, p95)
  - Throughput (tokens/sec)
  - Output length consistency
  - Side-by-side response comparison

Usage:
    # Compare two SOU checkpoints
    python scripts/benchmark_compare.py \\
        --checkpoints models/gpt2_distill/model.soul models/gpt2_distill_finetuned/model.soul

    # Compare a SOU checkpoint against HuggingFace
    python scripts/benchmark_compare.py \\
        --checkpoints models/gpt2_distill/model.soul \\
        --hf-models gpt2 Qwen/Qwen2.5-0.5B-Instruct

    # Custom prompts
    python scripts/benchmark_compare.py \\
        --checkpoints model.soul \\
        --prompts "What is 2+2?" "Explain gravity" "Write a haiku"

    # JSON output
    python scripts/benchmark_compare.py --checkpoints model.soul --json
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# Add core-py to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "core-py"))


DEFAULT_PROMPTS = [
    "What is 2+2?",
    "The capital of France is",
    "Write a short poem about rain:",
    "def fibonacci(n):",
    "Explain quantum computing in one sentence:",
]


@dataclass
class ModelResult:
    name: str
    prompts: List[str]
    responses: List[str]
    latencies: List[float]
    token_counts: List[int]

    @property
    def mean_latency(self) -> float:
        return float(np.mean(self.latencies)) if self.latencies else 0.0

    @property
    def p50_latency(self) -> float:
        return float(np.percentile(self.latencies, 50)) if self.latencies else 0.0

    @property
    def p95_latency(self) -> float:
        return float(np.percentile(self.latencies, 95)) if self.latencies else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(self.token_counts)

    @property
    def tokens_per_sec(self) -> float:
        total_time = sum(self.latencies)
        return self.total_tokens / total_time if total_time > 0 else 0.0

    @property
    def mean_response_len(self) -> float:
        return float(np.mean([len(r) for r in self.responses])) if self.responses else 0.0

    @property
    def len_cv(self) -> float:
        lengths = [len(r) for r in self.responses]
        if len(lengths) < 2:
            return 0.0
        return float(np.std(lengths) / (np.mean(lengths) + 1e-10))


@dataclass
class ComparisonReport:
    models: List[ModelResult]
    prompts: List[str]


def benchmark_sou(checkpoint_path: str, prompts: List[str],
                  max_new_tokens: int = 50, runs: int = 1) -> ModelResult:
    """Benchmark a SOU checkpoint via SloNet numpy engine."""
    from domains.training.slonet import SloTransformer, SloNet, import_from_sou

    model = import_from_sou(checkpoint_path)
    model.eval()

    has_generate = hasattr(model, 'generate_numpy')

    # Build vocab from tokenizer if available
    tokenizer = getattr(model, '_tokenizer', None)
    if tokenizer is None:
        # Try loading from the checkpoint's directory
        ckpt_dir = Path(checkpoint_path).parent
        tok_path = ckpt_dir / "tokenizer.json"
        if tok_path.exists():
            from domains.multimodal.char_tokenizer import CharTokenizer
            tokenizer = CharTokenizer()
            tokenizer.load(str(tok_path))

    if tokenizer is None:
        # Fallback: char tokenizer from checkpoint metadata
        from domains.multimodal.char_tokenizer import CharTokenizer
        tokenizer = CharTokenizer()
        tokenizer.build_vocab("".join(prompts))

    if not has_generate:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is a plain SloNet (not SloTransformer). "
            "Only SloTransformer checkpoints support generate_numpy()."
        )

    responses = []
    latencies = []
    token_counts = []

    for prompt in prompts:
        for _ in range(runs):
            input_ids = tokenizer.encode(prompt)
            input_arr = np.array([input_ids], dtype=np.int64)

            t0 = time.perf_counter()
            output = model.generate_numpy(
                input_arr,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                repetition_penalty=1.0,
            )
            elapsed = time.perf_counter() - t0

            gen_ids = output[0, len(input_ids):]
            text = tokenizer.decode(gen_ids.tolist())

            responses.append(text)
            latencies.append(elapsed)
            token_counts.append(len(gen_ids))

    name = Path(checkpoint_path).stem
    return ModelResult(
        name=f"SOU:{name}",
        prompts=prompts,
        responses=responses,
        latencies=latencies,
        token_counts=token_counts,
    )


def benchmark_hf(model_name: str, prompts: List[str],
                 max_new_tokens: int = 50, runs: int = 1) -> Optional[ModelResult]:
    """Benchmark a HuggingFace model (requires torch)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print(f"  ⚠ torch/transformers not installed — skipping {model_name}")
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True,
        )
        model.eval()
    except Exception as e:
        print(f"  ⚠ Failed to load {model_name}: {e}")
        return None

    responses = []
    latencies = []
    token_counts = []

    for prompt in prompts:
        for _ in range(runs):
            inputs = tokenizer(prompt, return_tensors="pt")
            prompt_len = inputs["input_ids"].shape[1]

            t0 = time.perf_counter()
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            elapsed = time.perf_counter() - t0

            gen_ids = output[0, prompt_len:].tolist()
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            responses.append(text)
            latencies.append(elapsed)
            token_counts.append(len(gen_ids))

    # Free memory
    del model
    import gc
    gc.collect()

    return ModelResult(
        name=f"HF:{model_name}",
        prompts=prompts,
        responses=responses,
        latencies=latencies,
        token_counts=token_counts,
    )


def print_comparison(results: List[ModelResult], prompts: List[str]):
    """Print a formatted comparison table."""
    if not results:
        print("No results to display.")
        return

    # ── Summary Table ──
    print("\n" + "=" * 80)
    print("MODEL COMPARISON BENCHMARK")
    print("=" * 80)

    header = f"{'Model':<25} {'Mean(ms)':>10} {'P95(ms)':>10} {'tok/s':>8} {'AvgLen':>8} {'LenCV':>8}"
    print(header)
    print("-" * 80)

    for r in results:
        print(
            f"{r.name:<25} {r.mean_latency*1000:>10.1f} {r.p95_latency*1000:>10.1f} "
            f"{r.tokens_per_sec:>8.1f} {r.mean_response_len:>8.1f} {r.len_cv:>8.3f}"
        )

    # ── Side-by-side responses ──
    print("\n" + "=" * 80)
    print("RESPONSE COMPARISON")
    print("=" * 80)

    n_models = len(results)
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt: {prompt!r}")
        print("-" * 60)
        for j, r in enumerate(results):
            resp = r.responses[i] if i < len(r.responses) else "(no response)"
            print(f"  [{r.name}] {resp[:200]}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Model Comparison Benchmark")
    parser.add_argument("--checkpoints", nargs="*", default=[],
                        help="SOU checkpoint paths to compare")
    parser.add_argument("--hf-models", nargs="*", default=[],
                        help="HuggingFace model names to compare")
    parser.add_argument("--prompts", nargs="*", default=None,
                        help="Custom prompts (default: 5 built-in)")
    parser.add_argument("--max-new-tokens", type=int, default=50,
                        help="Max tokens to generate per prompt")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs per prompt (for stable latency)")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()

    prompts = args.prompts or DEFAULT_PROMPTS

    if not args.checkpoints and not args.hf_models:
        parser.error("Provide at least --checkpoints or --hf-models")

    results = []

    for ckpt in args.checkpoints:
        print(f"Benchmarking SOU: {ckpt}")
        try:
            r = benchmark_sou(ckpt, prompts, args.max_new_tokens, args.runs)
            results.append(r)
        except Exception as e:
            print(f"  ⚠ Failed: {e}")

    for model_name in args.hf_models:
        print(f"Benchmarking HF: {model_name}")
        r = benchmark_hf(model_name, prompts, args.max_new_tokens, args.runs)
        if r:
            results.append(r)

    if args.json:
        report = ComparisonReport(models=results, prompts=prompts)
        print(json.dumps(asdict(report), indent=2))
    else:
        print_comparison(results, prompts)


if __name__ == "__main__":
    main()
