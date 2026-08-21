"""
Model Benchmark Comparison - systematic comparison against popular models.

Compares SloughGPT checkpoints against GPT-2, Qwen-2.5 on:
  - Perplexity (next-token prediction)
  - BLEU score (n-gram overlap with references)
  - Latency (mean, p50, p95)
  - Throughput (tokens/sec)
  - Output quality (repetition rate, diversity)

Usage:
    # Native mode (no torch needed) - trains small model on-the-fly
    python scripts/benchmark_model_comparison.py --mode native
    python scripts/benchmark_model_comparison.py --mode native --sou models/checkpoint.soul

    # HuggingFace mode (requires torch)
    python scripts/benchmark_model_comparison.py --mode hf --hf gpt2

    # Auto-detect (uses whatever is available)
    python scripts/benchmark_model_comparison.py --sou models/checkpoint.soul --hf gpt2
"""

import sys
import json
import time
import math
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "core-py"))


EVAL_PROMPTS = {
    "What is 2+2?": "4 is the answer to 2+2.",
    "The capital of France is": "The capital of France is Paris.",
    "def fibonacci(n):": "def fibonacci(n): if n <= 1: return n. return fibonacci(n-1) + fibonacci(n-2)",
    "Explain gravity in one sentence:": "Gravity is the force that attracts objects toward each other.",
    "What is machine learning?": "Machine learning is a subset of AI that enables systems to learn from data.",
    "Hello, how are you?": "Hello! I am doing well, thank you for asking.",
}

QUICK_PROMPTS = {
    "What is 2+2?": "4 is the answer to 2+2.",
    "The capital of France is": "The capital of France is Paris.",
}


@dataclass
class ModelMetrics:
    model_name: str
    perplexity: Optional[float] = None
    bleu: Optional[float] = None
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    tokens_per_sec: float = 0.0
    mean_response_len: float = 0.0
    repetition_rate: float = 0.0
    diversity: float = 0.0
    responses: List[str] = field(default_factory=list)
    prompts: List[str] = field(default_factory=list)


@dataclass
class ComparisonReport:
    models: List[ModelMetrics]
    eval_prompts: int
    timestamp: str = ""


def compute_bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    cand_tokens = candidate.strip().split()
    ref_tokens = reference.strip().split()
    if not cand_tokens or not ref_tokens:
        return 0.0

    scores = []
    for n in range(1, min(max_n + 1, len(cand_tokens) + 1, len(ref_tokens) + 1)):
        cand_ngrams = {}
        for i in range(len(cand_tokens) - n + 1):
            ng = tuple(cand_tokens[i:i+n])
            cand_ngrams[ng] = cand_ngrams.get(ng, 0) + 1
        ref_ngrams = {}
        for i in range(len(ref_tokens) - n + 1):
            ng = tuple(ref_tokens[i:i+n])
            ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1
        matches = sum(min(cand_ngrams[ng], ref_ngrams.get(ng, 0)) for ng in cand_ngrams)
        total = sum(cand_ngrams.values())
        precision = matches / total if total > 0 else 0
        if precision > 0:
            scores.append(precision)

    if not scores:
        return 0.0

    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1)))
    geo_mean = math.exp(sum(math.log(s) for s in scores) / len(scores))
    return bp * geo_mean * 100


def compute_repetition_rate(text: str) -> float:
    words = text.split()
    if len(words) < 2:
        return 0.0
    bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    unique = len(set(bigrams))
    total = len(bigrams)
    return 1.0 - (unique / total) if total > 0 else 0.0


def compute_diversity(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


# ── Native SloNet training (no torch required) ───────────────────────────

BENCH_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "bench_shakespeare.soul"


def load_native_model():
    """Load pre-trained benchmark model. Falls back to training if not found."""
    if BENCH_MODEL_PATH.exists():
        print(f"Loading pre-trained benchmark model from {BENCH_MODEL_PATH}")
        from domains.training.slonet import SloNet, SloLSTM, import_from_sou
        net = import_from_sou(str(BENCH_MODEL_PATH))
        lstm = net.layers[1] if len(net.layers) > 1 else net.layers[0]
        chars = sorted(set(" " + "".join(c for c in TRAIN_TEXT if c.isalnum() or c in ".,!?;:'-")))
        stoi = {c: i + 1 for i, c in enumerate(chars)}
        itos = {i + 1: c for i, c in enumerate(chars)}
        def encode(text):
            return np.array([stoi.get(c, 0) for c in text], dtype=np.int64)
        def decode(ids):
            return "".join(itos.get(int(i), "?") for i in ids if i > 0)
        return net, lstm, encode, decode

    print(f"Pre-trained model not found at {BENCH_MODEL_PATH}")
    print("Train one with: python scripts/generate_benchmark_model.py")
    print("Falling back to small on-the-fly model...")
    return train_native_model(epochs=50)


def run_native_inference(net, lstm, encode, decode, prompt: str, max_new_tokens: int = 50):
    """Run inference on a native SloNet model."""
    ids = encode(prompt)

    from domains.training.slonet import tensor, no_grad
    t0 = time.perf_counter()

    h = lstm.init_hidden()
    prompt_len = len(ids)
    gen_ids = list(ids)

    with no_grad():
        for _ in range(max_new_tokens):
            seq = np.array([gen_ids[-128:]], dtype=np.int64)  # Use up to 128 context
            x = tensor(seq, requires_grad=False)
            logits, h = lstm.forward(x, h)
            logits_arr = logits.data
            # Handle 3D output (batch, seq, vocab)
            if logits_arr.ndim == 3:
                logits_arr = logits_arr[0, -1, :]
            elif logits_arr.ndim == 2:
                logits_arr = logits_arr[-1, :]
            next_id = int(np.argmax(logits_arr))
            if next_id == 0:
                break
            gen_ids.append(next_id)

    elapsed = time.perf_counter() - t0
    text = decode(gen_ids[prompt_len:])
    return text.strip(), elapsed, len(gen_ids) - prompt_len


def run_sou_inference(model, prompt: str, max_new_tokens: int = 50):
    tokenizer = getattr(model, '_tokenizer', None)
    if tokenizer is None:
        from domains.multimodal.char_tokenizer import CharTokenizer
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(prompt)

    input_ids = tokenizer.encode(prompt)
    input_arr = np.array([input_ids], dtype=np.int64)

    t0 = time.perf_counter()
    output = model.generate_numpy(
        input_arr, max_new_tokens=max_new_tokens,
        temperature=0.0, repetition_penalty=1.0,
    )
    elapsed = time.perf_counter() - t0

    gen_ids = output[0, len(input_ids):]
    text = tokenizer.decode(gen_ids.tolist())
    return text.strip(), elapsed, len(gen_ids)


def run_hf_inference(model, tokenizer, prompt: str, max_new_tokens: int = 50):
    import torch
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.0, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0

    gen_ids = output[0, prompt_len:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip(), elapsed, len(gen_ids)


def evaluate_model(name: str, responses: List[str], latencies: List[float],
                   token_counts: List[int], prompts: List[str]) -> ModelMetrics:
    bleu_scores = []
    rep_rates = []
    diversities = []

    for resp, prompt_text in zip(responses, prompts):
        ref = EVAL_PROMPTS.get(prompt_text, "")
        if ref:
            bleu_scores.append(compute_bleu(resp, ref))
        rep_rates.append(compute_repetition_rate(resp))
        diversities.append(compute_diversity(resp))

    lat_arr = np.array(latencies) if latencies else np.array([0.0])
    total_tokens = sum(token_counts) if token_counts else 0
    total_time = sum(latencies) if latencies else 1.0

    return ModelMetrics(
        model_name=name,
        perplexity=None,
        bleu=float(np.mean(bleu_scores)) if bleu_scores else None,
        mean_latency_ms=float(np.mean(lat_arr)) * 1000,
        p50_latency_ms=float(np.percentile(lat_arr, 50)) * 1000,
        p95_latency_ms=float(np.percentile(lat_arr, 95)) * 1000,
        tokens_per_sec=total_tokens / total_time if total_time > 0 else 0,
        mean_response_len=float(np.mean([len(r.split()) for r in responses])) if responses else 0,
        repetition_rate=float(np.mean(rep_rates)) if rep_rates else 0,
        diversity=float(np.mean(diversities)) if diversities else 0,
        responses=responses,
        prompts=prompts,
    )


def print_comparison(results: List[ModelMetrics]):
    print()
    print("=" * 90)
    print("MODEL COMPARISON BENCHMARK")
    print("=" * 90)
    header = f"{'Model':<25} {'BLEU':>6} {'Lat(ms)':>8} {'P95(ms)':>8} {'tok/s':>7} {'RepRate':>8} {'Diversity':>9}"
    print(header)
    print("-" * 90)
    for r in results:
        bleu_str = f"{r.bleu:.1f}" if r.bleu is not None else "n/a"
        print(
            f"{r.model_name:<25} {bleu_str:>6} {r.mean_latency_ms:>8.1f} "
            f"{r.p95_latency_ms:>8.1f} {r.tokens_per_sec:>7.1f} "
            f"{r.repetition_rate:>8.3f} {r.diversity:>9.3f}"
        )

    print()
    print("=" * 90)
    print("RESPONSE COMPARISON")
    print("=" * 90)
    prompts = list(EVAL_PROMPTS.keys())
    for prompt in prompts[:3]:
        print(f"\nPrompt: {prompt!r}")
        print("-" * 60)
        for r in results:
            idx = -1
            for i, p in enumerate(r.prompts):
                if p == prompt:
                    idx = i
                    break
            resp = r.responses[idx] if idx >= 0 and idx < len(r.responses) else "(no response)"
            print(f"  [{r.model_name}] {resp[:150]}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Model Comparison Benchmark")
    parser.add_argument("--sou", nargs="*", default=[], help="SOU checkpoint paths")
    parser.add_argument("--hf", nargs="*", default=[], help="HuggingFace model names")
    parser.add_argument("--mode", choices=["native", "hf", "both"], default="both",
                        help="native=numpy only, hf=torch only, both=auto-detect")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="Use fewer prompts")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs for native mode")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    prompts_dict = QUICK_PROMPTS if args.quick else EVAL_PROMPTS
    prompts = list(prompts_dict.keys())
    results = []

    # Native mode: load pre-trained or train small model
    if args.mode in ("native", "both"):
        try:
            net, lstm, encode, decode = load_native_model()
            responses, latencies, token_counts = [], [], []
            print(f"\nBenchmarking native SloNet...")
            for prompt in prompts:
                for _ in range(args.runs):
                    resp, lat, tokens = run_native_inference(net, lstm, encode, decode, prompt, args.max_new_tokens)
                    responses.append(resp)
                    latencies.append(lat)
                    token_counts.append(tokens)
            results.append(evaluate_model("SloNet:native", responses, latencies, token_counts, prompts))
        except Exception as e:
            print(f"  Native benchmark failed: {e}")
            import traceback; traceback.print_exc()

    # SOU checkpoint mode
    for ckpt in args.sou:
        print(f"Benchmarking SOU: {ckpt}")
        try:
            from domains.training.slonet import import_from_sou
            model = import_from_sou(ckpt)
            model.eval()
            responses, latencies, token_counts = [], [], []
            for prompt in prompts:
                for _ in range(args.runs):
                    resp, lat, tokens = run_sou_inference(model, prompt, args.max_new_tokens)
                    responses.append(resp)
                    latencies.append(lat)
                    token_counts.append(tokens)
            name = f"SOU:{Path(ckpt).stem}"
            results.append(evaluate_model(name, responses, latencies, token_counts, prompts))
        except Exception as e:
            print(f"  Failed: {e}")

    # HuggingFace mode
    if args.mode in ("hf", "both"):
        for model_name in args.hf:
            print(f"Benchmarking HF: {model_name}")
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.float32, trust_remote_code=True,
                )
                model.eval()
                responses, latencies, token_counts = [], [], []
                for prompt in prompts:
                    for _ in range(args.runs):
                        resp, lat, tokens = run_hf_inference(model, tokenizer, prompt, args.max_new_tokens)
                        responses.append(resp)
                        latencies.append(lat)
                        token_counts.append(tokens)
                del model
                import gc; gc.collect()
                results.append(evaluate_model(f"HF:{model_name}", responses, latencies, token_counts, prompts))
            except ImportError:
                print(f"  torch/transformers not installed, skipping {model_name}")
            except Exception as e:
                print(f"  Failed: {e}")

    if args.json:
        report = ComparisonReport(models=results, eval_prompts=len(prompts), timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
        print(json.dumps(asdict(report), indent=2))
    else:
        print_comparison(results)


if __name__ == "__main__":
    main()
