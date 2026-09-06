"""
Model Measurement Harness — comprehensive model quality and performance measurement.

Measures:
  - Perplexity (next-token prediction quality)
  - Latency (mean, p50, p95, p99 inference speed)
  - Throughput (tokens/sec)
  - Quality Score (coherence, repetition, diversity)
  - Memory Usage (RSS, peak)
  - Drift Detection (perplexity change over time)

Usage:
    # Measure default model
    python scripts/measure_model.py

    # Measure specific checkpoint
    python scripts/measure_model.py --soul models/checkpoint.soul

    # Measure with comparison baseline
    python scripts/measure_model.py --soul models/checkpoint.soul --baseline models/previous.soul

    # Export results to JSON
    python scripts/measure_model.py --output data/model_measurements.json

    # Quick measurement (fewer prompts)
    python scripts/measure_model.py --quick
"""

import sys
import json
import time
import math
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "core-py"))


def log(msg: str = ""):
    """Print progress to stderr so JSON output stays clean."""
    print(msg, file=sys.stderr)


# ── Evaluation Prompts ────────────────────────────────────────

FULL_EVAL_PROMPTS = {
    "What is 2+2?": "4 is the answer to 2+2.",
    "The capital of France is": "The capital of France is Paris.",
    "def fibonacci(n):": "def fibonacci(n): if n <= 1: return n. return fibonacci(n-1) + fibonacci(n-2)",
    "Explain gravity in one sentence:": "Gravity is the force that attracts objects toward each other.",
    "What is machine learning?": "Machine learning is a subset of AI that enables systems to learn from data.",
    "Hello, how are you?": "Hello! I am doing well, thank you for asking.",
    "The quick brown fox": "The quick brown fox jumps over the lazy dog.",
    "Once upon a time": "Once upon a time, in a land far away, there lived a king.",
    "To be or not ": "To be or not to be, that is the question:",
    "Write a function to sort a list:": "def sort_list(lst): return sorted(lst)",
}

QUICK_EVAL_PROMPTS = {
    "What is 2+2?": "4 is the answer to 2+2.",
    "The capital of France is": "The capital of France is Paris.",
    "Hello, how are you?": "Hello! I am doing well, thank you for asking.",
}

CORPUS_SENTENCES = [
    "the quick brown fox jumps over the lazy dog",
    "artificial intelligence is transforming how we interact with computers",
    "the sun rose over the mountains casting golden light across the valley",
    "machine learning models learn patterns from data to make predictions",
    "she opened the old wooden door and stepped into the dimly lit room",
    "natural language processing enables computers to understand human speech",
    "the scientist conducted experiments to verify the hypothesis",
    "a neural network consists of layers of interconnected nodes",
    "the river flowed gently through the ancient forest",
    "democracy is a system of government where citizens exercise power",
    "the chef prepared a magnificent meal with fresh local ingredients",
    "deep learning uses multiple layers to progressively extract features",
    "the spaceship launched into orbit carrying supplies to the station",
    "education is the most powerful weapon to change the world",
    "the musician played a melody that filled the concert hall with emotion",
]


# ── Metrics Dataclass ─────────────────────────────────────────

@dataclass
class ModelMeasurement:
    model_name: str
    timestamp: float = 0.0

    # Perplexity
    perplexity: Optional[float] = None
    perplexity_std: Optional[float] = None
    loss: Optional[float] = None

    # Latency
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Throughput
    tokens_per_sec: float = 0.0
    total_tokens: int = 0
    total_time_sec: float = 0.0

    # Quality
    bleu_score: float = 0.0
    repetition_rate: float = 0.0
    diversity_score: float = 0.0
    coherence_score: float = 0.0
    quality_score: float = 0.0

    # Memory
    peak_memory_mb: float = 0.0

    # Per-response data
    responses: List[Dict[str, Any]] = field(default_factory=list)

    # Comparison
    vs_baseline: Optional[Dict[str, Any]] = None


# ── BLEU Scorer ───────────────────────────────────────────────

class SimpleBLEU:
    """Simple BLEU scorer without external dependencies."""

    @staticmethod
    def score(candidate: str, reference: str, max_n: int = 4) -> float:
        import re
        cand_tokens = re.findall(r'\w+', candidate.lower())
        ref_tokens = re.findall(r'\w+', reference.lower())

        if not cand_tokens or not ref_tokens:
            return 0.0

        scores = []
        for n in range(1, max_n + 1):
            cand_ngrams = Counter(tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens) - n + 1))
            ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1))

            clipped = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in cand_ngrams.items())
            total = sum(cand_ngrams.values())

            if total == 0:
                scores.append(0.0)
            else:
                scores.append(clipped / total)

        # Geometric mean
        if any(s == 0 for s in scores):
            return 0.0

        log_avg = sum(math.log(s) for s in scores) / len(scores)
        return math.exp(log_avg)


# ── Measurement Functions ─────────────────────────────────────

def measure_perplexity(model, tokenizer, sentences: List[str]) -> Dict[str, float]:
    """Measure perplexity on a corpus of sentences."""
    losses = []

    for sentence in sentences:
        try:
            tokens = tokenizer.encode(sentence) if hasattr(tokenizer, 'encode') else list(sentence)
            if len(tokens) < 2:
                continue

            # Forward pass to get loss
            input_ids = np.array([tokens[:-1]], dtype=np.int64)
            targets = np.array([tokens[1:]], dtype=np.int64)

            logits = model.forward(input_ids)
            if hasattr(logits, 'logits'):
                logits = logits.logits

            # Compute cross-entropy loss
            vocab_size = logits.shape[-1]
            logits_flat = logits.reshape(-1, vocab_size)
            targets_flat = targets.reshape(-1)

            # Softmax
            exp_logits = np.exp(logits_flat - np.max(logits_flat, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            # Negative log likelihood
            nll = -np.log(probs[np.arange(len(targets_flat)), targets_flat] + 1e-10)
            loss = np.mean(nll)
            losses.append(float(loss))
        except Exception as e:
            log(f"  Warning: perplexity failed for sentence: {e}")
            continue

    if not losses:
        return {"perplexity": float('inf'), "loss": float('inf'), "std": 0.0}

    mean_loss = np.mean(losses)
    perplexity = float(np.exp(mean_loss))
    std = float(np.std(losses))

    return {"perplexity": perplexity, "loss": float(mean_loss), "std": std}


def measure_latency(model, tokenizer, prompts: List[str], warmup: int = 2) -> Dict[str, float]:
    """Measure inference latency across prompts."""
    latencies = []

    # Warmup
    for prompt in prompts[:warmup]:
        try:
            tokens = tokenizer.encode(prompt) if hasattr(tokenizer, 'encode') else list(prompt)
            input_ids = np.array([tokens], dtype=np.int64)
            model.generate(input_ids, max_new_tokens=20)
        except Exception:
            pass

    # Measurement
    for prompt in prompts:
        try:
            tokens = tokenizer.encode(prompt) if hasattr(tokenizer, 'encode') else list(prompt)
            input_ids = np.array([tokens], dtype=np.int64)

            start = time.perf_counter()
            output = model.generate(input_ids, max_new_tokens=50)
            elapsed = (time.perf_counter() - start) * 1000  # ms

            latencies.append(elapsed)
        except Exception:
            continue

    if not latencies:
        return {"mean": 0, "p50": 0, "p95": 0, "p99": 0}

    arr = np.array(latencies)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def measure_throughput(model, tokenizer, prompts: List[str]) -> Dict[str, Any]:
    """Measure tokens/sec throughput."""
    total_tokens = 0
    total_time = 0

    for prompt in prompts:
        try:
            tokens = tokenizer.encode(prompt) if hasattr(tokenizer, 'encode') else list(prompt)
            input_ids = np.array([tokens], dtype=np.int64)

            start = time.perf_counter()
            output = model.generate(input_ids, max_new_tokens=50)
            elapsed = time.perf_counter() - start

            if hasattr(output, 'shape'):
                num_tokens = output.shape[-1] - len(tokens)
            else:
                num_tokens = len(output) - len(tokens)

            total_tokens += max(0, num_tokens)
            total_time += elapsed
        except Exception:
            continue

    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0

    return {
        "tokens_per_sec": tokens_per_sec,
        "total_tokens": total_tokens,
        "total_time_sec": total_time,
    }


def measure_quality(model, tokenizer, prompts: Dict[str, str]) -> Dict[str, float]:
    """Measure output quality: BLEU, repetition, diversity, coherence."""
    bleu_scores = []
    repetition_rates = []
    response_lengths = []
    all_responses = []

    for prompt, reference in prompts.items():
        try:
            tokens = tokenizer.encode(prompt) if hasattr(tokenizer, 'encode') else list(prompt)
            input_ids = np.array([tokens], dtype=np.int64)
            output = model.generate(input_ids, max_new_tokens=80)

            if hasattr(output, 'tolist'):
                output_tokens = output.tolist()[0]
            else:
                output_tokens = output[0] if isinstance(output[0], list) else output

            # Decode response (skip input tokens)
            response_tokens = output_tokens[len(tokens):]
            if hasattr(tokenizer, 'decode'):
                response = tokenizer.decode(response_tokens)
            else:
                response = ''.join(chr(t) if 32 <= t < 127 else ' ' for t in response_tokens)

            all_responses.append(response)

            # BLEU
            bleu = SimpleBLEU.score(response, reference)
            bleu_scores.append(bleu)

            # Repetition rate (bigram repetition)
            words = response.lower().split()
            if len(words) > 1:
                bigrams = list(zip(words[:-1], words[1:]))
                unique_bigrams = len(set(bigrams))
                total_bigrams = len(bigrams)
                rep_rate = 1.0 - (unique_bigrams / total_bigrams) if total_bigrams > 0 else 0
                repetition_rates.append(rep_rate)

            response_lengths.append(len(words))
        except Exception as e:
            log(f"  Warning: quality measurement failed: {e}")
            continue

    # Diversity (type-token ratio across all responses)
    all_words = ' '.join(all_responses).lower().split()
    diversity = len(set(all_words)) / len(all_words) if all_words else 0

    # Coherence (inverse of repetition, weighted by BLEU)
    avg_rep = np.mean(repetition_rates) if repetition_rates else 0
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
    coherence = (1.0 - avg_rep) * 0.5 + avg_bleu * 0.5

    # Overall quality score (0-5 scale)
    quality = min(5.0, (avg_bleu * 2.0 + (1.0 - avg_rep) * 1.5 + diversity * 1.5))

    return {
        "bleu_score": float(avg_bleu),
        "repetition_rate": float(avg_rep),
        "diversity_score": float(diversity),
        "coherence_score": float(coherence),
        "quality_score": float(quality),
    }


def measure_memory() -> float:
    """Measure current process memory in MB."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024  # Convert KB to MB on Linux
    except Exception:
        return 0.0


# ── Main Measurement ──────────────────────────────────────────

def measure_model(
    soul_path: Optional[str] = None,
    quick: bool = False,
) -> ModelMeasurement:
    """Run full measurement suite on a model."""
    from domains.infrastructure.slonet_chat_provider import SloNetChatProvider

    log("Loading model...")
    if soul_path:
        provider = SloNetChatProvider.from_slnc(soul_path)
    else:
        provider = SloNetChatProvider.from_slnc("models/sloughgpt.soul")

    model = provider.model
    tokenizer = provider.tokenizer
    model_name = Path(soul_path).stem if soul_path else "default"

    prompts = QUICK_EVAL_PROMPTS if quick else FULL_EVAL_PROMPTS

    measurement = ModelMeasurement(
        model_name=model_name,
        timestamp=time.time(),
    )

    # Perplexity
    log("Measuring perplexity...")
    pplx = measure_perplexity(model, tokenizer, CORPUS_SENTENCES)
    measurement.perplexity = pplx["perplexity"]
    measurement.perplexity_std = pplx["std"]
    measurement.loss = pplx["loss"]
    log(f"  Perplexity: {pplx['perplexity']:.2f} (±{pplx['std']:.2f})")

    # Latency
    log("Measuring latency...")
    lat = measure_latency(model, tokenizer, list(prompts.keys()))
    measurement.mean_latency_ms = lat["mean"]
    measurement.p50_latency_ms = lat["p50"]
    measurement.p95_latency_ms = lat["p95"]
    measurement.p99_latency_ms = lat["p99"]
    log(f"  Latency: mean={lat['mean']:.1f}ms p50={lat['p50']:.1f}ms p95={lat['p95']:.1f}ms")

    # Throughput
    log("Measuring throughput...")
    tp = measure_throughput(model, tokenizer, list(prompts.keys()))
    measurement.tokens_per_sec = tp["tokens_per_sec"]
    measurement.total_tokens = tp["total_tokens"]
    measurement.total_time_sec = tp["total_time_sec"]
    log(f"  Throughput: {tp['tokens_per_sec']:.1f} tokens/sec")

    # Quality
    log("Measuring quality...")
    qual = measure_quality(model, tokenizer, prompts)
    measurement.bleu_score = qual["bleu_score"]
    measurement.repetition_rate = qual["repetition_rate"]
    measurement.diversity_score = qual["diversity_score"]
    measurement.coherence_score = qual["coherence_score"]
    measurement.quality_score = qual["quality_score"]
    log(f"  Quality: BLEU={qual['bleu_score']:.3f} rep={qual['repetition_rate']:.3f} div={qual['diversity_score']:.3f}")
    log(f"  Overall: {qual['quality_score']:.2f}/5.0")

    # Memory
    measurement.peak_memory_mb = measure_memory()
    log(f"  Memory: {measurement.peak_memory_mb:.1f} MB")

    return measurement


def compare_measurements(
    current: ModelMeasurement,
    baseline: Optional[ModelMeasurement] = None,
) -> Dict[str, Any]:
    """Compare current measurement against baseline."""
    if baseline is None:
        return {"status": "no_baseline"}

    comparison = {}

    # Perplexity (lower is better)
    if current.perplexity and baseline.perplexity:
        delta = current.perplexity - baseline.perplexity
        pct = (delta / baseline.perplexity) * 100
        comparison["perplexity"] = {
            "current": current.perplexity,
            "baseline": baseline.perplexity,
            "delta": delta,
            "pct_change": pct,
            "improved": delta < 0,
        }

    # Latency (lower is better)
    if current.mean_latency_ms and baseline.mean_latency_ms:
        delta = current.mean_latency_ms - baseline.mean_latency_ms
        pct = (delta / baseline.mean_latency_ms) * 100
        comparison["latency"] = {
            "current_ms": current.mean_latency_ms,
            "baseline_ms": baseline.mean_latency_ms,
            "delta_ms": delta,
            "pct_change": pct,
            "improved": delta < 0,
        }

    # Throughput (higher is better)
    if current.tokens_per_sec and baseline.tokens_per_sec:
        delta = current.tokens_per_sec - baseline.tokens_per_sec
        pct = (delta / baseline.tokens_per_sec) * 100
        comparison["throughput"] = {
            "current": current.tokens_per_sec,
            "baseline": baseline.tokens_per_sec,
            "delta": delta,
            "pct_change": pct,
            "improved": delta > 0,
        }

    # Quality (higher is better)
    if current.quality_score and baseline.quality_score:
        delta = current.quality_score - baseline.quality_score
        comparison["quality"] = {
            "current": current.quality_score,
            "baseline": baseline.quality_score,
            "delta": delta,
            "improved": delta > 0,
        }

    # Overall verdict
    improved = sum(1 for v in comparison.values() if v.get("improved"))
    degraded = sum(1 for v in comparison.values() if not v.get("improved") and v.get("delta", 0) != 0)
    comparison["verdict"] = "improved" if improved > degraded else "degraded" if degraded > improved else "mixed"

    return comparison


def save_measurement(measurement: ModelMeasurement, output_path: str):
    """Save measurement to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing measurements
    measurements = []
    if path.exists():
        try:
            measurements = json.loads(path.read_text())
        except Exception:
            measurements = []

    measurements.append(asdict(measurement))

    # Keep last 100 measurements
    measurements = measurements[-100:]

    path.write_text(json.dumps(measurements, indent=2))
    log(f"Saved measurement to {path}")


# ── CLI ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Measure model quality and performance")
    parser.add_argument("--soul", help="Path to .soul checkpoint")
    parser.add_argument("--baseline", help="Path to baseline .soul for comparison")
    parser.add_argument("--output", default="data/model_measurements.json", help="Output JSON path")
    parser.add_argument("--quick", action="store_true", help="Quick measurement (fewer prompts)")
    args = parser.parse_args()

    log("=" * 60)
    log("Model Measurement Harness")
    log("=" * 60)

    # Measure current model
    measurement = measure_model(soul_path=args.soul, quick=args.quick)

    # Compare against baseline if provided
    if args.baseline:
        log("\nMeasuring baseline...")
        baseline = measure_model(soul_path=args.baseline, quick=args.quick)
        comparison = compare_measurements(measurement, baseline)
        measurement.vs_baseline = comparison

        log("\n" + "=" * 60)
        log("Comparison Results:")
        log("=" * 60)
        for metric, data in comparison.items():
            if metric == "verdict":
                log(f"  Verdict: {data}")
            elif isinstance(data, dict) and "delta" in data:
                direction = "↑" if data.get("improved") else "↓"
                log(f"  {metric}: {data.get('current', 0):.3f} vs {data.get('baseline', 0):.3f} ({direction} {abs(data.get('pct_change', 0)):.1f}%)")

    # Save results
    save_measurement(measurement, args.output)

    # Print summary
    log("\n" + "=" * 60)
    log("Summary:")
    log("=" * 60)
    log(f"  Perplexity:     {measurement.perplexity:.2f}" if measurement.perplexity else "  Perplexity:     N/A")
    log(f"  Latency (p50):  {measurement.p50_latency_ms:.1f}ms")
    log(f"  Throughput:     {measurement.tokens_per_sec:.1f} tokens/sec")
    log(f"  BLEU:           {measurement.bleu_score:.3f}")
    log(f"  Repetition:     {measurement.repetition_rate:.3f}")
    log(f"  Diversity:      {measurement.diversity_score:.3f}")
    log(f"  Quality:        {measurement.quality_score:.2f}/5.0")
    log(f"  Memory:         {measurement.peak_memory_mb:.1f} MB")
    log("=" * 60)

    # Output JSON to stdout
    print(json.dumps(asdict(measurement), indent=2))


if __name__ == "__main__":
    main()
