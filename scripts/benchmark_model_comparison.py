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

SHAKESPEARE_PROMPTS = {
    "To be or not ": "To be or not to be, that is the question:",
    "The quality of ": "The quality of mercy is not strained;",
    "Friends, Romans, ": "Friends, Romans, countrymen, lend me your ears;",
    "If music be ": "If music be the food of love, play on;",
    "All the world ": "All the world's a stage,",
    "Shall I compare ": "Shall I compare thee to a summer's day?",
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
TRAINED_MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "auto-training"

TRAIN_TEXT = (
    "the quick brown fox jumps over the lazy dog. "
    "pack my box with five dozen liquor jugs. "
    "how vexingly quick daft zebras jump. "
    "the five boxing wizards jump quickly. "
    "sphinx of black quartz judge my vow. "
    "two driven jocks help fax my big quiz. "
    "five quacking zephyrs jolt my wax bed. "
    "the jay pig fox zebra and my wolves quack. "
    "blowzy red vixens fight for a quick jump. "
    "cozy lummox gives smart squid who asks for job pen. "
) * 10


def train_native_model(epochs: int = 30):
    """Train a small SloNet model on-the-fly for benchmarking."""
    from domains.training.slonet import (
        SloNet, SloEmbedding, SloLSTM, SloAdam,
        cross_entropy, tensor, _sample_from_logits,
    )

    chars = sorted(set(TRAIN_TEXT))
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    itos = {i + 1: c for i, c in enumerate(chars)}
    vocab_size = len(chars) + 1
    charset = "".join(chars)

    def encode(text):
        return np.array([stoi.get(c, 0) for c in text], dtype=np.int64)

    def decode(ids):
        return "".join(itos.get(int(i), "?") for i in ids if i > 0)

    net = SloNet(
        layers=[SloEmbedding(vocab_size, 32), SloLSTM(vocab_size, 32, 64, num_layers=1, dropout=0.0)],
        soul_name="bench_native",
    )
    lstm = net.layers[1]
    opt = SloAdam(lr=0.01)
    data = encode(TRAIN_TEXT)
    chunk = 32

    print(f"Training native model ({epochs} epochs, vocab={vocab_size})...")
    for ep in range(epochs):
        order = np.random.permutation(max(1, len(data) - chunk))
        ep_loss = 0.0
        steps = 0
        for pos in order[:30]:
            x = tensor(data[pos:pos + chunk].reshape(1, -1), requires_grad=True)
            y = tensor(data[pos + 1:pos + chunk + 1].reshape(1, -1))
            h = lstm.init_hidden()
            logits, _ = lstm.forward(x, h)
            loss = cross_entropy(logits, y.reshape(-1))
            ep_loss += float(loss.data)
            steps += 1
            loss.backward()
            opt.step(lstm.parameters())
            lstm.zero_grad()
        if ep % 10 == 0:
            print(f"  epoch {ep}: loss={ep_loss / max(steps, 1):.4f}")

    return net, lstm, encode, decode, charset


def _find_best_trained_model():
    """Find the best available trained model in auto-training dir."""
    if not TRAINED_MODELS_DIR.exists():
        return None
    candidates = sorted(TRAINED_MODELS_DIR.glob("*.soul"), key=lambda p: p.stat().st_size, reverse=True)
    for path in candidates:
        if path.stat().st_size > 1_000_000:  # >1MB = real model
            return path
    return None


def load_native_model():
    """Load pre-trained model for benchmarking.
    Returns (net, lstm_or_None, encode_fn, decode_fn, charset_str).
    Priority: trained transformer > bench LSTM > train on-the-fly."""
    # 1. Try finding a trained transformer in auto-training
    trained_path = _find_best_trained_model()
    if trained_path:
        print(f"Loading trained model from {trained_path}")
        from domains.training.slonet import import_from_sou
        net = import_from_sou(str(trained_path))
        if hasattr(net, 'generate'):
            meta_raw = _read_soul_metadata(trained_path)
            md = meta_raw.get("metadata", {}) if meta_raw else {}
            stoi = md.get("stoi", {})
            itos = md.get("itos", {})
            charset = md.get("chars", "")
            if isinstance(charset, list):
                charset = "".join(charset)
            if itos:
                if isinstance(itos, list):
                    itos_map = {i: c for i, c in enumerate(itos)}
                else:
                    itos_map = {int(k): v for k, v in itos.items()}
                stoi_map = {v: k for k, v in itos_map.items()}
                encode = lambda text: np.array([stoi_map.get(c, 0) for c in text], dtype=np.int64).reshape(1, -1)
                decode_tokens = lambda ids: "".join(itos_map.get(int(i), "?") for i in ids.flatten() if int(i) in itos_map)
                return net, None, encode, decode_tokens, charset
            else:
                vocab_size = md.get("vocab_size", 256)
                encode = lambda text: np.array([ord(c) % vocab_size for c in text], dtype=np.int64).reshape(1, -1)
                decode_tokens = lambda ids: "".join(chr(int(i)) if 32 <= int(i) < 127 else "?" for i in ids.flatten())
                return net, None, encode, decode_tokens, ""

    # 2. Try the bench LSTM .soul
    soul_path = BENCH_MODEL_PATH
    if soul_path.exists():
        print(f"Loading benchmark LSTM from {soul_path}")
        from domains.training.slonet import SloNet, SloLSTM, import_from_sou
        net = import_from_sou(str(soul_path))
        lstm = net.layers[1] if len(net.layers) > 1 else net.layers[0]
        meta = getattr(net, 'metadata', {})
        inner_meta = meta.get('metadata', {})
        charset = inner_meta.get('charset', '') if isinstance(inner_meta, dict) else ''
        if charset:
            chars = sorted(set(charset))
        else:
            chars = sorted(set(" " + "".join(c for c in TRAIN_TEXT if c.isalnum() or c in ".,!?;:'-")))
        stoi = {c: i + 1 for i, c in enumerate(chars)}
        itos = {i + 1: c for i, c in enumerate(chars)}
        def encode(text):
            return np.array([stoi.get(c, 0) for c in text], dtype=np.int64)
        def decode(ids):
            return "".join(itos.get(int(i), "?") for i in ids if i > 0)
        return net, lstm, encode, decode, charset

    print("No pre-trained model found.")
    print("Train one with: python scripts/generate_benchmark_model.py")
    print("Falling back to small on-the-fly model...")
    return *train_native_model(epochs=50), ""


def _read_soul_metadata(path):
    """Read the JSON metadata from a .soul file."""
    import struct, json
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:4] not in (b"SOU\x00", b"SOUL"):
        return None
    json_len = struct.unpack("<I", raw[8:12])[0]
    meta_bytes = raw[12:12 + json_len].rstrip(b"\x00")
    return json.loads(meta_bytes.decode())


def _detect_charset(net):
    """Detect charset from a loaded model's metadata."""
    meta_raw = getattr(net, "_raw_metadata", None)
    if meta_raw is None:
        # Try reading from metadata attribute
        meta = getattr(net, "metadata", {})
        inner = meta.get("metadata", {})
        if isinstance(inner, dict):
            return inner.get("charset", "")
    return ""


def run_native_inference(net, lstm, encode, decode, prompt: str, max_new_tokens: int = 50):
    """Run inference on a native model (LSTM or Transformer)."""
    input_ids = encode(prompt)

    t0 = time.perf_counter()

    if lstm is not None:
        # LSTM path
        gen_ids = list(input_ids.flatten())
        from domains.training.slonet import tensor, no_grad
        h = lstm.init_hidden()
        prompt_len = len(gen_ids)
        with no_grad():
            for _ in range(max_new_tokens):
                seq = np.array([gen_ids[-128:]], dtype=np.int64)
                x = tensor(seq, requires_grad=False)
                logits, h = lstm.forward(x, h)
                logits_arr = logits.data
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
    else:
        # Transformer path
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        result = net.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.8)
        elapsed = time.perf_counter() - t0
        text = decode(result.data)
        return text.strip(), elapsed, result.data.shape[1] - input_ids.shape[1]


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
            net, lstm, encode, decode, charset = load_native_model()
            # Auto-detect: if charset is small (<80 chars), use Shakespeare prompts
            if charset and len(charset) < 80:
                prompts_dict = SHAKESPEARE_PROMPTS
                prompts = list(prompts_dict.keys())
                print(f"  Using Shakespeare prompts (charset has {len(charset)} chars)")
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
