"""
SloughGPT Performance Testing Suite
Tests inference performance with various models and configurations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import torch
import argparse
from typing import Dict, List, Optional
import statistics


def get_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def test_model_loading(model_name: str, device: str) -> Dict:
    """Test model loading time and memory."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # Test tokenizer loading
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_time = time.time() - start
    print(f"Tokenizer loaded in: {tokenizer_time:.2f}s")

    # Test model loading
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    load_time = time.time() - start
    print(f"Model loaded in: {load_time:.2f}s")

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params/1e9:.2f}B)")

    # Memory usage
    if device == "cuda":
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

    return {
        "model": model_name,
        "device": device,
        "tokenizer_time": tokenizer_time,
        "load_time": load_time,
        "parameters": params,
        "model": model,
        "tokenizer": tokenizer,
    }


def test_inference_latency(
    model: any,
    tokenizer: any,
    prompt: str,
    device: str,
    num_runs: int = 10,
    max_new_tokens: int = 50,
) -> Dict:
    """Test inference latency."""
    print(f"\n{'='*60}")
    print("Latency Test")
    print(f"{'='*60}")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_length = input_ids.shape[1]

    latencies = []

    for i in range(num_runs):
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.perf_counter()

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        torch.cuda.synchronize() if device == "cuda" else None
        elapsed = time.perf_counter() - start

        latencies.append(elapsed * 1000)  # Convert to ms

        if i == 0:
            generated = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"Sample output: {generated[:100]}...")

    # Calculate statistics
    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    avg = statistics.mean(latencies)
    std = statistics.stdev(latencies) if len(latencies) > 1 else 0

    print(f"Prompt length: {prompt_length} tokens")
    print(f"Generated: {max_new_tokens} tokens")
    print(f"Latency - Mean: {avg:.1f}ms, Std: {std:.1f}ms")
    print(f"Latency - P50: {p50:.1f}ms, P95: {p95:.1f}ms, P99: {p99:.1f}ms")

    return {
        "mean_ms": avg,
        "std_ms": std,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "prompt_tokens": prompt_length,
        "generated_tokens": max_new_tokens,
    }


def test_throughput(
    model: any,
    tokenizer: any,
    prompt: str,
    device: str,
    num_runs: int = 5,
    max_new_tokens: int = 100,
) -> Dict:
    """Test throughput (tokens per second)."""
    print(f"\n{'='*60}")
    print("Throughput Test")
    print(f"{'='*60}")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    throughputs = []

    for i in range(num_runs):
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.perf_counter()

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        torch.cuda.synchronize() if device == "cuda" else None
        elapsed = time.perf_counter() - start

        total_tokens = output.shape[1]
        tokens_per_sec = total_tokens / elapsed

        throughputs.append(tokens_per_sec)
        print(f"Run {i+1}: {tokens_per_sec:.1f} tokens/sec")

    avg_throughput = statistics.mean(throughputs)
    print(f"\nAverage throughput: {avg_throughput:.1f} tokens/sec")

    return {
        "tokens_per_sec": avg_throughput,
        "runs": num_runs,
        "max_new_tokens": max_new_tokens,
    }


def test_batch_inference(
    model: any,
    tokenizer: any,
    prompts: List[str],
    device: str,
    batch_sizes: List[int] = [1, 2, 4, 8],
) -> Dict:
    """Test batch inference performance."""
    print(f"\n{'='*60}")
    print("Batch Inference Test")
    print(f"{'='*60}")

    results = {}

    for batch_size in batch_sizes:
        if batch_size > len(prompts):
            batch_size = len(prompts)

        selected_prompts = prompts[:batch_size]

        # Tokenize
        inputs = tokenizer(
            selected_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        torch.cuda.synchronize() if device == "cuda" else None
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=20,
                do_sample=False,
            )

        torch.cuda.synchronize() if device == "cuda" else None
        elapsed = time.perf_counter() - start

        tokens_per_sec = outputs.numel() / elapsed
        print(f"Batch {batch_size}: {tokens_per_sec:.1f} tokens/sec ({elapsed*1000:.1f}ms)")

        results[batch_size] = {
            "tokens_per_sec": tokens_per_sec,
            "total_ms": elapsed * 1000,
        }

    return results


def test_quantization_impact(
    model_name: str,
    device: str,
) -> Dict:
    """Test performance with different quantization levels."""
    print(f"\n{'='*60}")
    print("Quantization Impact Test")
    print(f"{'='*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    results = {}

    for dtype_name, dtype in [
        ("FP32", torch.float32),
        ("FP16", torch.float16),
        ("BF16", torch.bfloat16),
    ]:
        try:
            print(f"\nTesting {dtype_name}...")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
            model = model.to(device)
            model.eval()

            # Quick benchmark
            input_ids = tokenizer.encode("Hello, world!", return_tensors="pt").to(device)

            torch.cuda.synchronize() if device == "cuda" else None
            start = time.perf_counter()

            with torch.no_grad():
                _ = model.model(input_ids)

            torch.cuda.synchronize() if device == "cuda" else None
            elapsed = time.perf_counter() - start

            # Memory
            if device == "cuda":
                memory = torch.cuda.memory_allocated() / 1e9
            else:
                memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9

            results[dtype_name] = {
                "latency_ms": elapsed * 1000,
                "memory_gb": memory,
            }

            print(f"  Latency: {elapsed*1000:.1f}ms, Memory: {memory:.2f}GB")

            del model
            torch.cuda.empty_cache() if device == "cuda" else None

        except Exception as e:
            print(f"  Skipped {dtype_name}: {e}")
            results[dtype_name] = {"error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(description="SloughGPT Performance Testing")
    parser.add_argument("--model", "-m", default="gpt2", help="Model to test")
    parser.add_argument("--device", "-d", default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--test", "-t", default="all",
                        choices=["all", "latency", "throughput", "batch", "quantization"],
                        help="Test to run")
    parser.add_argument("--runs", "-r", type=int, default=10, help="Number of test runs")
    parser.add_argument("--tokens", "-k", type=int, default=50, help="Max new tokens")
    parser.add_argument("--prompt", "-p", default="The quick brown fox jumps over the lazy dog",
                        help="Test prompt")

    args = parser.parse_args()

    device = args.device if args.device != "auto" else get_device()

    print(f"\nSloughGPT Performance Test Suite")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    # Load model
    result = test_model_loading(args.model, device)
    model = result["model"]
    tokenizer = result["tokenizer"]

    prompts = [
        "Hello, how are you today?",
        "What is the meaning of life?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about programming.",
        "What are the benefits of exercise?",
    ]

    # Run selected tests
    if args.test in ["all", "latency"]:
        test_inference_latency(
            model, tokenizer, args.prompt, device,
            num_runs=args.runs, max_new_tokens=args.tokens
        )

    if args.test in ["all", "throughput"]:
        test_throughput(
            model, tokenizer, args.prompt, device,
            num_runs=args.runs, max_new_tokens=args.tokens
        )

    if args.test in ["all", "batch"]:
        test_batch_inference(
            model, tokenizer, prompts, device,
            batch_sizes=[1, 2, 4]
        )

    if args.test in ["all", "quantization"]:
        test_quantization_impact(args.model, device)

    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
