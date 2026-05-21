#!/usr/bin/env python3
"""
SloughGPT Benchmark Suite
Measures model performance and training throughput.
"""

import time
import torch
import torch.nn as nn
from typing import Dict, Any, Callable
from dataclasses import dataclass
import json


@dataclass
class BenchmarkResult:
    """Result of a benchmark."""
    name: str
    value: float
    unit: str
    metadata: Dict[str, Any] = None


class BenchmarkSuite:
    """Benchmark suite for SloughGPT."""
    
    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device
        self.results = []
    
    def benchmark_forward(self, batch_size: int = 1, seq_len: int = 128, iterations: int = 100) -> BenchmarkResult:
        """Benchmark forward pass."""
        vocab_size = self.model.vocab_size
        
        # Warmup
        x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(self.device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len)).to(self.device)
        
        for _ in range(5):
            with torch.no_grad():
                _, _ = self.model(x, y)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start = time.time()
        for _ in range(iterations):
            with torch.no_grad():
                _, _ = self.model(x, y)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        
        ms_per_iter = (elapsed / iterations) * 1000
        tokens_per_sec = (batch_size * seq_len * iterations) / elapsed
        
        return BenchmarkResult(
            name="forward_pass",
            value=ms_per_iter,
            unit="ms",
            metadata={
                "batch_size": batch_size,
                "seq_len": seq_len,
                "tokens_per_sec": tokens_per_sec
            }
        )
    
    def benchmark_backward(self, batch_size: int = 1, seq_len: int = 128, iterations: int = 50) -> BenchmarkResult:
        """Benchmark forward + backward pass."""
        vocab_size = self.model.vocab_size
        
        x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(self.device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len)).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        # Warmup
        for _ in range(3):
            optimizer.zero_grad()
            _, loss = self.model(x, y)
            loss.backward()
        
        # Benchmark
        start = time.time()
        for _ in range(iterations):
            optimizer.zero_grad()
            _, loss = self.model(x, y)
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        
        ms_per_iter = (elapsed / iterations) * 1000
        
        return BenchmarkResult(
            name="forward_backward",
            value=ms_per_iter,
            unit="ms",
            metadata={
                "batch_size": batch_size,
                "seq_len": seq_len,
                "iterations": iterations
            }
        )
    
    def benchmark_generation(self, prompt_len: int = 10, max_new_tokens: int = 100, iterations: int = 10) -> BenchmarkResult:
        """Benchmark text generation."""
        vocab_size = self.model.vocab_size
        
        # Warmup
        prompt = torch.randint(0, vocab_size, (1, prompt_len)).to(self.device)
        for _ in range(3):
            with torch.no_grad():
                _ = self.model.generate(prompt, max_new_tokens=10)
        
        # Benchmark
        start = time.time()
        total_tokens = 0
        for _ in range(iterations):
            prompt = torch.randint(0, vocab_size, (1, prompt_len)).to(self.device)
            with torch.no_grad():
                output = self.model.generate(prompt, max_new_tokens=max_new_tokens)
            total_tokens += output.shape[1]
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        
        tokens_per_sec = total_tokens / elapsed
        ms_per_token = (elapsed / total_tokens) * 1000
        
        return BenchmarkResult(
            name="generation",
            value=ms_per_token,
            unit="ms/token",
            metadata={
                "prompt_len": prompt_len,
                "max_new_tokens": max_new_tokens,
                "total_tokens": total_tokens,
                "tokens_per_sec": tokens_per_sec
            }
        )
    
    def benchmark_memory(self) -> BenchmarkResult:
        """Benchmark memory usage."""
        if not torch.cuda.is_available():
            return BenchmarkResult(
                name="memory",
                value=0,
                unit="MB",
                metadata={"note": "CUDA not available"}
            )
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        vocab_size = self.model.vocab_size
        x = torch.randint(0, vocab_size, (4, 128)).to("cuda")
        y = torch.randint(0, vocab_size, (4, 128)).to("cuda")
        
        _, loss = self.model(x, y)
        loss.backward()
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        return BenchmarkResult(
            name="peak_memory",
            value=peak_memory,
            unit="MB",
            metadata={"batch_size": 4, "seq_len": 128}
        )
    
    def run_all(self) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks."""
        results = {}
        
        print("Running benchmarks...")
        
        print("  Forward pass...")
        results["forward"] = self.benchmark_forward()
        
        print("  Forward + Backward...")
        results["forward_backward"] = self.benchmark_backward()
        
        print("  Generation...")
        results["generation"] = self.benchmark_generation()
        
        print("  Memory...")
        results["memory"] = self.benchmark_memory()
        
        self.results = results
        return results
    
    def print_results(self):
        """Print benchmark results."""
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)
        
        for name, result in self.results.items():
            print(f"\n{name}: {result.value:.2f} {result.unit}")
            if result.metadata:
                for k, v in result.metadata.items():
                    print(f"  {k}: {v}")
        
        print("\n" + "=" * 50)
    
    def save_results(self, path: str = "benchmark_results.json"):
        """Save results to JSON."""
        data = {
            name: {
                "value": result.value,
                "unit": result.unit,
                "metadata": result.metadata or {}
            }
            for name, result in self.results.items()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {path}")


def benchmark_model(checkpoint_path: str, device: str = "cpu"):
    """Benchmark a trained model."""
    from domains.inference.loader import load_model

    print(f"Loading model from {checkpoint_path}...")
    engine = load_model(checkpoint_path, quantize=None)
    
    suite = BenchmarkSuite(engine.model, device=device)
    results = suite.run_all()
    suite.print_results()
    suite.save_results()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SloughGPT benchmarks")
    parser.add_argument("--checkpoint", default="models/sloughgpt.pt", help="Model checkpoint path")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda)")
    
    args = parser.parse_args()
    
    benchmark_model(args.checkpoint, args.device)
