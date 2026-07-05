"""
Benchmarking Module
Model evaluation and performance metrics.
"""

import time
from domains.training.slonet_compat import torch
F = torch.F
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import statistics


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    model_name: str
    num_parameters: int
    memory_mb: float
    inference_time_ms: float
    throughput_tokens_per_sec: float
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    accuracy: Optional[float] = None
    perplexity: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "num_parameters": self.num_parameters,
            "memory_mb": round(self.memory_mb, 2),
            "inference_time_ms": round(self.inference_time_ms, 2),
            "throughput_tokens_per_sec": round(self.throughput_tokens_per_sec, 2),
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "accuracy": self.accuracy,
            "perplexity": self.perplexity,
        }


class Benchmarker:
    """Benchmark model performance."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any = None,
        device: str = "cpu",
        warmup_steps: int = 3,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.warmup_steps = warmup_steps

        self.model.eval()

    def count_parameters(self) -> int:
        """Count model parameters."""
        return sum(p.numel() for p in self.model.parameters())

    def measure_memory(self) -> float:
        """Measure memory usage in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)

    def benchmark_inference(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        num_runs: int = 10,
    ) -> BenchmarkResult:
        """Benchmark inference performance."""

        # Warmup
        for _ in range(self.warmup_steps):
            if self.tokenizer:
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    _ = self.model(input_ids)

        # Benchmark
        latencies = []
        total_tokens = 0

        for _ in range(num_runs):
            start_time = time.time()

            if self.tokenizer:
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    for _ in range(max_new_tokens):
                        outputs = self.model(input_ids)
                        logits = outputs.logits[:, -1, :] if hasattr(outputs, 'logits') else outputs[:, -1, :]
                        next_token = logits.argmax(dim=-1, keepdim=True)
                        input_ids = torch.cat([input_ids, next_token], dim=1)

                generated_tokens = input_ids.shape[1]
            else:
                generated_tokens = max_new_tokens

            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
            total_tokens += generated_tokens

        total_time = sum(latencies)
        throughput = (total_tokens / total_time) * 1000 if total_time > 0 else 0

        sorted_latencies = sorted(latencies)
        p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        return BenchmarkResult(
            model_name=getattr(self.model, 'name', 'unknown'),
            num_parameters=self.count_parameters(),
            memory_mb=self.measure_memory(),
            inference_time_ms=statistics.mean(latencies),
            throughput_tokens_per_sec=throughput,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
        )

    def benchmark_batch(
        self,
        prompts: List[str],
        batch_size: int = 8,
        max_new_tokens: int = 50,
    ) -> Dict[str, float]:
        """Benchmark batch inference."""

        if not self.tokenizer:
            return {"error": "Tokenizer required for batch benchmarking"}

        # Tokenize all prompts
        input_ids_list = []
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            input_ids_list.append(input_ids)

        # Pad to same length
        max_len = max(ids.shape[1] for ids in input_ids_list)
        padded = []
        attention_mask = []

        for ids in input_ids_list:
            pad_len = max_len - ids.shape[1]
            padded_ids = torch.cat([
                torch.full((1, pad_len), self.tokenizer.pad_token_id, device=self.device),
                ids
            ], dim=1)
            padded.append(padded_ids)
            attention_mask.append(torch.cat([
                torch.zeros(1, pad_len, device=self.device),
                torch.ones(1, ids.shape[1], device=self.device)
            ], dim=1))

        batch_input = torch.cat(padded, dim=0)
        batch_mask = torch.cat(attention_mask, dim=0)

        # Warmup
        for _ in range(2):
            with torch.no_grad():
                _ = self.model(batch_input, attention_mask=batch_mask)

        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(batch_input, attention_mask=batch_mask)
                logits = outputs.logits[:, -1, :] if hasattr(outputs, 'logits') else outputs[:, -1, :]
                next_tokens = logits.argmax(dim=-1, keepdim=True)
                batch_input = torch.cat([batch_input, next_tokens], dim=1)
                batch_mask = torch.cat([
                    batch_mask,
                    torch.ones(batch_size, 1, device=self.device)
                ], dim=1)

        total_time = time.time() - start_time
        total_output_tokens = batch_size * max_new_tokens
        throughput = total_output_tokens / total_time

        return {
            "batch_size": batch_size,
            "num_prompts": len(prompts),
            "total_time_sec": round(total_time, 3),
            "throughput_tokens_per_sec": round(throughput, 2),
            "avg_latency_per_batch_ms": round((total_time / max_new_tokens) * 1000, 2),
        }

    def calculate_perplexity(
        self,
        text: str,
        block_size: int = 512,
    ) -> float:
        """Calculate perplexity on text."""
        if not self.tokenizer:
            return None

        encodings = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=block_size)
        input_ids = encodings.input_ids.to(self.device)

        total_loss = 0.0
        num_tokens = 0

        with torch.no_grad():
            for i in range(0, input_ids.shape[1] - 1, block_size):
                chunk = input_ids[:, i:i + block_size + 1]
                outputs = self.model(chunk)

                logits = outputs.logits[:, :-1, :] if hasattr(outputs, 'logits') else outputs[:, :-1, :]
                targets = chunk[:, 1:]

                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction='mean',
                    ignore_index=self.tokenizer.pad_token_id
                )

                total_loss += loss.item() * (chunk.shape[1] - 1)
                num_tokens += chunk.shape[1] - 1

        avg_loss = total_loss / num_tokens if num_tokens > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return perplexity


def benchmark_model(
    model: torch.nn.Module,
    tokenizer: Any = None,
    device: str = "cpu",
    prompt: str = "The quick brown fox jumps over the lazy dog",
    max_new_tokens: int = 50,
) -> BenchmarkResult:
    """Quick benchmark function."""
    benchmarker = Benchmarker(model, tokenizer, device)
    return benchmarker.benchmark_inference(prompt, max_new_tokens)


def compare_models(
    models: Dict[str, torch.nn.Module],
    tokenizer: Any = None,
    device: str = "cpu",
    prompt: str = "Hello world",
) -> List[BenchmarkResult]:
    """Compare multiple models."""
    results = []
    for name, model in models.items():
        model.name = name
        benchmarker = Benchmarker(model, tokenizer, device)
        result = benchmarker.benchmark_inference(prompt)
        results.append(result)
    return results


__all__ = [
    "BenchmarkResult",
    "Benchmarker",
    "benchmark_model",
    "compare_models",
]
