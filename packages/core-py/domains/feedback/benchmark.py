"""
Response Benchmark

Evaluates response quality: coherence, correctness, perplexity.
Offline analysis of logged responses.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger("man.benchmark")


@dataclass
class BenchmarkResult:
    """Benchmark run result."""
    timestamp: str
    model: str
    num_responses: int

    # Coherence metrics
    avg_length: float
    length_std: float

    # Repetition metrics
    repetition_rate: float
    repetition_bigrams: float

    # Perplexity estimate
    avg_log_prob: float

    # Diversity
    unique_bigrams: float
    unique_trigrams: float

    # Quality flags
    empty_rate: float
    truncation_rate: float

    # Overall score
    coherence_score: float
    quality_score: float


class ResponseBenchmark:
    """
    Benchmark response quality offline.

    Usage:
        bench = ResponseBenchmark()

        results = bench.run(
            responses=[
                {"user": "Hi", "assistant": "Hello!"},
                {"user": "What is AI?", "assistant": "AI is..."},
            ],
            model="gpt2",
        )

        # Results include:
        # - coherence_score: 0-1 based on repetition/diversity
        # - quality_score: 0-1 based on length/variety
    """

    def __init__(self):
        self.cache_dir = Path("data/response_logs")

    def run(
        self,
        responses: List[Dict[str, str]],
        model: str,
    ) -> BenchmarkResult:
        """Run benchmark on responses."""

        assistant_responses = [r.get("assistant", "") for r in responses]

        # Filter empty
        non_empty = [r for r in assistant_responses if r and r.strip()]
        empty_count = len(assistant_responses) - len(non_empty)
        empty_rate = empty_count / max(1, len(assistant_responses))

        # Length stats
        lengths = [len(r.split()) for r in non_empty]
        avg_length = sum(lengths) / max(1, len(lengths))
        length_std = (sum((l - avg_length) ** 2 for l in lengths) / max(1, len(lengths))) ** 0.5

        # Repetition (repeated n-grams)
        rep_count = 0
        bigram_count = 0
        unique_bigrams = set()

        for resp in non_empty:
            words = resp.lower().split()
            bigrams = set(f"{words[i]} {words[i+1]}" for i in range(len(words)-1))

            # Count repeats
            rep_count += len(bigrams) - len(set(bigrams))
            bigram_count += len(bigrams)
            unique_bigrams.update(bigrams)

        repetition_rate = rep_count / max(1, bigram_count)

        # Unique bigrams/trigrams ratio
        unique_bigrams_ratio = len(unique_bigrams) / max(1, bigram_count)

        # Trigrams
        unique_trigrams = set()
        for resp in non_empty:
            words = resp.lower().split()
            trigrams = set(f"{words[i]} {words[i+1]} {words[i+2]}"
                        for i in range(len(words)-2))
            unique_trigrams.update(trigrams)

        unique_trigrams_ratio = len(unique_trigrams) / max(1, len(unique_bigrams))

        # Truncation (response near max tokens)
        truncation_count = sum(1 for l in lengths if l > 100)
        truncation_rate = truncation_count / max(1, len(non_empty))

        # Estimate log prob (simple - based on repetition)
        # Lower repetition = higher log prob estimate
        avg_log_prob = -repetition_rate * 10

        # Coherence score (higher = more coherent)
        # Based on: low repetition, high diversity, reasonable length
        coherence_score = (
            (1 - repetition_rate) * 0.4 +
            unique_bigrams_ratio * 0.3 +
            (1 - min(avg_length / 100, 1)) * 0.3
        )

        # Quality score (higher = better)
        # Based on: non-empty, not truncated, good length
        quality_score = (
            (1 - empty_rate) * 0.3 +
            (1 - truncation_rate) * 0.3 +
            min(avg_length / 50, 1) * 0.4
        )

        return BenchmarkResult(
            timestamp=datetime.now().isoformat(),
            model=model,
            num_responses=len(assistant_responses),
            avg_length=avg_length,
            length_std=length_std,
            repetition_rate=repetition_rate,
            repetition_bigrams=rep_count,
            avg_log_prob=avg_log_prob,
            unique_bigrams=unique_bigrams_ratio,
            unique_trigrams=unique_trigrams_ratio,
            empty_rate=empty_rate,
            truncation_rate=truncation_rate,
            coherence_score=coherence_score,
            quality_score=quality_score,
        )

    def run_on_logs(
        self,
        model: Optional[str] = None,
        limit: int = 100,
    ) -> BenchmarkResult:
        """Run benchmark on logged responses."""
        # Load from log file
        log_file = self.cache_dir / f"responses_{datetime.now().strftime('%Y%m%d')}.jsonl"

        if not log_file.exists():
            logger.warning(f"No log file: {log_file}", extra={"tag": "INFRA"})
            return self.run([], model or "unknown")

        responses = []
        with open(log_file) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if model and data.get("model") != model:
                        continue
                    responses.append(data)
                except (json.JSONDecodeError, ValueError):
                    continue

        responses = responses[-limit:]

        return self.run(responses, model or responses[0].get("model", "unknown") if responses else "unknown")

    def compare_models(
        self,
        models: List[str],
    ) -> Dict[str, BenchmarkResult]:
        """Compare multiple models."""
        results = {}

        for model in models:
            result = self.run_on_logs(model=model)
            results[model] = result

        return results


def run_quick_benchmark(limit: int = 50) -> Dict[str, Any]:
    """Run quick benchmark on recent logs."""
    bench = ResponseBenchmark()
    result = bench.run_on_logs(limit=limit)

    return {
        "model": result.model,
        "num_responses": result.num_responses,
        "coherence_score": result.coherence_score,
        "quality_score": result.quality_score,
        "repetition_rate": result.repetition_rate,
        "avg_length": result.avg_length,
        "empty_rate": result.empty_rate,
        "truncation_rate": result.truncation_rate,
    }


__all__ = [
    "BenchmarkResult",
    "ResponseBenchmark",
    "run_quick_benchmark",
]
