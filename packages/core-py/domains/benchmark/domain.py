"""
BenchmarkDomain — response quality tracking and evaluation.

Provides BenchmarkDomain for coherence scoring, repetition detection,
perplexity estimation, and response history management.
"""

import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger("slo.benchmark")

_RESPONSES_DIR = Path(__file__).resolve().parents[3] / "data" / "logged_responses"

# Global singleton
_benchmark_domain: Optional["BenchmarkDomain"] = None


def get_benchmark_domain() -> "BenchmarkDomain":
    global _benchmark_domain
    if _benchmark_domain is None:
        _benchmark_domain = BenchmarkDomain()
    return _benchmark_domain


def reset_benchmark_domain() -> None:
    global _benchmark_domain
    _benchmark_domain = None


@dataclass
class BenchmarkResult:
    timestamp: str
    model: str
    num_responses: int
    avg_length: float
    length_std: float
    repetition_rate: float
    repetition_bigrams: float
    avg_log_prob: float
    unique_bigrams: float
    unique_trigrams: float


class BenchmarkDomain:
    """Response quality tracking and benchmark evaluation."""

    def __init__(self) -> None:
        self._responses_dir = _RESPONSES_DIR
        self._responses_dir.mkdir(parents=True, exist_ok=True)

    def _load_responses(self) -> list[dict]:
        """Load all logged responses from disk."""
        responses: list[dict] = []
        if not self._responses_dir.exists():
            return responses
        for f in sorted(self._responses_dir.iterdir()):
            if f.suffix == ".json":
                try:
                    data = json.loads(f.read_text())
                    if isinstance(data, list):
                        responses.extend(data)
                    else:
                        responses.append(data)
                except Exception as exc:
                    logger.warning("Failed to load %s: %s", f.name, exc, extra={"tag": "BENCH"})
        return responses

    def get_stats(self) -> dict:
        """Return aggregate statistics about logged responses."""
        responses = self._load_responses()
        if not responses:
            return {"total_responses": 0, "models": [], "avg_length": 0}
        models = list({r.get("model", "unknown") for r in responses if isinstance(r, dict)})
        lengths = [len(r.get("text", "")) for r in responses if isinstance(r, dict)]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        return {
            "total_responses": len(responses),
            "models": models,
            "avg_length": round(avg_len, 1),
        }

    def evaluate_latest(self, limit: int = 50) -> dict:
        """Evaluate quality of the most recent responses."""
        responses = self._load_responses()
        recent = responses[-limit:] if len(responses) > limit else responses
        if not recent:
            return {"responses_analyzed": 0, "metrics": {}}

        texts = [r.get("text", "") for r in recent if isinstance(r, dict)]
        lengths = [len(t) for t in texts]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        std_len = (sum((l - avg_len) ** 2 for l in lengths) / len(lengths)) ** 0.5 if lengths else 0

        # Repetition detection
        total_bigrams = 0
        repeated_bigrams = 0
        seen_bigrams: set[tuple[str, ...]] = set()
        all_bigrams: set[tuple[str, ...]] = set()
        for t in texts:
            words = t.lower().split()
            for i in range(len(words) - 1):
                bg = (words[i], words[i + 1])
                all_bigrams.add(bg)
                total_bigrams += 1
                if bg in seen_bigrams:
                    repeated_bigrams += 1
                seen_bigrams.add(bg)

        rep_rate = repeated_bigrams / total_bigrams if total_bigrams else 0
        unique_bigram_ratio = len(all_bigrams) / total_bigrams if total_bigrams else 0

        return {
            "responses_analyzed": len(texts),
            "metrics": {
                "avg_length": round(avg_len, 1),
                "length_std": round(std_len, 1),
                "repetition_rate": round(rep_rate, 4),
                "unique_bigram_ratio": round(unique_bigram_ratio, 4),
            },
        }

    def clear_history(self) -> None:
        """Delete all logged response files."""
        if self._responses_dir.exists():
            import shutil
            shutil.rmtree(self._responses_dir)
            self._responses_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared benchmark response history", extra={"tag": "BENCH"})
