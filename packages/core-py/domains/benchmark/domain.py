"""
Benchmark Domain - Clean quality metrics

Simple: log responses → calculate metrics
"""

from typing import Dict, Any, List
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class QualityMetrics:
    coherence_score: float = 0.0
    quality_score: float = 0.0
    repetition_rate: float = 0.0
    avg_length: float = 0.0
    empty_rate: float = 0.0


class BenchmarkDomain:
    """
    Clean benchmark domain - measures response quality.
    
    Usage:
        bench = BenchmarkDomain()
        
        # Calculate quality on responses
        metrics = bench.evaluate_responses(responses)
        
        # Get stats
        stats = bench.get_stats()
    """
    
    def __init__(self, log_dir: str = "data/response_logs"):
        self.log_dir = Path(log_dir)
    
    def evaluate_responses(self, responses: List[Dict[str, str]]) -> QualityMetrics:
        """Evaluate quality of responses."""
        if not responses:
            return QualityMetrics()
        
        assistant_responses = [r.get("assistant", "") for r in responses]
        
        # Filter empty
        non_empty = [r for r in assistant_responses if r and r.strip()]
        empty_count = len(assistant_responses) - len(non_empty)
        empty_rate = empty_count / max(1, len(assistant_responses))
        
        # Length stats
        lengths = [len(r.split()) for r in non_empty]
        avg_length = sum(lengths) / max(1, len(lengths))
        
        # Repetition (simple bigram check)
        rep_count = 0
        bigram_total = 0
        unique_bigrams = set()
        
        for resp in non_empty:
            words = resp.lower().split()
            for i in range(len(words)-1):
                bigram = f"{words[i]} {words[i+1]}"
                bigram_total += 1
                if bigram in unique_bigrams:
                    rep_count += 1
                unique_bigrams.add(bigram)
        
        repetition_rate = rep_count / max(1, bigram_total)
        
        # Coherence score (higher = more coherent)
        # Based on low repetition and reasonable length
        coherence_score = (
            (1 - repetition_rate) * 0.5 +
            min(avg_length / 50, 1) * 0.5
        )
        
        # Quality score (higher = better)
        quality_score = (
            (1 - empty_rate) * 0.5 +
            min(avg_length / 30, 1) * 0.5
        )
        
        return QualityMetrics(
            coherence_score=round(coherence_score, 2),
            quality_score=round(quality_score, 2),
            repetition_rate=round(repetition_rate, 2),
            avg_length=round(avg_length, 1),
            empty_rate=round(empty_rate, 2),
        )
    
    def evaluate_latest(self, limit: int = 50) -> Dict[str, Any]:
        """Evaluate recent logged responses."""
        responses = self._load_responses(limit)
        
        if not responses:
            return {"status": "no_data", "message": "No responses logged"}
        
        metrics = self.evaluate_responses(responses)
        
        return {
            "status": "ok",
            "total_responses": len(responses),
            "coherence_score": metrics.coherence_score,
            "quality_score": metrics.quality_score,
            "repetition_rate": metrics.repetition_rate,
            "avg_length": metrics.avg_length,
            "empty_rate": metrics.empty_rate,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get simple stats from logged data."""
        responses = self._load_responses(1000)
        
        if not responses:
            return {"total": 0}
        
        return {
            "total": len(responses),
            "avg_tokens": sum(r.get("tokens_generated", 0) for r in responses) / len(responses),
            "models": list(set(r.get("model") for r in responses)),
        }
    
    def _load_responses(self, limit: int) -> List[Dict[str, Any]]:
        """Load responses from log file."""
        import datetime
        responses = []
        
        log_file = self.log_dir / f"responses_{datetime.datetime.now().strftime('%Y%m%d')}.jsonl"
        
        if not log_file.exists():
            return responses
        
        with open(log_file) as f:
            for line in f:
                try:
                    responses.append(json.loads(line))
                except:
                    continue
        
        return responses[-limit:]


# Global instance
_benchmark_domain: BenchmarkDomain = None


def get_benchmark_domain() -> BenchmarkDomain:
    """Get global benchmark domain."""
    global _benchmark_domain
    if _benchmark_domain is None:
        _benchmark_domain = BenchmarkDomain()
    return _benchmark_domain


__all__ = [
    "QualityMetrics",
    "BenchmarkDomain", 
    "get_benchmark_domain",
]