"""Training Outcome Tracker — records every training run's config and results.

Stores outcomes as JSONL so the adaptive engine can learn from history.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("slo.training.outcome_tracker")

DEFAULT_HISTORY_PATH = Path.home() / ".config" / "sloughgpt" / "training_history.jsonl"


@dataclass
class TrainingOutcome:
    """A single training run's config and results."""

    # Identity
    run_id: str = ""
    timestamp: float = 0.0
    dataset: str = ""
    dataset_size: int = 0
    dataset_format: str = ""

    # Config used
    model: str = ""
    method: str = ""
    epochs: int = 0
    batch_size: int = 0
    learning_rate: float = 0.0
    max_seq_length: int = 0
    warmup_steps: int = 0
    weight_decay: float = 0.0
    use_lora: bool = False
    lora_rank: int = 0
    lora_alpha: int = 0

    # Results
    final_loss: float = 0.0
    best_loss: float = 0.0
    perplexity: float = 0.0
    bleu_score: float = 0.0
    training_time_s: float = 0.0
    converged: bool = False
    early_stopped: bool = False

    # Derived quality score (0-1, higher = better)
    quality_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TrainingOutcome:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class TrainingOutcomeTracker:
    """Records and queries training outcomes for adaptive learning.

    Outcomes are stored as JSONL (one JSON object per line) so they can
    be appended efficiently and read incrementally.
    """

    def __init__(self, history_path: Optional[Path] = None):
        self.history_path = history_path or DEFAULT_HISTORY_PATH
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, outcome: TrainingOutcome) -> None:
        """Append a training outcome to history."""
        if not outcome.run_id:
            outcome.run_id = f"run_{int(outcome.timestamp * 1000)}"
        if outcome.timestamp == 0.0:
            outcome.timestamp = time.time()

        # Compute quality score
        outcome.quality_score = self._compute_quality(outcome)

        with open(self.history_path, "a") as f:
            f.write(json.dumps(outcome.to_dict()) + "\n")

        logger.info(
            "Recorded training outcome: run=%s quality=%.3f loss=%.4f",
            outcome.run_id, outcome.quality_score, outcome.final_loss,
        )

    def load_outcomes(self) -> List[TrainingOutcome]:
        """Load all recorded outcomes."""
        if not self.history_path.exists():
            return []
        outcomes = []
        with open(self.history_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    outcomes.append(TrainingOutcome.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError):
                    continue
        return outcomes

    def get_best_outcomes(self, n: int = 10) -> List[TrainingOutcome]:
        """Get the top N outcomes by quality score."""
        outcomes = self.load_outcomes()
        outcomes.sort(key=lambda o: o.quality_score, reverse=True)
        return outcomes[:n]

    def get_outcomes_for_dataset(self, dataset: str) -> List[TrainingOutcome]:
        """Get all outcomes for a specific dataset."""
        return [o for o in self.load_outcomes() if o.dataset == dataset]

    def get_outcomes_for_model(self, model: str) -> List[TrainingOutcome]:
        """Get all outcomes for a specific model."""
        return [o for o in self.load_outcomes() if o.model == model]

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics of all outcomes."""
        outcomes = self.load_outcomes()
        if not outcomes:
            return {"total_runs": 0}

        qualities = [o.quality_score for o in outcomes]
        losses = [o.final_loss for o in outcomes if o.final_loss > 0]
        return {
            "total_runs": len(outcomes),
            "avg_quality": sum(qualities) / len(qualities),
            "best_quality": max(qualities),
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
            "converged_count": sum(1 for o in outcomes if o.converged),
            "early_stopped_count": sum(1 for o in outcomes if o.early_stopped),
        }

    def clear(self) -> int:
        """Clear all history. Returns number of entries removed."""
        count = len(self.load_outcomes())
        if self.history_path.exists():
            self.history_path.unlink()
        return count

    @staticmethod
    def _compute_quality(outcome: TrainingOutcome) -> float:
        """Compute a 0-1 quality score from training results.

        Factors:
        - Lower loss is better (normalized to 0-1)
        - Converged is better
        - Early stopping means wasted epochs
        - Training time efficiency matters
        """
        score = 0.0

        # Loss component (lower = better, assume 0-10 range)
        if outcome.final_loss > 0:
            loss_score = max(0.0, 1.0 - outcome.final_loss / 10.0)
            score += loss_score * 0.5

        # Convergence bonus
        if outcome.converged:
            score += 0.3

        # Early stopping penalty (wasted compute)
        if outcome.early_stopped:
            score -= 0.1

        # Perplexity component (lower = better, assume 1-100 range)
        if outcome.perplexity > 0:
            ppl_score = max(0.0, 1.0 - outcome.perplexity / 100.0)
            score += ppl_score * 0.2

        return max(0.0, min(1.0, score))


__all__ = ["TrainingOutcome", "TrainingOutcomeTracker", "DEFAULT_HISTORY_PATH"]
