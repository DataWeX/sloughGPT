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

    # User annotations
    tags: List[str] = field(default_factory=list)
    notes: str = ""

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

    def update_run(self, run_id: str, **kwargs) -> Optional[TrainingOutcome]:
        """Update a training run's tags, notes, or other mutable fields."""
        outcomes = self.load_outcomes()
        target = None
        for o in outcomes:
            if o.run_id == run_id:
                target = o
                break
        if not target:
            return None
        for key, value in kwargs.items():
            if hasattr(target, key):
                setattr(target, key, value)
        # Rewrite history
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "w") as f:
            for o in outcomes:
                f.write(json.dumps(o.to_dict()) + "\n")
        return target

    def add_tag(self, run_id: str, tag: str) -> Optional[TrainingOutcome]:
        """Add a tag to a training run."""
        outcomes = self.load_outcomes()
        target = next((o for o in outcomes if o.run_id == run_id), None)
        if not target:
            return None
        if tag not in target.tags:
            target.tags.append(tag)
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_path, "w") as f:
                for o in outcomes:
                    f.write(json.dumps(o.to_dict()) + "\n")
        return target

    def remove_tag(self, run_id: str, tag: str) -> Optional[TrainingOutcome]:
        """Remove a tag from a training run."""
        outcomes = self.load_outcomes()
        target = next((o for o in outcomes if o.run_id == run_id), None)
        if not target:
            return None
        if tag in target.tags:
            target.tags.remove(tag)
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_path, "w") as f:
                for o in outcomes:
                    f.write(json.dumps(o.to_dict()) + "\n")
        return target

    def set_notes(self, run_id: str, notes: str) -> Optional[TrainingOutcome]:
        """Set notes on a training run."""
        return self.update_run(run_id, notes=notes)

    def get_outcomes_by_tag(self, tag: str) -> List[TrainingOutcome]:
        """Get all outcomes with a specific tag."""
        return [o for o in self.load_outcomes() if tag in o.tags]

    def get_all_tags(self) -> List[str]:
        """Get all unique tags across all runs."""
        tags = set()
        for o in self.load_outcomes():
            tags.update(o.tags)
        return sorted(tags)

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

    def export_json(self, limit: int = 0) -> List[Dict[str, Any]]:
        """Export outcomes as a list of dicts (JSON-serializable)."""
        outcomes = self.load_outcomes()
        if limit > 0:
            outcomes = outcomes[-limit:]
        return [o.to_dict() for o in outcomes]

    def export_csv(self, limit: int = 0) -> str:
        """Export outcomes as CSV string."""
        outcomes = self.load_outcomes()
        if limit > 0:
            outcomes = outcomes[-limit:]
        if not outcomes:
            return ""
        fields = list(outcomes[0].to_dict().keys())
        lines = [",".join(fields)]
        for o in outcomes:
            row = o.to_dict()
            lines.append(",".join(str(row.get(f, "")).replace(",", ";") for f in fields))
        return "\n".join(lines)

    def compare(self, run_id_a: str, run_id_b: str) -> Optional[Dict[str, Any]]:
        """Compare two training runs side by side."""
        outcomes = self.load_outcomes()
        a = next((o for o in outcomes if o.run_id == run_id_a), None)
        b = next((o for o in outcomes if o.run_id == run_id_b), None)
        if not a or not b:
            return None

        diff = {}
        for field_name in TrainingOutcome.__dataclass_fields__:
            va = getattr(a, field_name)
            vb = getattr(b, field_name)
            if va != vb:
                diff[field_name] = {"run_a": va, "run_b": vb}

        return {
            "run_a": a.to_dict(),
            "run_b": b.to_dict(),
            "differences": diff,
            "a_wins": sum(1 for k, v in diff.items() if k in ("final_loss", "best_loss", "perplexity", "training_time_s") and v["run_a"] < v["run_b"]) + (1 if a.quality_score > b.quality_score else 0),
            "b_wins": sum(1 for k, v in diff.items() if k in ("final_loss", "best_loss", "perplexity", "training_time_s") and v["run_a"] > v["run_b"]) + (1 if b.quality_score > a.quality_score else 0),
        }


__all__ = ["TrainingOutcome", "TrainingOutcomeTracker", "DEFAULT_HISTORY_PATH"]
