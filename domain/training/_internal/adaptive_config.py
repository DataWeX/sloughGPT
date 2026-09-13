"""Adaptive Config Engine — learns from past training outcomes to recommend better configs.

Replaces the static lookup tables in auto_config.py with a system that
improves over time as more training runs are recorded.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from domains.training.outcome_tracker import TrainingOutcome, TrainingOutcomeTracker

logger = logging.getLogger("slo.training.adaptive")


@dataclass
class AdaptiveRecommendation:
    """Config recommendation backed by historical evidence."""

    # Recommended values
    learning_rate: float = 2e-4
    batch_size: int = 4
    epochs: int = 3
    warmup_steps: int = 50
    lora_rank: int = 8
    lora_alpha: int = 16

    # Confidence & reasoning
    confidence: float = 0.5
    based_on_runs: int = 0
    reason: str = ""

    # What similar runs achieved
    avg_quality: float = 0.0
    avg_loss: float = 0.0
    best_quality: float = 0.0


class AdaptiveConfigEngine:
    """Learns from training history to recommend better configs.

    The engine maintains a simple but effective model:
    - For each (dataset_size_category, model, method) tuple, it tracks
      the best-performing configs and uses weighted averages to recommend.
    - It uses exponential moving averages so recent runs matter more.
    - It explores by occasionally suggesting slightly different configs
      when confidence is low.
    """

    def __init__(self, tracker: Optional[TrainingOutcomeTracker] = None):
        self.tracker = tracker or TrainingOutcomeTracker()

    def recommend(
        self,
        dataset_size: int,
        model: str = "gpt2",
        method: str = "finetune",
        available_models: Optional[List[str]] = None,
    ) -> AdaptiveRecommendation:
        """Recommend training config based on historical outcomes.

        Args:
            dataset_size: Number of training examples
            model: Preferred model ID
            method: Training method
            available_models: List of available models
        """
        size_cat = self._size_category(dataset_size)
        outcomes = self.tracker.load_outcomes()

        # Find similar runs
        similar = [
            o for o in outcomes
            if self._size_category(o.dataset_size) == size_cat
            and o.model == model
            and o.method == method
        ]

        if not similar:
            # Try relaxing to same model regardless of method
            similar = [
                o for o in outcomes
                if self._size_category(o.dataset_size) == size_cat
                and o.model == model
            ]

        if not similar:
            # Try relaxing to same method regardless of model
            similar = [
                o for o in outcomes
                if self._size_category(o.dataset_size) == size_cat
                and o.method == method
            ]

        if not similar:
            return self._fallback_recommendation(dataset_size, model, method)

        return self._recommend_from_history(similar, dataset_size, model, method)

    def suggest_variation(
        self, base: AdaptiveRecommendation, iteration: int
    ) -> AdaptiveRecommendation:
        """Suggest a slight variation for exploration.

        When confidence is low, we explore by tweaking one parameter at a time.
        This is how the system learns which direction is better.
        """
        import copy
        varied = copy.deepcopy(base)

        # Only explore when confidence is low
        if base.confidence > 0.8:
            return varied

        # Cycle through parameters to explore
        param = iteration % 4
        if param == 0:
            # Vary learning rate by ±20%
            factor = 1.2 if iteration % 8 < 4 else 0.8
            varied.learning_rate *= factor
        elif param == 1:
            # Vary batch size by ±1 step
            varied.batch_size = max(2, varied.batch_size + (1 if iteration % 6 < 3 else -1))
        elif param == 2:
            # Vary lora rank
            varied.lora_rank = max(2, varied.lora_rank + (2 if iteration % 6 < 3 else -2))
        else:
            # Vary epochs
            varied.epochs = max(1, varied.epochs + (1 if iteration % 6 < 3 else -1))

        varied.confidence *= 0.8  # Lower confidence for variations
        varied.reason = f"Exploration variation #{iteration}"
        return varied

    def update_after_run(self, outcome: TrainingOutcome) -> None:
        """Record a training outcome and update internal state.

        This is the feedback loop — after each run, the system learns
        what worked and what didn't.
        """
        self.tracker.record(outcome)

        # Log insight
        stats = self.tracker.get_stats()
        logger.info(
            "Adaptive engine updated: total_runs=%d avg_quality=%.3f",
            stats.get("total_runs", 0),
            stats.get("avg_quality", 0.0),
            extra={"tag": "TRAIN"},
        )

    def get_insights(self, model: str = "", method: str = "") -> Dict[str, Any]:
        """Get insights from training history for display in UI."""
        outcomes = self.tracker.load_outcomes()
        if not outcomes:
            return {"message": "No training history yet. Run a training to start learning."}

        if model:
            outcomes = [o for o in outcomes if o.model == model]
        if method:
            outcomes = [o for o in outcomes if o.method == method]

        if not outcomes:
            return {"message": f"No history for model={model} method={method}"}

        # Find best config
        best = max(outcomes, key=lambda o: o.quality_score)
        avg_loss = sum(o.final_loss for o in outcomes if o.final_loss > 0) / max(1, len(outcomes))
        avg_quality = sum(o.quality_score for o in outcomes) / len(outcomes)

        # Trend: compare recent vs older runs
        mid = len(outcomes) // 2
        if mid > 0:
            older = outcomes[:mid]
            newer = outcomes[mid:]
            older_avg = sum(o.quality_score for o in older) / len(older)
            newer_avg = sum(o.quality_score for o in newer) / len(newer)
            trend = "improving" if newer_avg > older_avg else "declining"
        else:
            trend = "insufficient data"

        return {
            "total_runs": len(outcomes),
            "avg_quality": round(avg_quality, 3),
            "avg_loss": round(avg_loss, 4),
            "best_quality": round(best.quality_score, 3),
            "best_config": {
                "learning_rate": best.learning_rate,
                "batch_size": best.batch_size,
                "epochs": best.epochs,
                "lora_rank": best.lora_rank,
            },
            "trend": trend,
            "recommendation": self.recommend(
                best.dataset_size, best.model, best.method
            ).reason,
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    def _recommend_from_history(
        self,
        similar: List[TrainingOutcome],
        dataset_size: int,
        model: str,
        method: str,
    ) -> AdaptiveRecommendation:
        """Build recommendation from similar past runs using weighted average."""
        # Weight by quality score (better runs influence more)
        total_weight = sum(o.quality_score for o in similar)
        if total_weight == 0:
            total_weight = len(similar)
            weights = [1.0 / len(similar)] * len(similar)
        else:
            weights = [o.quality_score / total_weight for o in similar]

        # Weighted average of hyperparameters
        lr = sum(o.learning_rate * w for o, w in zip(similar, weights))
        bs = sum(o.batch_size * w for o, w in zip(similar, weights))
        epochs = sum(o.epochs * w for o, w in zip(similar, weights))
        warmup = sum(o.warmup_steps * w for o, w in zip(similar, weights))
        lora_rank = sum(o.lora_rank * w for o, w in zip(similar, weights))
        lora_alpha = sum(o.lora_alpha * w for o, w in zip(similar, weights))

        # Round to sensible values
        bs = max(2, int(round(bs / 2) * 2))  # Round to nearest even
        epochs = max(1, int(round(epochs)))
        warmup = max(0, int(round(warmup)))
        lora_rank = max(2, int(round(lora_rank / 2) * 2))
        lora_alpha = max(1, int(round(lora_alpha)))

        # Confidence based on number of runs and consistency
        n = len(similar)
        qualities = [o.quality_score for o in similar]
        variance = sum((q - sum(qualities) / n) ** 2 for q in qualities) / n
        consistency = max(0.0, 1.0 - variance * 4)  # Low variance = high consistency
        confidence = min(1.0, (0.3 + 0.1 * min(n, 7)) * (0.5 + 0.5 * consistency))

        # Best run quality
        best_q = max(o.quality_score for o in similar)
        avg_q = sum(o.quality_score for o in similar) / len(similar)
        avg_l = sum(o.final_loss for o in similar if o.final_loss > 0) / max(
            1, sum(1 for o in similar if o.final_loss > 0)
        )

        reason_parts = []
        if n >= 5:
            reason_parts.append(f"Based on {n} similar runs")
        else:
            reason_parts.append(f"Based on {n} similar runs (limited data)")
        reason_parts.append(f"avg quality {avg_q:.2f}, best {best_q:.2f}")

        return AdaptiveRecommendation(
            learning_rate=lr,
            batch_size=bs,
            epochs=epochs,
            warmup_steps=warmup,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            confidence=confidence,
            based_on_runs=n,
            reason="; ".join(reason_parts),
            avg_quality=avg_q,
            avg_loss=avg_l,
            best_quality=best_q,
        )

    def _fallback_recommendation(
        self, dataset_size: int, model: str, method: str
    ) -> AdaptiveRecommendation:
        """No history — use heuristics (same as auto_config)."""
        # Mirrors auto_config logic as baseline
        if dataset_size < 1000:
            lr, bs, epochs = 1e-4, 4, 10
            reason = "Small dataset heuristic (no history)"
        elif dataset_size < 10_000:
            lr, bs, epochs = 2e-4, 8, 5
            reason = "Medium dataset heuristic (no history)"
        else:
            lr, bs, epochs = 3e-4, 16, 3
            reason = "Large dataset heuristic (no history)"

        return AdaptiveRecommendation(
            learning_rate=lr,
            batch_size=bs,
            epochs=epochs,
            warmup_steps=max(10, dataset_size // 200),
            lora_rank=8,
            lora_alpha=16,
            confidence=0.3,
            based_on_runs=0,
            reason=reason,
        )

    @staticmethod
    def _size_category(n: int) -> str:
        if n < 1000:
            return "tiny"
        if n < 10_000:
            return "small"
        if n < 100_000:
            return "medium"
        return "large"


__all__ = ["AdaptiveRecommendation", "AdaptiveConfigEngine"]
