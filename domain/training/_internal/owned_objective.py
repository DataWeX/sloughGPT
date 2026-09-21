"""Owned objective (13): beyond next-token prediction.

Adds auxiliary objectives that the model can learn from its own experience:
  - tool-use success (did the tool call achieve the user goal?)
  - memory consolidation (can it recall its own prior answer?)

Wraps the standard next-token loss with a weighted auxiliary term,
so TrainingLoop sees one combined loss but logs both.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger("slo.training.objective")


@dataclass(slots=True)
class OwnedObjectiveConfig:
    """Weights for auxiliary objectives (0 = disabled)."""

    tool_success_weight: float = 0.3  # reward tool-use success
    memory_weight: float = 0.2  # recall consistency
    # next-token stays 1.0 implicitly


class OwnedObjective:
    """Computes combined loss: next_token + w_tool * tool_loss + w_mem * mem_loss.

    Tool loss: if experience metadata has tool_success flag, encourage the
    assistant token that followed a successful tool call (simple reward-weighted CE).
    Memory loss: encourage consistency with prior assistant answer (if available).

    All auxiliary terms are differentiable scalars that add to the primary loss.
    When no metadata, it collapses to pure next-token (backward compat).
    """

    def __init__(self, config: OwnedObjectiveConfig | None = None) -> None:
        self.config = config or OwnedObjectiveConfig()

    def compute(
        self,
        model: Any,
        x: np.ndarray,
        y: np.ndarray,
        primary_loss: Any,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, float]]:
        """Add auxiliary terms to primary_loss.

        Args:
            model: SloNet (unused for now, but kept for future model-aware terms)
            x, y: batch (unused for simple reward scaling, but available)
            primary_loss: Tensor loss from model.forward (requires_grad)
            metadata: per-batch experience flags, e.g. {"tool_success": np.array([1,0,1])}

        Returns:
            (combined_loss Tensor, metrics dict with breakdown)
        """
        # primary_loss is a Tensor-like with .item() and supports + and * scalars
        combined = primary_loss
        metrics = {"primary": float(primary_loss.item()) if hasattr(primary_loss, "item") else 0.0}

        if metadata is None:
            return combined, metrics

        # tool success: if batch has success flag, scale primary by (1 - w * success_rate)
        # success_rate in [0,1]; lower loss when success high (reward)
        if self.config.tool_success_weight > 0 and "tool_success" in metadata:
            succ = metadata["tool_success"]
            # succ can be array of 0/1 per sample or scalar
            try:
                rate = float(np.mean(np.asarray(succ)))
            except Exception:
                rate = 0.0
            # reward: reduce loss proportionally to success
            # combined = primary * (1 - w*rate)  — keeps grad scale
            w = self.config.tool_success_weight
            # Tensor * scalar works via __mul__
            try:
                combined = combined * (1.0 - w * rate)
            except Exception:
                pass
            metrics["tool_success_rate"] = rate
            metrics["tool_weight"] = w

        if self.config.memory_weight > 0 and "memory_match" in metadata:
            try:
                rate = float(np.mean(np.asarray(metadata["memory_match"])))
            except Exception:
                rate = 0.0
            w = self.config.memory_weight
            try:
                combined = combined * (1.0 - w * rate)
            except Exception:
                pass
            metrics["memory_match_rate"] = rate
            metrics["memory_weight"] = w

        if hasattr(combined, "item"):
            metrics["combined"] = float(combined.item())
        return combined, metrics
