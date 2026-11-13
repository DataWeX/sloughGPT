"""
StepStrategy — the single per-step strategy seam in SloughGPTTrainer.train_step().

One training loop (``SloughGPTTrainer``) owns forward/loss/backward/clip/step;
strategies only reshape ``(loss, metrics)`` at the seam between the composed
loss and ``backward()``. EWC, the owned objective (experience/feedback) and
distillation are adapters over this protocol — ``MetaTrainer`` composes them,
``ComprehensiveTrainer`` keeps phase scheduling (*when* strategies exist at all).
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

import numpy as np

from domain.training._internal.slonet import kl_div_loss, log_softmax, no_grad, softmax

logger = logging.getLogger("slo.step_strategy")


@runtime_checkable
class StepStrategy(Protocol):
    """Per-step hook run by ``SloughGPTTrainer.train_step`` after the NaN guard.

    Contract:

    - ``loss`` is the loss composed so far (primary + earlier strategies); the
      returned loss must survive ``* scale_factor`` and ``.backward()`` when a
      gradient signal is intended (add a SloNet ``Tensor`` that carries a
      graph — a plain float shifts the scalar without teaching anything).
    - ``logits`` are this batch's fresh student logits (may be ignored).
    - ``metadata`` is the trainer's experience metadata (may be ``None``).
    - returned metric keys are **final** (already namespaced) and are merged
      into the step metrics as-is.
    - a strategy that does not apply this step returns ``(loss, {})``.
    """

    name: str

    def compute(
        self,
        model: Any,
        x: np.ndarray,
        y: np.ndarray,
        loss: Any,
        *,
        logits: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, float]]: ...


class EwcStrategy:
    """EWC penalty adapter over ``EwcContinualLearner``.

    Parity with the legacy inline block in ``train_step``: applies only when
    ``task_snapshots`` is non-empty; the legacy path dropped the stats dict
    and this adapter keeps dropping it (no metric-key changes).
    """

    name = "ewc"

    def __init__(self, learner: Any) -> None:
        self.learner = learner

    def compute(
        self,
        model: Any,
        x: np.ndarray,
        y: np.ndarray,
        loss: Any,
        *,
        logits: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, float]]:
        learner = self.learner
        if learner is None or not getattr(learner, "task_snapshots", None):
            return loss, {}
        ewc_loss, _stats = learner.ewc_loss()
        return loss + ewc_loss, {}


class OwnedObjectiveStrategy:
    """Adapter over ``OwnedObjective`` (goal 13): experience/feedback terms.

    Metric keys keep their historical ``owned_`` namespace so step metrics are
    byte-identical to the pre-strategy inline path.
    """

    name = "owned_objective"

    def __init__(self, objective: Any) -> None:
        self.objective = objective

    def compute(
        self,
        model: Any,
        x: np.ndarray,
        y: np.ndarray,
        loss: Any,
        *,
        logits: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, float]]:
        if self.objective is None:
            return loss, {}
        combined, metrics = self.objective.compute(model, x, y, loss, metadata=metadata)
        return combined, {f"owned_{k}": float(v) for k, v in metrics.items()}


class DistillationStrategy:
    """Soft-label distillation against a teacher model.

    ``DistillationLoss.forward`` converts to numpy/float at its API boundary by
    design, so its total cannot carry gradients; this adapter computes the same
    temperature-scaled soft KL directly on SloNet tensors, keeping the graph
    from student logits through to ``backward()``. The teacher runs under
    ``no_grad`` (no graph, no teacher-side accumulation).
    """

    name = "distillation"

    def __init__(self, teacher: Any, *, weight: float = 1.0, temperature: float = 2.0) -> None:
        if teacher is None:
            raise ValueError("DistillationStrategy requires a teacher")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.teacher = teacher
        self.weight = float(weight)
        self.temperature = float(temperature)

    def compute(
        self,
        model: Any,
        x: np.ndarray,
        y: np.ndarray,
        loss: Any,
        *,
        logits: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, float]]:
        if self.weight <= 0 or logits is None:
            return loss, {}
        temp = self.temperature
        with no_grad():
            teacher_out = self.teacher(x, y)
            t_logits = teacher_out[0] if isinstance(teacher_out, tuple) else teacher_out
            t_soft = softmax(t_logits / temp, dim=-1)
        s_soft = log_softmax(logits / temp, dim=-1)
        soft = kl_div_loss(s_soft, t_soft, reduction="batchmean") * (temp**2)
        metrics = {"distill_soft": float(soft.item())}
        logger.debug("distill_soft=%.6f temp=%.2f", metrics["distill_soft"], temp)
        return loss + soft * self.weight, metrics
