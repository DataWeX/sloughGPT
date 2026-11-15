"""
MetaTrainer — strategy host over the one training loop.

Composes *what* is applied per step by handing a ``StepStrategy`` list to
``SloughGPTTrainer``; it never re-implements forward/loss/backward/optimize
and never schedules phases — ``ComprehensiveTrainer`` owns *when* training
modes run. Feedback (experience metadata) needs no extra strategy: configure
the trainer with ``use_owned_objective=True`` on an experience data_path and
the owned objective appears automatically as ``OwnedObjectiveStrategy`` via
``SloughGPTTrainer._iter_strategies``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from domain.training._internal.step_strategy import (
    DistillationStrategy,
    EwcStrategy,
    StepStrategy,
)
from domain.training._internal.train_pipeline import SloughGPTTrainer, TrainerConfig

logger = logging.getLogger("slo.meta_trainer")


class MetaTrainer:
    """Build and hold a ``SloughGPTTrainer`` with composed step strategies."""

    def __init__(
        self,
        data_path: str,
        config: TrainerConfig | None = None,
        strategies: Sequence[StepStrategy] = (),
        **trainer_kwargs: Any,
    ) -> None:
        self.data_path = data_path
        self.config = config
        self.strategies: list[StepStrategy] = list(strategies)
        self._trainer_kwargs = trainer_kwargs
        self.trainer: SloughGPTTrainer | None = None

    def build_trainer(self) -> SloughGPTTrainer:
        """Create the single-loop trainer with this host's strategies injected."""
        self.trainer = SloughGPTTrainer(
            self.data_path,
            config=self.config,
            strategies=self.strategies,
            **self._trainer_kwargs,
        )
        logger.info(
            "MetaTrainer: built trainer with %d strategies",
            len(self.strategies),
            extra={"tag": "TRAIN"},
        )
        return self.trainer

    def train(self, **kwargs: Any) -> Any:
        """Delegate to ``SloughGPTTrainer.train`` — MetaTrainer adds no loop."""
        trainer = self.trainer if self.trainer is not None else self.build_trainer()
        return trainer.train(**kwargs)

    @classmethod
    def with_ewc(
        cls,
        data_path: str,
        ewc_learner: Any,
        *,
        config: TrainerConfig | None = None,
        **trainer_kwargs: Any,
    ) -> MetaTrainer:
        """Compose the continual-learning (EWC) penalty as a strategy."""
        return cls(
            data_path,
            config=config,
            strategies=[EwcStrategy(ewc_learner)],
            **trainer_kwargs,
        )

    @classmethod
    def with_distillation(
        cls,
        data_path: str,
        teacher: Any,
        *,
        weight: float = 1.0,
        temperature: float = 2.0,
        config: TrainerConfig | None = None,
        **trainer_kwargs: Any,
    ) -> MetaTrainer:
        """Compose soft-label distillation against ``teacher`` as a strategy."""
        return cls(
            data_path,
            config=config,
            strategies=[DistillationStrategy(teacher, weight=weight, temperature=temperature)],
            **trainer_kwargs,
        )
