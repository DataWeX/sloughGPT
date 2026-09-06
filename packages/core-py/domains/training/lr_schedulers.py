"""
Learning Rate Schedulers — pure math.

All implementations use SloLRScheduler from slonet as base.
Factory delegates to slonet.create_scheduler for native types.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("slo.lr_schedulers")

from domains.training.slonet import (
    LinearWarmupScheduler as SloLinearWarmup,
)
from domains.training.slonet import (
    PolynomialDecayScheduler as SloPolynomial,
)
from domains.training.slonet import (
    WarmupCosineScheduler as SloWarmupCosine,
)
from domains.training.slonet import (
    create_scheduler as soul_create_scheduler,
)

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SchedulerConfig:
    name: str = "none"
    initial_lr: float = 1e-4


@dataclass
class CosineAnnealingConfig(SchedulerConfig):
    name: str = "cosine"
    min_lr: float = 1e-6
    warmup_steps: int = 0
    total_steps: int = 10000
    num_cycles: float = 0.5


@dataclass
class WarmupConfig(SchedulerConfig):
    name: str = "warmup"
    warmup_steps: int = 500
    warmup_start_lr: float = 1e-7


@dataclass
class OneCycleConfig(SchedulerConfig):
    name: str = "onecycle"
    max_lr: float = 1e-3
    pct_start: float = 0.1
    anneal_strategy: str = "cos"


@dataclass
class CyclicConfig(SchedulerConfig):
    name: str = "cyclic"
    base_lr: float = 1e-5
    max_lr: float = 1e-3
    step_size_up: int = 1000
    step_size_down: Optional[int] = None
    mode: str = "triangular2"


# =============================================================================
# Custom Schedulers (wrap slonet impls for backward compat)
# =============================================================================


class WarmupCosineScheduler(SloWarmupCosine):
    """Cosine annealing with linear warmup."""
    pass


class PolynomialDecayScheduler(SloPolynomial):
    """Polynomial learning rate decay."""
    pass


class LinearWarmupScheduler(SloLinearWarmup):
    """Linear warmup then hold or decay."""
    pass


# =============================================================================
# Factory
# =============================================================================


def create_scheduler(optimizer, scheduler_type: str, total_steps=None,
                     warmup_steps=0, min_lr=1e-6, max_lr=3e-4, **kwargs):
    """
    Create an LR scheduler.

    Delegates to slonet.create_scheduler for all types.
    Supports: cosine, warmup, onecycle, cyclic, polynomial, step, plateau,
              cosine_annealing, none

    Args:
        optimizer: The optimizer to schedule
        scheduler_type: Type of scheduler ('cosine', 'warmup', etc.)
        total_steps: Total training steps
        warmup_steps: Number of warmup steps (default: 0)
        min_lr: Minimum learning rate (default: 1e-6)
        max_lr: Maximum/initial learning rate (default: 3e-4)
        **kwargs: Additional scheduler-specific arguments
    """
    # Auto-calculate warmup steps if not provided
    if warmup_steps == 0 and total_steps and total_steps > 1000:
        # Default to 5% of total steps for warmup, minimum 100 steps
        warmup_steps = max(100, int(total_steps * 0.05))
        logger.info(f"Auto warmup_steps={warmup_steps} (5% of total_steps={total_steps})")

    return soul_create_scheduler(optimizer, scheduler_type,
                                 total_steps=total_steps, warmup_steps=warmup_steps,
                                 min_lr=min_lr, max_lr=max_lr, **kwargs)


BEST_PRACTICES = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     LR SCHEDULER BEST PRACTICES                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  1. WARMUP IS ESSENTIAL                                                     ║
║  2. COSINE ANNEALING IS DEFAULT                                             ║
║  3. ONECYCLE FOR FAST CONVERGENCE                                           ║
║  4. LR RANGES: embeddings 1e-5-1e-4, full 1e-5-1e-4, LoRA 1e-4-1e-3       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

__all__ = [
    "SchedulerConfig",
    "CosineAnnealingConfig",
    "WarmupConfig",
    "OneCycleConfig",
    "CyclicConfig",
    "WarmupCosineScheduler",
    "PolynomialDecayScheduler",
    "LinearWarmupScheduler",
    "create_scheduler",
    "BEST_PRACTICES",
]
