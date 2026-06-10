"""
Learning Rate Schedulers — pure math, no torch dependency.

All implementations use SloLRScheduler from slonet as base.
Factory delegates to slonet.create_scheduler for torch-native types.
"""

from dataclasses import dataclass
from typing import Optional
import math
import logging

logger = logging.getLogger("man.lr_schedulers")

from domains.training.slonet import (
    SloLRScheduler,
    WarmupCosineScheduler as SloWarmupCosine,
    PolynomialDecayScheduler as SloPolynomial,
    LinearWarmupScheduler as SloLinearWarmup,
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
                     warmup_steps=0, min_lr=1e-6, max_lr=1e-3, **kwargs):
    """
    Create an LR scheduler.

    Delegates to slonet.create_scheduler for all types.
    Supports: cosine, warmup, onecycle, cyclic, polynomial, step, plateau,
              cosine_annealing, none
    """
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
