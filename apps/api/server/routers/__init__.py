"""
API Routers Package
Modular API endpoints organized by feature.

All router imports are deferred to ``get_all_routers()`` to avoid pulling in
heavy dependencies (JAX, sentence-transformers, PyTorch) at module-load time.
This cuts API cold-start from ~100s to ~8s.
"""
from typing import List, Optional
from fastapi import APIRouter

__all__ = [
    "get_all_routers",
]

_cached_routers: Optional[List[APIRouter]] = None


def get_all_routers() -> List[APIRouter]:
    """Get all routers for main.py to include.

    Imports are deferred to first call to avoid 90s+ cold-start from
    transitive heavy imports (JAX via datasets, sentence-transformers, etc.).
    """
    global _cached_routers
    if _cached_routers is not None:
        return _cached_routers

    from . import auth
    from . import auto_train
    from . import models
    from . import inference
    from . import feedback
    from . import kb as knowledge
    from . import agents
    from . import system
    from . import status
    from . import souls
    from . import config
    from . import health
    from . import security
    from . import metrics
    from . import datasets
    from . import ratelimit
    from . import workflow
    from . import experiments
    from . import benchmark
    from . import user_adapters
    from . import vector
    from . import registry
    from . import session
    from . import meta_weights
    from . import lora_eval
    from . import companion
    from . import multimodal
    from . import tokenizer
    from . import learner
    from . import self_train
    from . import errors as error_logger
    from . import mobile
    from . import images
    from . import files
    from . import voice
    from . import activity

    _cached_routers = [
        auth.router, auto_train.router, models.router, inference.router,
        feedback.router, knowledge.router, agents.router, system.router,
        status.router, souls.router, config.router, health.router,
        metrics.router, datasets.router, security.router,
        ratelimit.router, workflow.router, experiments.router,
        benchmark.router,
        user_adapters.router, vector.router, registry.router,
        session.router, meta_weights.router, lora_eval.router,
        companion.router, multimodal.router, tokenizer.router,
        learner.router, self_train.router,
        error_logger.router,
        mobile.router,
        images.router,
        files.router,
        voice.router,
        activity.router,
    ]
    return _cached_routers
