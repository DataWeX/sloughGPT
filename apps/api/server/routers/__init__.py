"""
API Routers Package
Modular API endpoints organized by feature.
"""
from typing import List
from fastapi import APIRouter
from . import auth
from . import auto_train
# training endpoints are in apps/api/server/training/router.py — registered directly in main.py
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
from . import personalities
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
from . import labs
from . import learner
from . import self_train

__all__ = [
    "auth", "auto_train", "models", "inference",
    "feedback", "knowledge", "agents", "system", "status", "souls",
    "config", "health", "metrics", "datasets", "security",
    "ratelimit", "workflow", "experiments", "personalities", "benchmark",
    "user_adapters", "vector", "registry",
    "session", "meta_weights", "lora_eval", "companion", "multimodal",
    "tokenizer", "labs", "learner", "self_train",
]

def get_all_routers() -> list[APIRouter]:
    """Get all routers for main.py to include."""
    return [
        auth.router, auto_train.router, models.router, inference.router,
        feedback.router, knowledge.router, agents.router, system.router,
        status.router, souls.router, config.router, health.router,
        metrics.router, datasets.router, security.router,
        ratelimit.router, workflow.router, experiments.router,
        personalities.router, benchmark.router,
        user_adapters.router, vector.router, registry.router,
        session.router, meta_weights.router, lora_eval.router,
        companion.router, multimodal.router, tokenizer.router, labs.router,
        learner.router, self_train.router,
    ]