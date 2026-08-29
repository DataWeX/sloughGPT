"""
API Routers Package
Modular API endpoints organized by feature.

All router imports are deferred to ``get_all_routers()`` to avoid pulling in
heavy dependencies (JAX, sentence-transformers, PyTorch) at module-load time.
This cuts API cold-start from ~100s to ~8s.
"""
import logging
from typing import List, Optional
from fastapi import APIRouter

logger = logging.getLogger("slo.routers")

__all__ = [
    "get_all_routers",
]

_cached_routers: Optional[List[APIRouter]] = None


def _try_import_router(module_name: str, attribute: str = "router") -> Optional[APIRouter]:
    """Import a single router module, returning None on failure."""
    try:
        import importlib
        mod = importlib.import_module(f".{module_name}", package=__name__)
        return getattr(mod, attribute)
    except Exception:
        logger.warning("Router '%s' failed to import — skipping", module_name, exc_info=True)
        return None


def get_all_routers() -> List[APIRouter]:
    """Get all routers for main.py to include.

    Imports are deferred to first call to avoid 90s+ cold-start from
    transitive heavy imports (JAX via datasets, sentence-transformers, etc.).
    Individual router failures are caught so one broken module cannot prevent
    all other routes from registering.
    """
    global _cached_routers
    if _cached_routers is not None:
        return _cached_routers

    # Note: "health" and "status" are registered directly in main.py
    # pre-lifespan (needed during model load). Do NOT list them here.
    _router_names = [
        "auth", "auto_train", "models", "inference",
        "feedback", "kb", "agents", "system",
        "souls", "config",
        "security", "metrics", "datasets",
        "ratelimit", "workflow", "experiments",
        "benchmark",
        "user_adapters", "vector", "registry",
        "session", "meta_weights", "lora_eval",
        "companion", "multimodal", "tokenizer",
        "learner", "self_train",
        "token_tree",
        "errors", "mobile",
        "images", "files", "voice", "infer",
        "vm", "memory", "docstore",
        "collections",
        "shell", "world_render",
        "tokens",
    ]

    _cached_routers = []
    for name in _router_names:
        r = _try_import_router(name)
        if r is not None:
            _cached_routers.append(r)

    return _cached_routers
