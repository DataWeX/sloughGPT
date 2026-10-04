"""
API Routers Package
Modular API endpoints organized by feature.

Which routers exist, and the order they mount in, lives in the generated
``_manifest.py`` (``scripts/gen_router_manifest.py``) rather than a
hand-maintained list here — see that script's docstring for why order is
load-bearing. This package only decides *how* they are loaded.

All router imports are deferred to ``get_all_routers()`` to avoid pulling in
heavy dependencies (JAX, sentence-transformers, PyTorch) at module-load time.
This cuts API cold-start from ~100s to ~8s.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from ._manifest import ROUTER_MODULES

logger = logging.getLogger("slo.routers")

__all__ = [
    "get_all_routers",
]

_cached_routers: list[APIRouter] | None = None


def _try_import_router(module_name: str, attribute: str = "router") -> APIRouter | None:
    """Import a single router module, returning None on failure."""
    try:
        import importlib

        mod = importlib.import_module(f".{module_name}", package=__name__)
        return getattr(mod, attribute)
    except Exception:
        logger.warning("Router '%s' failed to import — skipping", module_name, exc_info=True)
        return None


def get_all_routers() -> list[APIRouter]:
    """Get all routers for main.py to include.

    Imports are deferred to first call to avoid 90s+ cold-start from
    transitive heavy imports (JAX via datasets, sentence-transformers, etc.).
    Individual router failures are caught so one broken module cannot prevent
    all other routes from registering.
    """
    global _cached_routers
    if _cached_routers is not None:
        return _cached_routers

    # "health", "status", "consciousness" and "dashboard" are registered
    # directly in main.py pre-lifespan (they must answer while the model is
    # still loading), so the manifest excludes them — and the generator
    # verifies that exclusion against main.py on every run.
    #
    # Failures stay per-router: one broken module cannot prevent the rest
    # from registering.
    _cached_routers = []
    for name in ROUTER_MODULES:
        r = _try_import_router(name)
        if r is not None:
            _cached_routers.append(r)

    return _cached_routers
