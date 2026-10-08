"""Regression tests for startup preloading of the model-load + router graphs.

Guards the fix where the router-registration graph is now warmed by
``_preload_model_imports()`` on the main thread. Previously the routers hook
(level 2) performed cold first-time imports while the model-load thread was
reading weights from disk, which could stretch the hook past the CLI's
startup deadline on cold boots.
"""

import os
import sys

os.environ.setdefault("SLO_AUTO_WORKFLOW", "false")
os.environ.setdefault("SLO_AUTOLOAD_MODEL", "")

import routers
from infrastructure.startup import _PREWARM_MODEL_LOAD_IMPORTS, _preload_model_imports


class TestPreloadModelImports:
    def test_warms_model_load_graph(self):
        _preload_model_imports()
        for mod in _PREWARM_MODEL_LOAD_IMPORTS:
            assert mod in sys.modules, f"{mod} not preloaded"

    def test_warms_router_registration_graph(self):
        _preload_model_imports()
        assert routers._cached_routers is not None
        assert len(routers._cached_routers) > 0

    def test_preload_does_not_raise_on_cold_graph(self):
        _preload_model_imports()
