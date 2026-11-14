"""Consolidation pins — card 20261101_154058 (v1 /training router row).

``routers/self_train.py``, ``routers/lora_eval.py`` and
``routers/cloud_training.py`` are folded into the training aggregate:
descriptor-projected inside ``apps/api/server/training/``, mounted by
``training/router.py``. The standalone ``routers/`` modules, their boot
manifest entries and their hand-built route sites are all gone — the same
facade-absorption shape as ``TestShellFacadeAbsorbsVm`` (test_vm_router).

Journeys: UX_FLOWS §12 Train My AI (self-train) and §14 Add an Ability
(cloud-training) — paths stay byte-identical for their controllers.
"""

import importlib
import os
import sys

import pytest

_server_dir = os.path.join(os.path.dirname(__file__), "..", "..")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

# The 10 wire paths the frontend controllers pin (self-train-controller,
# lora-eval-controller, plugins-cloud-controller) — byte-identical or bust.
TEN_PATHS = [
    "/self-train/start",
    "/self-train/stop",
    "/self-train/status",
    "/lora-eval/run",
    "/lora-eval/history",
    "/lora-eval/aggregate",
    "/cloud-training/jobs",
    "/cloud-training/submit",
    "/cloud-training/{job_id}/status",
    "/cloud-training/{job_id}/cancel",
]
OLD_MODULES = ["routers.self_train", "routers.lora_eval", "routers.cloud_training"]
OLD_MANIFEST = ["self_train", "lora_eval", "cloud_training"]


class TestTrainingAggregateAbsorbsThreeRouters:
    """The three routers live under training/ — and only under training/."""

    def test_ten_paths_served_by_training_aggregate(self):
        from fastapi import FastAPI
        from training.router import router as training_router

        facade = FastAPI()
        facade.include_router(training_router)
        paths = set(facade.openapi()["paths"])
        missing = [p for p in TEN_PATHS if p not in paths]
        assert not missing, f"training aggregate does not serve: {missing}"

    @pytest.mark.parametrize("module", OLD_MODULES)
    def test_old_module_gone(self, module):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)

    def test_manifest_dropped_three_entries(self):
        from routers._manifest import ROUTER_MODULES

        for name in OLD_MANIFEST:
            assert name not in ROUTER_MODULES, f"{name} still in boot manifest"
