"""Shared test fixtures and helpers for router tests."""

import asyncio
import sys
from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "live: marks tests that require running servers")


# parents[3] = repo root for a file at packages/core-py/tests depth (same
# idiom as the _avion_src line below); parents[2] yielded <repo>/packages/apps/...
_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

# Journey tests drive the shared avion computer-use library. rootdir for this
# package is packages/core-py (it has its own pytest.ini), so the repo-root
# conftest never loads here — add avion from source instead of installing it
# into the shared .venv (shared-workspace rule: no new installs).
_avion_src = str(Path(__file__).resolve().parents[3] / "packages" / "avion" / "src")
if _avion_src not in sys.path:
    sys.path.insert(0, _avion_src)


@pytest.fixture(autouse=True)
def _ensure_event_loop():
    """Ensure a working event loop exists for each test."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    yield


@pytest.fixture(autouse=True)
def _stop_pretrain_threads():
    """Stop any background multimodal pretrain thread after each test.

    Leaked pretrain threads run active BLAS compute and deadlock later
    os.fork() calls (SubprocessProcess start) — the core-py full-suite hang.
    """
    yield
    manager_mod = sys.modules.get("domain.multimodal._internal.manager")
    if manager_mod is not None:
        manager_mod.MultimodalManager.stop_pretrain(timeout=5)


@pytest.fixture(autouse=True)
def _stop_infra_threads():
    """L3: stop leaked infra background threads after each test.

    The leak sites already have stop APIs — workflow schedulers, the
    fire-and-forget pool, the idle-manager loop (gate run5: 44x timeouts,
    ~54 leaked threads). Only subsystems this test imported are touched.
    """
    yield
    workflow_mod = sys.modules.get("domain.feedback._internal.workflow")
    if workflow_mod is not None:
        stop_all = getattr(workflow_mod, "stop_all_workflows", None)
        if stop_all is not None:
            stop_all()
    faf_mod = sys.modules.get("domain.infrastructure._internal.fire_and_forget")
    if faf_mod is not None:
        reset = getattr(faf_mod, "reset_pool", None)
        if reset is not None:
            reset()
    idle_mod = sys.modules.get("domain.infrastructure._internal.idle_manager")
    if idle_mod is not None:
        stop_all = getattr(idle_mod, "stop_all_idle_managers", None)
        if stop_all is not None:
            stop_all()
    pugqeep_mod = sys.modules.get("domain.infrastructure._internal.pugqeep.engine")
    if pugqeep_mod is not None:
        stop_all = getattr(pugqeep_mod, "stop_all_pugqeep", None)
        if stop_all is not None:
            stop_all()


def build_test_app(*routers):
    """Build a FastAPI app with exception handlers registered.

    Usage::

        app = build_test_app(my_router)
        client = TestClient(app)

    This ensures raise_error() exceptions are properly caught and
    converted to JSON responses, matching production behavior.
    """
    from fastapi import FastAPI

    app = FastAPI()
    for r in routers:
        app.include_router(r)

    from infrastructure.exception_handlers import register_all_handlers

    register_all_handlers(app)

    return app
