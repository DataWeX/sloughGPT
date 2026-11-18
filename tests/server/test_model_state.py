"""The lazy preload path feeds the model lifecycle holder.

Card 20261104_053224 fix(health): readiness/health used an uptime window
because nothing recorded what the boot path was actually doing. These tests
pin the writers: scheduling a preload marks ``materialising``, a successful
preload marks ``loaded``, a failed one ``degraded``, and a shutdown skip
changes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = REPO_ROOT / "apps" / "api" / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from infrastructure import model_state
from infrastructure.startup import (
    _cancel_parent_preload,
    _preload_parent_weights,
    _shutting_down,
    _start_parent_preload,
)


class _FakeProvider:
    """Minimal provider: materialize_model() publishes ``_model`` or raises."""

    def __init__(self, model=None, fail: bool = False):
        self._model = None
        self._fail = fail
        self._result_model = model

    def materialize_model(self):
        if self._fail:
            raise RuntimeError("materialise failed")
        self._model = self._result_model


@pytest.fixture(autouse=True)
def _clean_holder():
    model_state.reset_model_state()
    was_shutting_down = _shutting_down.is_set()
    _shutting_down.clear()
    yield
    if was_shutting_down:
        _shutting_down.set()
    else:
        _shutting_down.clear()
    model_state.reset_model_state()


@pytest.fixture
def server_state():
    """The shared ``state`` module with model/provider saved and restored."""
    import state as ss

    old_model = getattr(ss, "model", None)
    old_provider = getattr(ss, "provider", None)
    yield ss
    ss.model = old_model
    ss.provider = old_provider


class TestPreloadScheduling:
    def test_scheduling_preload_marks_materialising(self):
        _start_parent_preload("test-model", delay_s=3600)
        try:
            assert model_state.get_model_state() == "materialising"
        finally:
            _cancel_parent_preload("test_cleanup")

    def test_cancel_does_not_change_state(self):
        """Cancel only happens at shutdown — it must not invent a state."""
        _start_parent_preload("test-model", delay_s=3600)
        assert _cancel_parent_preload("test_cleanup") is True
        assert model_state.get_model_state() == "materialising"


class TestPreloadExecution:
    def test_successful_preload_marks_loaded(self, server_state):
        fake_model = MagicMock(name="model")
        server_state.provider = _FakeProvider(model=fake_model)
        with patch(
            "domain.infrastructure.server_state.get_server_state",
            return_value=MagicMock(),
        ):
            ok = _preload_parent_weights("test-model")
        assert ok is True
        assert model_state.get_model_state() == "loaded"
        assert server_state.model is fake_model

    def test_failed_preload_marks_degraded(self, server_state):
        server_state.provider = _FakeProvider(fail=True)
        ok = _preload_parent_weights("test-model")
        assert ok is False
        assert model_state.get_model_state() == "degraded"

    def test_missing_provider_marks_degraded(self, server_state):
        server_state.provider = None
        ok = _preload_parent_weights("test-model")
        assert ok is False
        assert model_state.get_model_state() == "degraded"

    def test_shutdown_skip_changes_nothing(self, server_state):
        server_state.provider = _FakeProvider(model=MagicMock())
        _shutting_down.set()
        ok = _preload_parent_weights("test-model")
        assert ok is False
        assert model_state.get_model_state() == "absent"
