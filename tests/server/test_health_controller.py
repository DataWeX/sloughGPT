"""Tests for HealthController."""
import pytest
import time
from unittest.mock import patch, MagicMock
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'apps', 'api', 'server'))

import controllers.health as controllers_health
from controllers.health import (
    HealthController, _health_start_time, _get_model_info,
    _get_model_device, _is_app_ready,
    _get_lifecycle_info, _get_inference_stats, _get_quantization_info,
    _get_kv_session_info, _get_resource_allocation, _get_process_info,
    _get_executor_stats, _get_process_guard_status, _get_mps_monitor_info,
    _build_status_message, _is_model_loading,
)


@pytest.fixture
def ctrl():
    return HealthController()


class TestGetModelDevice:
    def test_returns_controller_device(self):
        ctrl = MagicMock()
        ctrl.get_current_model.return_value = {"model_id": "gpt2", "device": "cpu"}
        with patch("controllers.models.get_models_controller", return_value=ctrl):
            assert _get_model_device() == "cpu"

    def test_none_when_controller_has_no_model(self):
        ctrl = MagicMock()
        ctrl.get_current_model.return_value = None
        with patch("controllers.models.get_models_controller", return_value=ctrl):
            assert _get_model_device() is None


class TestHealthController:
    def test_init(self, ctrl):
        assert ctrl is not None

    def test_get_basic_health_returns_dict(self, ctrl):
        result = ctrl.get_basic_health()
        assert isinstance(result, dict)

    def test_basic_health_has_status(self, ctrl):
        result = ctrl.get_basic_health()
        assert "status" in result

    def test_health_has_timestamp(self, ctrl):
        result = ctrl.get_basic_health()
        assert "timestamp" in result

    def test_health_model_loaded_field(self, ctrl):
        result = ctrl.get_basic_health()
        assert "model_loaded" in result
        assert isinstance(result["model_loaded"], bool)

    def test_health_has_device_field(self, ctrl):
        result = ctrl.get_basic_health()
        assert "device" in result

    def test_health_device_none_when_no_model(self, ctrl):
        with patch.object(controllers_health, "_get_model_info", return_value=(False, None)):
            with patch.object(controllers_health, "_get_model_device", return_value=None):
                result = ctrl.get_basic_health()
        assert result["device"] is None

    def test_health_reports_device_when_loaded(self, ctrl):
        with patch.object(controllers_health, "_get_model_info", return_value=(True, "gpt2")):
            with patch.object(controllers_health, "_get_model_device", return_value="cpu"):
                result = ctrl.get_basic_health()
        assert result["device"] == "cpu"

    def test_detailed_health_has_device_field(self, ctrl):
        with patch.object(controllers_health, "_get_model_device", return_value="cpu"):
            result = ctrl.get_detailed_health()
        assert "device" in result
        assert result["device"] == "cpu"

    def test_detailed_health_device_none_when_no_model(self, ctrl):
        with patch.object(controllers_health, "_get_model_device", return_value=None):
            result = ctrl.get_detailed_health()
        assert result["device"] is None

    def test_get_liveness(self, ctrl):
        result = ctrl.get_liveness()
        assert isinstance(result, dict)
        assert result["status"] == "alive"

    def test_get_readiness(self, ctrl):
        result = ctrl.get_readiness()
        assert isinstance(result, dict)


class TestHelpers:
    def test_health_start_time_is_set(self):
        assert _health_start_time is not None


class TestBuildStatusMessage:
    def test_ready_with_model(self):
        msg = _build_status_message(True, "gpt2", False, None, 10, 0, {"phase": "running", "profile": "default", "is_running": True, "is_draining": False})
        assert "Ready" in msg
        assert "gpt2" in msg

    def test_ready_with_soul(self):
        msg = _build_status_message(True, "gpt2", False, "warm", 10, 0, {"phase": "running", "profile": "default", "is_running": True, "is_draining": False})
        assert "warm" in msg

    def test_ready_with_errors(self):
        msg = _build_status_message(True, "gpt2", False, None, 10, 3, {"phase": "running", "profile": "default", "is_running": True, "is_draining": False})
        assert "3 errors" in msg

    def test_loading_model(self):
        msg = _build_status_message(False, None, True, None, 0, 0, {"phase": "running", "profile": "default", "is_running": True, "is_draining": False})
        assert "Loading model" in msg

    def test_no_model(self):
        msg = _build_status_message(False, None, False, None, 0, 0, {"phase": "running", "profile": "default", "is_running": True, "is_draining": False})
        assert "no model loaded" in msg

    def test_draining(self):
        msg = _build_status_message(True, "gpt2", False, None, 10, 0, {"phase": "draining", "profile": "default", "is_running": True, "is_draining": True, "in_flight": 5})
        assert "Draining" in msg
        assert "5" in msg

    def test_starting(self):
        msg = _build_status_message(False, None, False, None, 0, 0, {"phase": "starting", "profile": "default", "is_running": False, "is_draining": False})
        assert "Starting" in msg


class TestGetModelInfo:
    def test_returns_tuple(self):
        result = _get_model_info()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_loaded_is_bool(self):
        loaded, model_type = _get_model_info()
        assert isinstance(loaded, bool)

    def test_model_type_or_none(self):
        loaded, model_type = _get_model_info()
        assert model_type is None or isinstance(model_type, str)


class TestGetLifecycleInfo:
    def test_returns_dict(self):
        result = _get_lifecycle_info()
        assert isinstance(result, dict)
        assert "phase" in result

    def test_has_profile(self):
        result = _get_lifecycle_info()
        assert "profile" in result


class TestGetInferenceStats:
    def test_returns_dict(self):
        result = _get_inference_stats()
        assert isinstance(result, dict)


class TestGetQuantizationInfo:
    def test_returns_dict(self):
        result = _get_quantization_info()
        assert isinstance(result, dict)


class TestGetKvSessionInfo:
    def test_returns_dict(self):
        result = _get_kv_session_info()
        assert isinstance(result, dict)
        assert "enabled" in result


class TestGetResourceAllocation:
    def test_returns_dict(self):
        result = _get_resource_allocation()
        assert isinstance(result, dict)


class TestGetProcessInfo:
    def test_returns_dict(self):
        result = _get_process_info()
        assert isinstance(result, dict)


class TestGetExecutorStats:
    def test_returns_dict_or_none(self):
        result = _get_executor_stats()
        assert result is None or isinstance(result, dict)


class TestGetProcessGuardStatus:
    def test_returns_dict_or_none(self):
        result = _get_process_guard_status()
        assert result is None or isinstance(result, dict)


class TestGetMpsMonitorInfo:
    def test_returns_dict_or_none(self):
        result = _get_mps_monitor_info()
        assert result is None or isinstance(result, dict)


class TestDetailedHealth:
    def test_returns_dict(self, ctrl):
        result = ctrl.get_detailed_health()
        assert isinstance(result, dict)

    def test_has_required_fields(self, ctrl):
        result = ctrl.get_detailed_health()
        assert "status" in result
        assert "system" in result
        assert "gpu" in result

    def test_system_has_cpu_percent(self, ctrl):
        result = ctrl.get_detailed_health()
        assert "cpu_percent" in result["system"]

    def test_caching(self, ctrl):
        r1 = ctrl.get_detailed_health()
        r2 = ctrl.get_detailed_health()
        assert r1 is r2

    def test_cache_expired(self, ctrl):
        r1 = ctrl.get_detailed_health()
        ctrl._cache_time = 0  # Force cache miss
        r2 = ctrl.get_detailed_health()
        assert r2 is not None


class TestIsAppReady:
    def test_ready_when_lifecycle_running(self):
        mgr = MagicMock()
        mgr.is_running.return_value = True
        with patch("domains.infrastructure.lifecycle.get_lifecycle_manager", return_value=mgr):
            assert _is_app_ready() is True

    def test_not_ready_when_lifecycle_not_running(self):
        mgr = MagicMock()
        mgr.is_running.return_value = False
        with patch("domains.infrastructure.lifecycle.get_lifecycle_manager", return_value=mgr):
            assert _is_app_ready() is False

    def test_ready_on_import_error(self):
        with patch("domains.infrastructure.lifecycle.get_lifecycle_manager", side_effect=ImportError):
            assert _is_app_ready() is True


class TestGetModelInfoReadyGate:
    def test_model_info_false_before_ready(self):
        mgr = MagicMock()
        mgr.is_running.return_value = False
        with patch("domains.infrastructure.lifecycle.get_lifecycle_manager", return_value=mgr), \
             patch("controllers.health._get_model_info_with_registry",
                   return_value=(True, "gpt2", {})):
            loaded, model_type = _get_model_info()
            assert loaded is False
            assert model_type == "gpt2"

    def test_model_info_true_after_ready(self):
        mgr = MagicMock()
        mgr.is_running.return_value = True
        with patch("domains.infrastructure.lifecycle.get_lifecycle_manager", return_value=mgr), \
             patch("controllers.health._get_model_info_with_registry",
                   return_value=(True, "gpt2", {})):
            loaded, model_type = _get_model_info()
            assert loaded is True
            assert model_type == "gpt2"


class TestIsModelLoading:
    def test_loading_when_no_model_and_recent(self):
        from datetime import datetime, timedelta
        import state as server_state
        old_model = getattr(server_state, "model", None)
        old_provider = getattr(server_state, "provider", None)
        old_start = controllers_health._health_start_time
        try:
            server_state.model = None
            server_state.provider = None
            controllers_health._health_start_time = datetime.now() - timedelta(seconds=30)
            result = _is_model_loading()
            assert result is True
        finally:
            server_state.model = old_model
            server_state.provider = old_provider
            controllers_health._health_start_time = old_start

    def test_not_loading_when_model_loaded(self):
        import state as server_state
        old_model = getattr(server_state, "model", None)
        try:
            server_state.model = MagicMock()
            result = _is_model_loading()
            assert result is False
        finally:
            server_state.model = old_model
