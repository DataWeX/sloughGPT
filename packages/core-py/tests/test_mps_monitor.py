"""Tests for MPSMemoryMonitor — MPS memory tracking and CPU fallback."""

import time

import pytest

from domains.infrastructure.mps_monitor import MPSMemoryMonitor, get_mps_monitor


@pytest.fixture
def monitor():
    m = MPSMemoryMonitor()
    m._last_check = 0.0
    return m


class TestGetDevice:
    def test_cpu_request_always_cpu(self, monitor):
        assert monitor.get_device("cpu") == "cpu"

    def test_cpu_even_when_locked(self, monitor):
        monitor.force_cpu()
        assert monitor.get_device("cpu") == "cpu"

    def test_auto_resolves_mps_when_safe(self, monitor):
        monitor._get_mps_usage = lambda: 0.0
        assert monitor.get_device("auto") == "mps"

    def test_auto_resolves_cpu_when_locked_cached(self, monitor):
        monitor.force_cpu()
        monitor._last_check = time.time()
        assert monitor.get_device("auto") == "cpu"

    def test_requested_device_passthrough(self, monitor):
        monitor._get_mps_usage = lambda: 0.0
        assert monitor.get_device("cuda") == "cuda"


class TestWarnThreshold:
    def test_warn_threshold_forces_cpu(self, monitor):
        monitor._get_mps_usage = lambda: 0.30
        assert monitor.get_device("auto") == "cpu"
        assert monitor.is_locked_to_cpu() is True

    def test_below_warn_keeps_mps(self, monitor):
        monitor._get_mps_usage = lambda: 0.29
        assert monitor.get_device("auto") == "mps"
        assert monitor.is_locked_to_cpu() is False


class TestForceCpuThreshold:
    def test_force_threshold_forces_cpu(self, monitor):
        cleared = []
        monitor._get_mps_usage = lambda: 0.40
        monitor._clear_mps_cache = lambda: cleared.append(1)
        assert monitor.get_device("auto") == "cpu"
        assert monitor.is_locked_to_cpu() is True
        assert cleared == [1]

    def test_between_warn_and_force_uses_cpu_without_clear(self, monitor):
        cleared = []
        monitor._get_mps_usage = lambda: 0.35
        monitor._clear_mps_cache = lambda: cleared.append(1)
        assert monitor.get_device("auto") == "cpu"
        assert cleared == []


class TestSafeThreshold:
    def test_unlocks_when_usage_drops_below_safe(self, monitor):
        monitor._get_mps_usage = lambda: 0.50
        assert monitor.get_device("auto") == "cpu"
        assert monitor.is_locked_to_cpu() is True

        monitor._last_check = 0.0
        monitor._get_mps_usage = lambda: 0.10
        assert monitor.get_device("auto") == "mps"
        assert monitor.is_locked_to_cpu() is False

    def test_stays_unlocked_when_usage_drops_below_safe(self, monitor):
        monitor._get_mps_usage = lambda: 0.10
        assert monitor.get_device("auto") == "mps"
        assert monitor.is_locked_to_cpu() is False


class TestCaching:
    def test_check_cached_within_interval(self, monitor):
        calls = {"n": 0}
        monitor._get_mps_usage = lambda: (calls.__setitem__("n", calls["n"] + 1) or 0.30)
        monitor.get_device("auto")
        monitor.get_device("auto")
        monitor.get_device("auto")
        assert calls["n"] == 1

    def test_check_recovers_after_interval(self, monitor):
        calls = {"n": 0}
        monitor._get_mps_usage = lambda: (calls.__setitem__("n", calls["n"] + 1) or 0.0)
        monitor.get_device("auto")
        monitor._last_check = 0.0
        monitor.get_device("auto")
        assert calls["n"] == 2


class TestCheckMidGeneration:
    def test_safe_usage_returns_true(self, monitor):
        monitor._get_mps_usage = lambda: 0.20
        assert monitor.check_mid_generation() is True

    def test_high_usage_clears_and_returns_false(self, monitor):
        usage = {"value": 0.50}
        monitor._get_mps_usage = lambda: usage["value"]
        cleared = []
        monitor._clear_mps_cache = lambda: (cleared.append(1) or usage.__setitem__("value", 0.60))
        assert monitor.check_mid_generation() is False
        assert monitor.is_locked_to_cpu() is True
        assert cleared == [1]

    def test_high_usage_recovers_after_clear(self, monitor):
        usage = {"value": 0.50}
        monitor._get_mps_usage = lambda: usage["value"]
        cleared = []
        monitor._clear_mps_cache = lambda: (cleared.append(1) or usage.__setitem__("value", 0.20))
        assert monitor.check_mid_generation() is True
        assert monitor.is_locked_to_cpu() is False

    def test_locked_returns_false_within_cache(self, monitor):
        monitor.force_cpu()
        monitor._last_check = time.time()
        assert monitor.check_mid_generation() is False

    def test_usage_exception_returns_not_locked(self, monitor):
        monitor._get_mps_usage = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        assert monitor.check_mid_generation() is True


class TestState:
    def test_force_cpu_locks(self, monitor):
        monitor.force_cpu()
        assert monitor.is_locked_to_cpu() is True

    def test_reset_unlocks_and_clears_usage(self, monitor):
        monitor.force_cpu()
        monitor._last_usage = 0.5
        monitor.reset()
        assert monitor.is_locked_to_cpu() is False
        assert monitor.get_usage() == 0.0

    def test_get_usage_initial_zero(self, monitor):
        assert monitor.get_usage() == 0.0

    def test_get_usage_after_check(self, monitor):
        monitor._get_mps_usage = lambda: 0.25
        monitor.get_device("auto")
        assert monitor.get_usage() == pytest.approx(0.25)

    def test_clear_cache_is_safe(self, monitor):
        monitor._clear_mps_cache()

    def test_get_mps_usage_returns_float(self, monitor):
        usage = monitor._get_mps_usage()
        assert isinstance(usage, float)
        assert 0.0 <= usage <= 1.0


class TestGetMpsMonitor:
    def test_singleton(self):
        assert get_mps_monitor() is get_mps_monitor()
        assert isinstance(get_mps_monitor(), MPSMemoryMonitor)
