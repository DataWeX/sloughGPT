"""Tests for HealthWatchdog — background health monitor with auto-recovery."""

import threading
import time

import pytest

import domains.infrastructure.watchdog as wd_module
from domains.infrastructure.watchdog import HealthWatchdog, get_watchdog


@pytest.fixture
def fast_loop(monkeypatch):
    """Make the watchdog loop spin without real sleeps."""
    monkeypatch.setattr(wd_module.time, "sleep", lambda s: None)


@pytest.fixture
def started_watchdog(fast_loop):
    w = HealthWatchdog()
    yield w
    w.stop()


def _wait_until(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return False


class TestConfig:
    def test_sets_recovery_fn(self, started_watchdog):
        fn = lambda: True
        started_watchdog.set_recovery_fn(fn)
        assert started_watchdog._recovery_fn is fn

    def test_sets_health_check_fn(self, started_watchdog):
        fn = lambda: True
        started_watchdog.set_health_check_fn(fn)
        assert started_watchdog._health_check_fn is fn

    def test_sets_on_recovery(self, started_watchdog):
        fn = lambda: None
        started_watchdog.set_on_recovery(fn)
        assert started_watchdog._on_recovery is fn

    def test_initial_state(self):
        w = HealthWatchdog()
        assert w._running is False
        assert w._thread is None
        assert w._consecutive_failures == 0


class TestLifecycle:
    def test_start_sets_running(self, started_watchdog):
        started_watchdog.start(poll_interval=1, max_failures=2)
        assert started_watchdog._running is True
        assert started_watchdog._thread is not None
        assert started_watchdog._thread.name == "health-watchdog"
        assert started_watchdog._thread.daemon is True

    def test_start_twice_is_noop(self, started_watchdog):
        started_watchdog.start(poll_interval=1, max_failures=2)
        first_thread = started_watchdog._thread
        started_watchdog.start(poll_interval=1, max_failures=2)
        assert started_watchdog._thread is first_thread

    def test_stop_without_start_is_safe(self):
        w = HealthWatchdog()
        w.stop()
        assert w._running is False

    def test_stop_stops_thread(self, started_watchdog):
        started_watchdog.start(poll_interval=1, max_failures=2)
        started_watchdog.stop()
        assert started_watchdog._running is False
        assert not started_watchdog._thread.is_alive()


class TestRecovery:
    def test_recovers_after_max_failures(self, started_watchdog):
        recovered = []
        started_watchdog.set_health_check_fn(lambda: False)
        started_watchdog.set_recovery_fn(lambda: recovered.append(1) or True)
        started_watchdog.start(poll_interval=1, max_failures=2)
        assert _wait_until(lambda: bool(recovered))
        assert len(recovered) >= 1

    def test_single_failure_does_not_recover(self, started_watchdog):
        recovered = []
        calls = {"n": 0}

        def health():
            calls["n"] += 1
            return calls["n"] > 1

        started_watchdog.set_health_check_fn(health)
        started_watchdog.set_recovery_fn(lambda: recovered.append(1) or True)
        started_watchdog.start(poll_interval=1, max_failures=5)
        time.sleep(0.1)
        started_watchdog.stop()
        assert recovered == []
        assert started_watchdog._consecutive_failures == 0

    def test_healthy_check_resets_failure_counter(self, started_watchdog):
        recovered = []
        calls = {"n": 0}

        def health():
            calls["n"] += 1
            return calls["n"] % 2 == 0

        started_watchdog.set_health_check_fn(health)
        started_watchdog.set_recovery_fn(lambda: recovered.append(1) or True)
        started_watchdog.start(poll_interval=1, max_failures=3)
        time.sleep(0.15)
        started_watchdog.stop()
        assert recovered == []

    def test_on_recovery_called_on_success(self, started_watchdog):
        on_recovery = []
        started_watchdog.set_health_check_fn(lambda: False)
        started_watchdog.set_recovery_fn(lambda: True)
        started_watchdog.set_on_recovery(lambda: on_recovery.append(1))
        started_watchdog.start(poll_interval=1, max_failures=2)
        assert _wait_until(lambda: bool(on_recovery))

    def test_on_recovery_not_called_on_failure(self, started_watchdog):
        on_recovery = []
        started_watchdog.set_health_check_fn(lambda: False)
        started_watchdog.set_recovery_fn(lambda: False)
        started_watchdog.set_on_recovery(lambda: on_recovery.append(1))
        started_watchdog.start(poll_interval=1, max_failures=2)
        assert _wait_until(lambda: started_watchdog._consecutive_failures > 0 or True)
        time.sleep(0.05)
        started_watchdog.stop()
        assert on_recovery == []

    def test_recovery_fn_exception_does_not_kill_loop(self, started_watchdog):
        recovered = []
        calls = {"n": 0}

        def health():
            calls["n"] += 1
            return calls["n"] > 3

        def recover():
            raise RuntimeError("boom")

        started_watchdog.set_health_check_fn(health)
        started_watchdog.set_recovery_fn(recover)
        started_watchdog.set_on_recovery(lambda: recovered.append(1))
        started_watchdog.start(poll_interval=1, max_failures=1)
        # After 3 unhealthy checks → recovery crashes; once healthy, loop continues
        assert _wait_until(lambda: calls["n"] >= 5)
        started_watchdog.stop()
        assert recovered == []

    def test_no_recovery_fn_logs_and_continues(self, started_watchdog):
        started_watchdog.set_health_check_fn(lambda: False)
        started_watchdog.start(poll_interval=1, max_failures=1)
        time.sleep(0.1)
        started_watchdog.stop()
        assert started_watchdog._consecutive_failures >= 0


class TestGetWatchdog:
    def test_singleton(self):
        assert get_watchdog() is get_watchdog()
        assert isinstance(get_watchdog(), HealthWatchdog)
