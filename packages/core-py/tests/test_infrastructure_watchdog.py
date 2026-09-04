"""Tests for HealthWatchdog — server health monitoring."""
from __future__ import annotations

import time

from domains.infrastructure.watchdog import HealthWatchdog, get_watchdog, reset_watchdog


class TestHealthWatchdog:
    def test_defaults(self):
        w = HealthWatchdog()
        assert w._running is False
        assert w._consecutive_failures == 0

    def test_set_recovery_fn(self):
        w = HealthWatchdog()
        w.set_recovery_fn(lambda: True)
        assert w._recovery_fn is not None

    def test_set_health_check_fn(self):
        w = HealthWatchdog()
        w.set_health_check_fn(lambda: True)
        assert w._health_check_fn is not None

    def test_start_stop(self):
        w = HealthWatchdog()
        w.set_health_check_fn(lambda: True)
        w.start(poll_interval=1, max_failures=2)
        assert w._running is True
        time.sleep(0.1)
        w.stop()
        assert w._running is False

    def test_recovery_triggered_on_failures(self):
        call_count = [0]

        def health_check():
            call_count[0] += 1
            return False  # always fail

        recovery_called = [False]

        def recovery():
            recovery_called[0] = True
            return True

        w = HealthWatchdog()
        w.set_health_check_fn(health_check)
        w.set_recovery_fn(recovery)
        w.start(poll_interval=1, max_failures=2)
        time.sleep(3)
        w.stop()
        assert recovery_called[0] is True


class TestSingleton:
    def test_get_watchdog(self):
        reset_watchdog()
        a = get_watchdog()
        b = get_watchdog()
        assert a is b
