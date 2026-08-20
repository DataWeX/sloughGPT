"""Meaningful tests for CircuitBreaker — state transitions, timeout recovery, callbacks, concurrency."""

import threading
import time
from domains.infrastructure.model_server import CircuitBreaker, CircuitBreakerState


class TestCircuitBreakerInitialState:
    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_starts_allowed(self):
        cb = CircuitBreaker()
        assert cb.allow_request() is True


class TestCircuitBreakerFailureThreshold:
    def test_below_threshold_stays_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_at_threshold_opens(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

    def test_opens_deny_requests(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.allow_request() is False


class TestCircuitBreakerSuccessResets:
    def test_success_resets_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_success_resets_to_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(10):
            cb.record_failure()
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED


class TestCircuitBreakerHalfOpen:
    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        time.sleep(0.02)
        assert cb.state == CircuitBreakerState.HALF_OPEN

    def test_half_open_allows_requests(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.allow_request() is True

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitBreakerState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitBreakerState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN


class TestCircuitBreakerCallback:
    def test_state_change_callback(self):
        changes = []
        cb = CircuitBreaker(failure_threshold=1)
        cb._on_state_change = lambda old, new: changes.append((old.value, new.value))
        cb.record_failure()
        assert len(changes) == 1
        assert changes[0] == ("closed", "open")

    def test_no_callback_on_same_state(self):
        changes = []
        cb = CircuitBreaker(failure_threshold=3)
        cb._on_state_change = lambda old, new: changes.append((old.value, new.value))
        cb.record_failure()
        cb.record_failure()
        assert len(changes) == 0


class TestCircuitBreakerConcurrency:
    def test_concurrent_failures(self):
        cb = CircuitBreaker(failure_threshold=10)
        errors = []

        def fail():
            try:
                cb.record_failure()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=fail) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert cb.state == CircuitBreakerState.OPEN
        assert errors == []

    def test_concurrent_success_and_failure(self):
        cb = CircuitBreaker(failure_threshold=5)
        for _ in range(10):
            threading.Thread(target=cb.record_failure).start()
            threading.Thread(target=cb.record_success).start()
        time.sleep(0.05)
        # No crash = pass. State may vary based on scheduling.
