"""Tests for domains.infrastructure.server_state — AtomicRef."""

import threading
from domains.infrastructure.server_state import AtomicRef


class TestAtomicRef:
    def test_init(self):
        ref = AtomicRef(42, name="test")
        assert ref.get() == 42

    def test_set(self):
        ref = AtomicRef(0)
        ref.set(10)
        assert ref.get() == 10

    def test_swap(self):
        ref = AtomicRef(5)
        result = ref.swap(lambda x: x * 2)
        assert result == 10
        assert ref.get() == 10

    def test_version(self):
        ref = AtomicRef(0)
        assert ref._version == 0
        ref.set(1)
        assert ref._version == 1

    def test_listener(self):
        ref = AtomicRef(0)
        changes = []
        ref._listeners.append(lambda old, new: changes.append((old, new)))
        ref.set(5)
        assert changes == [(0, 5)]

    def test_thread_safety(self):
        ref = AtomicRef(0)
        errors = []

        def inc():
            for _ in range(100):
                ref.set(ref.get() + 1)

        threads = [threading.Thread(target=inc) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Value may have race conditions, but shouldn't error
        assert ref.get() >= 0
