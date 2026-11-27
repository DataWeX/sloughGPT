"""Ordering + failure-caching tests for the lazy numba KV-kernel init.

Card 20261125_123719: ``_get_nb_kv_kernels`` published ``_nb_kv_built = True``
BEFORE assigning ``_nb_kv_kernels`` (which starts life as ``None``), so any
concurrent caller passed the flag, got ``None``, and ``quantize_kv_tensor``
died with ``TypeError: cannot unpack non-iterable NoneType object``. Proven
2026-11-25: 8 threads against fresh module state -> 7/8 raise; server repro
3/3 cycles on first ``POST /chat`` after cold boot (thread scheduling on the
server hits the nanosecond window that a GIL-bound local hammer cannot).

The fix deletes the second variable: ``_nb_kv_kernels is not None`` is the
ready-test, so the flag can never be observed True while the value is None.
These tests pin both properties: the ordering invariant (deterministic — a
reader thread samples the pair while a builder thread builds), and
failure-caching semantics (a numba failure caches ``(None, None)`` instead of
rebuilding per call).
"""

import threading

from domain.infrastructure._internal import quantization


def _reset():
    """Reset module state to fresh-boot values (the race window)."""
    quantization._nb_kv_kernels = None
    if hasattr(quantization, "_nb_kv_built"):
        quantization._nb_kv_built = False


def _observe_while_building(rounds=300):
    """Sample (built-flag, value) pairs while a builder thread builds.

    Returns every observed pair. On the buggy code the builder flips
    ``_nb_kv_built = True`` before the value exists, so the pair
    ``(True, None)`` is observable; on the fixed code the ready-test IS the
    value, so ``True`` with ``None`` is structurally impossible.
    """
    observations: list[tuple] = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            built = getattr(quantization, "_nb_kv_built", None)
            value = quantization._nb_kv_kernels
            observations.append((built, value))

    def builder():
        quantization._get_nb_kv_kernels()

    r = threading.Thread(target=reader)
    r.start()
    b = threading.Thread(target=builder)
    b.start()
    b.join(timeout=60)
    stop.set()
    r.join(timeout=10)
    return observations


class TestNbKvKernelOrdering:
    def test_flag_never_true_while_value_none(self):
        """The red test: flag-before-value ordering is observable."""
        _reset()
        observations = _observe_while_building()
        _reset()
        bad = [(f, v) for f, v in observations if f and v is None]
        assert not bad, (
            f"flag observed True while value None {len(bad)}x — "
            "ready-flag published before the value (card 20261125_123719)"
        )

    def test_value_is_the_ready_test(self):
        """Fixed code exposes no separate flag to disagree with the value."""
        _reset()
        try:
            quantization._get_nb_kv_kernels()
            # After a completed build the value is authoritative; a second
            # call returns the identical object without rebuilding.
            first = quantization._nb_kv_kernels
            second = quantization._get_nb_kv_kernels()
        finally:
            _reset()
        assert first is second


class TestNbKvKernelFailureCaching:
    def test_failure_is_cached_not_rebuilt(self, monkeypatch):
        """(None, None) is non-None, so a numba failure stays cached."""
        _reset()
        calls = {"n": 0}
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "numba":
                calls["n"] += 1
                raise ImportError("no numba")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        try:
            first = quantization._get_nb_kv_kernels()
            second = quantization._get_nb_kv_kernels()
            third = quantization._get_nb_kv_kernels()
        finally:
            _reset()
        assert first == (None, None)
        assert second == (None, None)
        assert third == (None, None)
        assert calls["n"] == 1, f"numba import attempted {calls['n']}x, expected 1"

    def test_success_result_is_stable(self):
        """After a successful build, every caller gets the same tuple."""
        _reset()
        try:
            first = quantization._get_nb_kv_kernels()
            second = quantization._get_nb_kv_kernels()
        finally:
            _reset()
        assert first is second
        qk, dq = first
        if qk is not None:  # numba present in this env
            assert dq is not None
        else:  # failure path caches (None, None) — also non-None, also stable
            assert dq is None
