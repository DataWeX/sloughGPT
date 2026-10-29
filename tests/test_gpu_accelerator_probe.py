"""Bounded, probe-once accelerator detection (card 20261029_042335).

Live incident 2026-10-29 05:06: the health path's lazy ``cupy`` first-import
hung natively (CUDA probe), poisoning the import lock — every concurrent
``get_accelerator()`` caller piled onto the same import (the cache only fills
AFTER detection returns, so in-flight callers re-ran detection), and :8000
stopped serving entirely. Invariants:

  * one probe per reset epoch — concurrent first callers never re-enter
    detection;
  * the caller's wait is bounded (``_PROBE_TIMEOUT``) — a hung probe serves
    the CPU fallback immediately instead of wedging the caller;
  * failure settles permanently (no re-probe storm);
  * a late probe result supersedes the fallback once it lands;
  * ``reset_accelerator()`` discards in-flight probes (generation counter);
  * ``_CUDABackend.is_available()`` checks driver presence BEFORE importing
    cupy — a driverless machine never pays the import (the wedge itself).
"""

from __future__ import annotations

import builtins
import threading
import time

import domain.slolib._internal.gpu as gpu_mod
from domain.slolib import get_accelerator, reset_accelerator


class _ProbeFixture:
    """Installs a controllable fake detector; releases hung threads on teardown."""

    def __init__(self, monkeypatch, impl):
        self.calls: list[int] = []
        self.release = threading.Event()
        self._real = impl
        monkeypatch.setattr(gpu_mod, "_detect_best_backend", self._run)
        reset_accelerator()

    def _run(self):
        self.calls.append(1)
        return self._real()

    def teardown(self):
        self.release.set()
        reset_accelerator()


class TestProbeOnce:
    def test_concurrent_first_calls_probe_exactly_once(self, monkeypatch):
        def slow_detect():
            time.sleep(0.3)
            return gpu_mod._CPUBackend()

        fx = _ProbeFixture(monkeypatch, slow_detect)
        try:
            results = []

            def worker():
                results.append(get_accelerator())

            threads = [threading.Thread(target=worker) for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
            # The cache fills only after detection: without a probe-once guard
            # BOTH callers run detection (the incident's pile-up).
            assert len(fx.calls) == 1
            assert len(results) == 2
            assert results[0] is results[1]
        finally:
            fx.teardown()

    def test_reset_allows_fresh_reprobe(self, monkeypatch):
        fx = _ProbeFixture(monkeypatch, lambda: gpu_mod._CPUBackend())
        try:
            assert get_accelerator() is not None
            assert len(fx.calls) == 1
            reset_accelerator()
            get_accelerator()
            assert len(fx.calls) == 2
        finally:
            fx.teardown()


class TestBoundedWait:
    def test_hung_probe_serves_cpu_fallback_within_bound(self, monkeypatch):
        def hung_detect():
            self_release = fx.release
            self_release.wait(timeout=10)
            return gpu_mod._CPUBackend()

        fx = _ProbeFixture(monkeypatch, hung_detect)
        monkeypatch.setattr(gpu_mod, "_PROBE_TIMEOUT", 0.3)
        try:
            t0 = time.perf_counter()
            acc = get_accelerator()
            elapsed = time.perf_counter() - t0
            assert elapsed < 2.0, f"caller waited {elapsed:.2f}s — probe must be bounded"
            assert acc.device_type == "cpu"
        finally:
            fx.teardown()

    def test_hung_probe_never_blocks_subsequent_calls(self, monkeypatch):
        def hung_detect():
            fx.release.wait(timeout=10)
            return gpu_mod._CPUBackend()

        fx = _ProbeFixture(monkeypatch, hung_detect)
        monkeypatch.setattr(gpu_mod, "_PROBE_TIMEOUT", 0.3)
        try:
            get_accelerator()  # latches the fallback
            t0 = time.perf_counter()
            acc = get_accelerator()
            assert time.perf_counter() - t0 < 0.2, "latched fallback must be instant"
            assert acc.device_type == "cpu"
            assert len(fx.calls) == 1, "no re-probe after the bound was spent"
        finally:
            fx.teardown()

    def test_late_probe_result_supersedes_fallback(self, monkeypatch):
        def hung_detect():
            fx.release.wait(timeout=10)
            sentinel = object()
            fx.sentinel = sentinel
            return sentinel

        fx = _ProbeFixture(monkeypatch, hung_detect)
        fx.sentinel = None
        monkeypatch.setattr(gpu_mod, "_PROBE_TIMEOUT", 0.2)
        try:
            acc = get_accelerator()
            assert acc.device_type == "cpu"  # bounded fallback served
            fx.release.set()  # probe completes in the background
            deadline = time.perf_counter() + 5.0
            while time.perf_counter() < deadline:
                if get_accelerator() is fx.sentinel:
                    break
                time.sleep(0.05)
            assert get_accelerator() is fx.sentinel, "late probe result must win"
        finally:
            fx.teardown()


class TestFailureSettles:
    def test_failed_probe_falls_back_and_does_not_retry(self, monkeypatch):
        def boom():
            raise RuntimeError("detection exploded")

        fx = _ProbeFixture(monkeypatch, boom)
        try:
            acc = get_accelerator()
            assert acc.device_type == "cpu", "failed detection must fall back to CPU"
            acc2 = get_accelerator()
            assert acc2.device_type == "cpu"
            assert len(fx.calls) == 1, "failure must settle — no re-probe storm"
        finally:
            fx.teardown()


class TestCudaFailFast:
    def _spy_imports(self, monkeypatch) -> list[str]:
        attempts: list[str] = []
        real_import = builtins.__import__

        def spy(name, *args, **kwargs):
            if name == "cupy" or name.startswith("cupy."):
                attempts.append(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", spy)
        return attempts

    def test_no_driver_never_imports_cupy(self, monkeypatch):
        # The wedge itself: on a driverless machine cupy's import must not
        # run at all — it is the expensive/hanging operation being guarded.
        monkeypatch.setattr(gpu_mod, "_cuda_driver_present", lambda: False)
        attempts = self._spy_imports(monkeypatch)
        assert gpu_mod._CUDABackend().is_available() is False
        assert attempts == [], f"cupy import attempted without a driver: {attempts}"

    def test_driver_present_proceeds_to_import(self, monkeypatch):
        monkeypatch.setattr(gpu_mod, "_cuda_driver_present", lambda: True)
        attempts = self._spy_imports(monkeypatch)
        gpu_mod._CUDABackend().is_available()
        assert attempts, "with a driver present the cupy check must proceed"
