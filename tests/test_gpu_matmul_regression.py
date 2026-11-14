"""GPU accelerator ``matmul`` must accept raw numpy input (regression).

Live incident 2026-10-31 (job_c87a2dd6): training crawled at 0.4 steps/s with a
677h ETA and a flood of ``GPU matmul failed, falling back to CPU: Argument 'a'
has incorrect type (expected cupy._core.core._ndarray_base, got numpy.ndarray)``.
Every large matmul threw, was caught by ``slonet._matmul``, and fell back to
``np.matmul`` — the GPU was never used.

Root cause: the on-device guard ``hasattr(x, "device")`` is defeated by
NumPy >= 2, which added ``np.ndarray.device`` (== "cpu"). A CPU numpy array was
mistaken for a GPU cupy array, so the host->device transfer was skipped and raw
numpy was fed to ``cp.matmul``. The fix detects on-device arrays by *type*
(``isinstance(x, cp.ndarray)``).

These tests call the accelerator ``matmul`` directly with numpy input — the
exact dispatch that was broken. Before the fix both raise ``TypeError``; the
older probe test never touched ``matmul``, which is why it passed with the bug
present. The suite is skipped when no CuPy GPU is present (driverless/CI).
"""

from __future__ import annotations

import numpy as np
import pytest

# Shapes far past slonet's 16384-element accelerator dispatch threshold, so the
# test exercises the same code path real training hits.
_A = np.random.RandomState(0).randn(32, 512, 768).astype(np.float32)
_B = np.random.RandomState(1).randn(768, 768).astype(np.float32)
_REF = np.matmul(_A, _B)


def _cuda_acc():
    """Return a live CuPy-backed accelerator, or skip when unavailable."""
    from domain.slolib._internal.gpu import get_accelerator as slolib_get

    acc = slolib_get()
    if acc is None or acc.name != "cuda" or not acc.is_available() or acc._cp is None:
        pytest.skip("no CuPy GPU accelerator available")
    return acc


def _legacy_cuda_acc():
    from domain.training._internal.gpu.accelerator import get_accelerator as legacy_get

    acc = legacy_get()
    if acc is None or acc.name != "cuda" or not acc.is_available() or acc._cp is None:
        pytest.skip("no legacy CuPy accelerator available")
    return acc


class TestCudaMatmulNumpyInput:
    """``cp.matmul`` rejects raw numpy; the accelerator must convert first."""

    def test_slolib_matmul_accepts_numpy_and_matches(self):
        acc = _cuda_acc()
        out = np.asarray(acc.matmul(_A, _B))
        assert out.shape == _REF.shape
        assert np.allclose(out, _REF, atol=1e-4), "GPU matmul diverged from numpy"

    def test_legacy_matmul_accepts_numpy_and_matches(self):
        acc = _legacy_cuda_acc()
        out = np.asarray(acc.matmul(_A, _B))
        assert out.shape == _REF.shape
        assert np.allclose(out, _REF, atol=1e-4), "legacy GPU matmul diverged from numpy"

    def test_slolib_matmul_accepts_precuped_input(self):
        # A second call with an already-on-device array must not double-copy or
        # break the isinstance fast path introduced by the fix.
        acc = _cuda_acc()
        a_dev = acc._cp.asarray(_A)
        out = np.asarray(acc.matmul(a_dev, _B))
        assert np.allclose(out, _REF, atol=1e-4)

    def test_matmul_does_not_raise_typeerror(self):
        # The precise failure mode: TypeError from feeding numpy to cp.matmul.
        # slonet swallows it into a CPU fallback, so assert at the source.
        for factory in (_cuda_acc, _legacy_cuda_acc):
            acc = factory()
            acc.matmul(_A, _B)  # must not raise

    def test_numpy_device_attribute_is_the_trap(self):
        # Documents the actual trap: under NumPy >= 2, hasattr(ndarray, "device")
        # is True, so any guard built on it misclassifies CPU arrays as on-device.
        # If this ever becomes False the original guard would be safe again — the
        # test pins the environment property that makes type-checking mandatory.
        assert hasattr(np.zeros(2), "device"), (
            "NumPy no longer exposes ndarray.device; revisit the hasattr guard"
        )


class TestSlonetDispatchNoFallback:
    """The real training matmul path must stay on GPU (no silent CPU fallback)."""

    def test_matmul_emits_no_gpu_fallback_warning(self, caplog):
        import logging

        import domain.training._internal.slonet as S

        acc = S._get_accelerator()
        if acc is None or acc.name != "cuda":
            pytest.skip("training accelerator is not CUDA")

        with caplog.at_level(logging.WARNING, logger="slo.training"):
            a = S.Tensor(_A.copy(), requires_grad=True)
            b = S.Tensor(_B.copy(), requires_grad=True)
            out = S._matmul(a, b)

        assert np.allclose(np.asarray(out.data), _REF, atol=1e-4)
        fallback = [r for r in caplog.records if "GPU matmul failed" in r.getMessage()]
        assert not fallback, f"training fell back to CPU: {fallback[0].getMessage()}"
