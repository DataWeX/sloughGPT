"""Native tensor kernels — parity, kill-switch, fallbacks (card d74e2cef, phase 2).

tensor_kernels.c compiled to two engines (gcc+OpenMP ctypes / NVRTC
RawKernel) via domain/slolib/_internal/native.py and routed through
TensorDevice's ioctl handlers behind SLO_TENSOR_NATIVE. Claims under
test:

  1. CAPS fills honestly once the engine exists (RED before routing:
     a compiler present but engine=numpy fails — not skips).
  2. Routed results == numpy oracle within rtol=1e-9 (float64; gcc
     fuses multiply-adds, libm vs numpy differ ~1 ulp, summation order
     differs — all inside tolerance).
  3. SLO_TENSOR_NATIVE=0 forces the oracle and the engine is never
     queried; =1 routes through it (mechanism spy, not results).
  4. Everything the kernel does not take (float32, 3-D softmax,
     matmul) stays byte-identical on the numpy path.

Run:
    PYTHONPATH=packages/downcraft:packages/core-py scripts/python \
        -m pytest packages/core-py/tests/test_tensor_native.py -q
"""

from __future__ import annotations

import ctypes
import shutil

import numpy as np
import pytest

from domain.shell._internal import tensor_device as tdmod
from domain.shell._internal.tensor_device import TensorDevice

RTOL = 1e-9
ATOL = 1e-12


def _has_compiler() -> bool:
    return shutil.which("gcc") is not None or shutil.which("clang") is not None


def _cuda_usable() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


needs_compiler = pytest.mark.skipif(
    not _has_compiler(), reason="no C compiler — numpy stays the engine"
)
needs_cuda = pytest.mark.skipif(not _cuda_usable(), reason="no CUDA device")

rng = np.random.default_rng(11)


def _require_native(td: TensorDevice) -> None:
    """Compiler present ⇒ capability must be filled. Fails, never skips —
    this is the RED watch carrying into every parity assertion."""
    b = td.caps()["backend"]
    assert b["native_available"] is True, "native engine not compiled"
    assert b["engine"] == "native-cpu"


def _pair(td: TensorDevice, run, monkeypatch) -> tuple[np.ndarray, np.ndarray]:
    """run() under the oracle, then under native. Returns (oracle, native)."""
    monkeypatch.setenv("SLO_TENSOR_NATIVE", "0")
    oracle = run()
    monkeypatch.setenv("SLO_TENSOR_NATIVE", "1")
    native = run()
    return oracle, native


def _assert_parity(oracle, native) -> None:
    assert oracle.shape == native.shape
    assert native.dtype == np.float64
    np.testing.assert_allclose(native, oracle, rtol=RTOL, atol=ATOL)


# ── 1. CAPS fills honestly ─────────────────────────────────────────────────


class TestCapsPhase2:
    @needs_compiler
    def test_backend_reports_native_engine(self):
        b = TensorDevice("tensor").caps()["backend"]
        assert b["native_available"] is True
        assert b["engine"] == "native-cpu"
        assert b["native_ops"] == sorted(b["native_ops"])
        # exact routing set — changing it requires benchmark evidence
        assert b["native_ops"] == ["GELU", "LAYER_NORM", "RMS_NORM", "SOFTMAX"]

    @needs_compiler
    def test_parity_covers_every_routed_op(self):
        caps = TensorDevice("tensor").caps()
        assert caps["parity"] == caps["backend"]["native_ops"]


# ── 2. CPU parity vs the numpy oracle ──────────────────────────────────────


class TestParityCPU:
    @needs_compiler
    def test_softmax_rows(self, monkeypatch):
        td = TensorDevice("tensor")
        _require_native(td)
        x = rng.standard_normal((64, 48)) * 5.0
        oracle, native = _pair(td, lambda: td.ioctl("SOFTMAX", x).value, monkeypatch)
        _assert_parity(oracle, native)

    @needs_compiler
    def test_softmax_large_dynamic_range(self, monkeypatch):
        # row-max shift keeps exp stable; oracle must agree anyway
        td = TensorDevice("tensor")
        _require_native(td)
        x = rng.standard_normal((32, 64)) * 800.0
        oracle, native = _pair(td, lambda: td.ioctl("SOFTMAX", x).value, monkeypatch)
        _assert_parity(oracle, native)

    @needs_compiler
    def test_softmax_1d(self, monkeypatch):
        td = TensorDevice("tensor")
        _require_native(td)
        x = rng.standard_normal(128)
        oracle, native = _pair(td, lambda: td.ioctl("SOFTMAX", x).value, monkeypatch)
        _assert_parity(oracle, native)

    @needs_compiler
    def test_softmax_global_axis_none(self, monkeypatch):
        # the VM SOFTMAX opcode calls softmax(a, axis=None) — global form
        td = TensorDevice("tensor")
        _require_native(td)
        x = rng.standard_normal((3, 4))
        oracle, native = _pair(td, lambda: td.softmax(x, axis=None), monkeypatch)
        _assert_parity(oracle, native)
        assert oracle.shape == x.shape

    @needs_compiler
    def test_layernorm_affine(self, monkeypatch):
        td = TensorDevice("tensor")
        _require_native(td)
        x = rng.standard_normal((40, 32))
        w = rng.standard_normal(32)
        b = rng.standard_normal(32)
        oracle, native = _pair(td, lambda: td.ioctl("LAYER_NORM", x, w, b).value, monkeypatch)
        _assert_parity(oracle, native)

    @needs_compiler
    def test_layernorm_global_weightless(self, monkeypatch):
        td = TensorDevice("tensor")
        _require_native(td)
        x = rng.standard_normal((5, 7))
        oracle, native = _pair(td, lambda: td._layernorm_global(x), monkeypatch)
        _assert_parity(oracle, native)
        assert oracle.shape == x.shape

    @needs_compiler
    def test_rmsnorm_affine(self, monkeypatch):
        td = TensorDevice("tensor")
        _require_native(td)
        x = rng.standard_normal((40, 32))
        w = rng.standard_normal(32)
        oracle, native = _pair(td, lambda: td.ioctl("RMS_NORM", x, w).value, monkeypatch)
        _assert_parity(oracle, native)

    @needs_compiler
    def test_rmsnorm_global_weightless(self, monkeypatch):
        td = TensorDevice("tensor")
        _require_native(td)
        x = rng.standard_normal((5, 7))
        oracle, native = _pair(td, lambda: td._rmsnorm_global(x), monkeypatch)
        _assert_parity(oracle, native)
        assert oracle.shape == x.shape

    @needs_compiler
    def test_gelu(self, monkeypatch):
        td = TensorDevice("tensor")
        _require_native(td)
        x = rng.standard_normal(5003) * 3.0
        oracle, native = _pair(td, lambda: td.ioctl("GELU", x).value, monkeypatch)
        _assert_parity(oracle, native)

    @needs_compiler
    def test_relu_kernel_cpu_entry_vs_oracle(self):
        """RELU is unrouted (0.97x) but the C kernel still ships — its
        CPU entry is exercised directly against the numpy oracle;
        GPU-vs-oracle is in TestParityGPU."""
        eng = tdmod._build_engines("RELU")
        assert eng is not None and eng.cpu is not None
        fn = eng.cpu.tens_relu_cpu
        x = np.ascontiguousarray(rng.standard_normal(1000))
        x[7] = np.nan
        x[13] = -0.0
        out = np.empty_like(x)
        _run_relu = fn  # local alias reads cleaner below
        _run_relu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
        _run_relu.restype = None
        _run_relu(tdmod._ptr(x), tdmod._ptr(out), x.size)
        oracle = np.maximum(0, x)
        _assert_parity(oracle, out)


# ── 3. kill-switch mechanism ───────────────────────────────────────────────


class TestKillSwitch:
    def test_disabled_never_queries_the_engine(self, monkeypatch):
        td = TensorDevice("tensor")

        def spy(cmd):
            raise AssertionError("engine queried while disabled")

        monkeypatch.setattr(tdmod, "_build_engines", spy)  # loader, not the gate
        monkeypatch.setenv("SLO_TENSOR_NATIVE", "0")
        res = td.ioctl("SOFTMAX", np.zeros((4, 4)))
        assert res.success, res.error

    def test_enabled_queries_the_engine(self, monkeypatch):
        td = TensorDevice("tensor")
        queried: list[str] = []

        def spy(cmd):
            queried.append(cmd)
            return None  # pretend no compile — handler falls back, no crash

        monkeypatch.setattr(tdmod, "_build_engines", spy)  # loader, not the gate
        monkeypatch.setenv("SLO_TENSOR_NATIVE", "1")
        res = td.ioctl("SOFTMAX", np.zeros((4, 4)))
        assert res.success, res.error
        assert queried == ["SOFTMAX"]


# ── 4. fallbacks stay byte-identical ───────────────────────────────────────


class TestFallbacks:
    def test_float32_stays_on_the_oracle(self, monkeypatch):
        td = TensorDevice("tensor")
        x = rng.standard_normal((16, 8)).astype(np.float32)
        oracle, native = _pair(td, lambda: td.ioctl("SOFTMAX", x).value, monkeypatch)
        assert native.dtype == np.float32
        np.testing.assert_array_equal(native, oracle)

    def test_3d_softmax_stays_on_the_oracle(self, monkeypatch):
        # attention scores are 3-D — the kernel takes <=2-D only
        td = TensorDevice("tensor")
        x = rng.standard_normal((2, 5, 7))
        oracle, native = _pair(td, lambda: td.ioctl("SOFTMAX", x).value, monkeypatch)
        np.testing.assert_array_equal(native, oracle)

    def test_matmul_control_stays_blas(self, monkeypatch):
        # matmul is deliberately unrouted: BLAS owns it. Native flag on
        # or off, the result must be byte-identical.
        td = TensorDevice("tensor")
        a = rng.standard_normal((33, 17))
        b = rng.standard_normal((17, 29))
        oracle, native = _pair(td, lambda: td.ioctl("MATMUL", a, b).value, monkeypatch)
        np.testing.assert_array_equal(native, oracle)

    def test_relu_control_stays_oracle(self, monkeypatch):
        # RELU is deliberately unrouted (0.97x) — env on/off must be
        # byte-identical, NaN and signed zero included
        td = TensorDevice("tensor")
        x = rng.standard_normal(1000)
        x[7] = np.nan
        x[13] = -0.0
        x[21] = 0.0
        oracle, native = _pair(td, lambda: td.ioctl("RELU", x).value, monkeypatch)
        np.testing.assert_array_equal(native, oracle)


# ── 5. structural CPU-vs-GPU parity (same source, two engines) ─────────────


class TestParityGPU:
    @needs_cuda
    @needs_compiler
    @pytest.mark.parametrize(
        "cmd,make_args",
        [
            ("SOFTMAX", lambda: (rng.standard_normal((33, 21)),)),
            (
                "LAYER_NORM",
                lambda: (
                    rng.standard_normal((33, 21)),
                    rng.standard_normal(21),
                    rng.standard_normal(21),
                ),
            ),
            ("RMS_NORM", lambda: (rng.standard_normal((33, 21)), rng.standard_normal(21))),
            ("GELU", lambda: (rng.standard_normal(701),)),
            ("RELU", lambda: (rng.standard_normal(701),)),
        ],
    )
    def test_gpu_matches_cpu(self, cmd, make_args, monkeypatch):
        import cupy as cp

        td = TensorDevice("tensor")
        _require_native(td)
        args = make_args()
        monkeypatch.setenv("SLO_TENSOR_NATIVE", "1")
        cpu_out = td.ioctl(cmd, *args).value

        eng = tdmod._engines(cmd)
        assert eng is not None and eng.gpu is not None, "gpu entry did not compile"

        x = np.ascontiguousarray(args[0], dtype=np.float64)
        if cmd in ("SOFTMAX", "LAYER_NORM", "RMS_NORM"):
            rows = 1 if x.ndim == 1 else x.shape[0]
            cols = x.shape[-1]
            block = 128
            grid = ((rows + block - 1) // block, 1, 1)
            dx = cp.asarray(x)
            out = cp.empty_like(dx)
            if cmd == "SOFTMAX":
                eng.gpu(grid, (block, 1, 1), (dx, out, np.int64(rows), np.int64(cols)))
            elif cmd == "LAYER_NORM":
                dw = cp.asarray(args[1], dtype=np.float64)
                db = cp.asarray(args[2], dtype=np.float64)
                eng.gpu(
                    grid,
                    (block, 1, 1),
                    (dx, dw, db, out, np.int64(rows), np.int64(cols), np.float64(1e-5)),
                )
            else:
                dw = cp.asarray(args[1], dtype=np.float64)
                eng.gpu(
                    grid,
                    (block, 1, 1),
                    (dx, dw, out, np.int64(rows), np.int64(cols), np.float64(1e-6)),
                )
            gpu_out = cp.asnumpy(out)
        else:
            n = x.size
            block = 128
            grid = ((n + block - 1) // block, 1, 1)
            dx = cp.asarray(x)
            out = cp.empty_like(dx)
            eng.gpu(grid, (block, 1, 1), (dx, out, np.int64(n)))
            gpu_out = cp.asnumpy(out)

        np.testing.assert_allclose(gpu_out, cpu_out, rtol=RTOL, atol=ATOL)
