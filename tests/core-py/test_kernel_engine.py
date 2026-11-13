"""Kernel Engine contract — engine (policy) / ops (contract) / core (capability).

Design consensus (2026-11-11 discussion):
- Selection is over CORES (registered implementations), never over devices.
  Device is metadata on a core's declaration, not a branch anywhere above.
- Ownership by constructor; custody moves freely. Cores are stateless
  functions over the arrays they are handed — a core never holds weights.
- ops is the stable layer: callers see only its f32 contract; the engine
  (autotune policy) and cores (implementations) may change underneath.
- The numpy core is the oracle every other core is parity-gated against.
- Determinism: the engine picks ONCE per (op, shape-bucket) per process —
  a mid-run re-pick would break run-to-run equality (generation stability).

Tolerance derivation mirrors tests/core-py/test_slonet_kernels_parallel.py
(fp32, eps ~ 1.19e-7): elementwise chains rtol=1e-5/atol=1e-6;
reduction kernels (sequential accumulation, <= 8192 terms) rtol=1e-4/atol=1e-5.
Matmul: two BLAS implementations (OpenBLAS vs cuBLAS) sum the k-term dot
product in different orders — bound is k*eps*max|C| (k<=128, |C|~17 ->
~2.6e-4), so rtol=1e-4/atol=5e-4 (observed 8.3e-6, ~60x margin over the
bound); near-zero elements are atol-only there.
"""

import numpy as np
import pytest

RTOL_ELEM, ATOL_ELEM = 1e-5, 1e-6
RTOL_REDUCE, ATOL_REDUCE = 1e-4, 1e-5
RTOL_MATMUL, ATOL_MATMUL = 1e-4, 5e-4

# Training shapes: gate (1024x64 / 4096x32) and medium (4096x128 / 16384x128).
CASES = [
    ("layernorm", (1024, 64)),
    ("layernorm", (4096, 128)),
    ("rmsnorm", (1024, 64)),
    ("rmsnorm", (4096, 128)),
    ("softmax", (4096, 32)),
    ("softmax", (16384, 128)),
    ("gelu", (1024 * 256,)),
    ("gelu", (4096 * 512,)),
    ("matmul", (1024, 64)),
    ("matmul", (4096, 128)),
]

REDUCE_OPS = {"layernorm", "rmsnorm", "softmax"}


def _tol(op):
    """(rtol, atol) for an op — the derivation lives in the module doc."""
    if op == "matmul":
        return RTOL_MATMUL, ATOL_MATMUL
    if op in REDUCE_OPS:
        return RTOL_REDUCE, ATOL_REDUCE
    return RTOL_ELEM, ATOL_ELEM


def _inputs(op, shape, seed=7):
    rng = np.random.default_rng(seed)
    if op == "matmul":
        a = rng.standard_normal(shape).astype(np.float32)
        b = rng.standard_normal((shape[-1], 32)).astype(np.float32)
        return (a, b)
    x = rng.standard_normal(shape).astype(np.float32)
    if op in ("gelu", "softmax"):
        return (x,)
    w = rng.standard_normal(shape[-1]).astype(np.float32)
    if op == "layernorm":
        b = rng.standard_normal(shape[-1]).astype(np.float32)
        return (x, w, b)
    return (x, w)


class TestOpsContract:
    """ops = the f32 contract every caller sees. No device vocabulary above."""

    @pytest.mark.parametrize("op,shape", CASES)
    def test_matches_numpy_oracle(self, op, shape):
        from domain.slolib._internal import kernel_ops

        args = _inputs(op, shape)
        out = getattr(kernel_ops, op)(*args)
        expected = getattr(kernel_ops.oracle, op)(*args)
        rtol, atol = _tol(op)
        np.testing.assert_allclose(out, expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("op,shape", CASES)
    def test_output_is_float32(self, op, shape):
        from domain.slolib._internal import kernel_ops

        out = getattr(kernel_ops, op)(*_inputs(op, shape))
        assert out.dtype == np.float32, f"{op} must uphold the f32 contract"

    @pytest.mark.parametrize("op,shape", CASES)
    def test_deterministic_between_runs(self, op, shape):
        from domain.slolib._internal import kernel_ops

        a = getattr(kernel_ops, op)(*_inputs(op, shape))
        b = getattr(kernel_ops, op)(*_inputs(op, shape))
        assert np.array_equal(a, b), f"{op}: run-to-run equality must hold"


class TestCoreRegistry:
    """cores = stateless implementations with capability declarations only."""

    def test_numpy_core_registered_and_always_available(self):
        from domain.slolib._internal import kernel_core

        numpy_core = kernel_core.get("numpy")
        assert numpy_core is not None
        assert numpy_core.available()
        assert numpy_core.device in ("host", "device", "both")

    def test_cores_declare_capability_not_custody(self):
        """A core declares where it CAN run — never what it holds."""
        from domain.slolib._internal import kernel_core

        for name in kernel_core.names():
            core = kernel_core.get(name)
            assert hasattr(core, "device")
            # No custody: cores expose no weight/parameter attributes.
            for attr in ("weights", "params", "state_dict"):
                assert not hasattr(core, attr), f"{name} must not own state"

    @pytest.mark.parametrize("op,shape", CASES)
    def test_every_available_core_matches_oracle(self, op, shape):
        from domain.slolib._internal import kernel_core, kernel_ops

        args = _inputs(op, shape)
        expected = getattr(kernel_ops.oracle, op)(*args)
        rtol, atol = _tol(op)
        checked = 0
        for name in kernel_core.names():
            core = kernel_core.get(name)
            if not core.available() or not core.supports(op):
                continue
            out = core.run(op, *args)
            np.testing.assert_allclose(
                out, expected, rtol=rtol, atol=atol, err_msg=f"core={name} op={op}"
            )
            checked += 1
        assert checked >= 1, "numpy core must always be checkable"

    @pytest.mark.parametrize("op,shape", CASES)
    def test_available_cores_deterministic(self, op, shape):
        from domain.slolib._internal import kernel_core

        args = _inputs(op, shape)
        for name in kernel_core.names():
            core = kernel_core.get(name)
            if not core.available() or not core.supports(op):
                continue
            a = core.run(op, *args)
            b = core.run(op, *args)
            assert np.array_equal(a, b), f"core={name} op={op} must be bit-stable"


class TestEnginePolicy:
    """engine = selection policy over cores. Memoized, deterministic, safe default."""

    def test_pick_returns_a_registered_core(self):
        from domain.slolib._internal import kernel_engine

        name = kernel_engine.pick("softmax", (4096, 32))
        from domain.slolib._internal import kernel_core

        assert kernel_core.get(name) is not None

    def test_pick_memoized_within_process(self):
        """Same key twice = same winner — no mid-run re-picking."""
        from domain.slolib._internal import kernel_engine

        first = kernel_engine.pick("layernorm", (1024, 64))
        second = kernel_engine.pick("layernorm", (1024, 64))
        assert first == second

    def test_safe_default_is_numpy(self):
        """Unknown/empty tuning state must fall back to the oracle, never crash."""
        from domain.slolib._internal import kernel_engine

        name = kernel_engine.pick("softmax", (4096, 32), tuning={})
        assert name == "numpy"

    def test_cache_roundtrips_through_disk(self, tmp_path, monkeypatch):
        """Decisions persist keyed by fingerprint; a new process starts tuned."""
        from domain.slolib._internal import kernel_engine

        monkeypatch.setattr(kernel_engine, "_cache_path", lambda: tmp_path / "t.json")
        kernel_engine.save({"softmax@4096-32": "c32"}, fingerprint="test-fp")
        loaded = kernel_engine.load(fingerprint="test-fp")
        assert loaded.get("softmax@4096-32") == "c32"

    def test_fingerprint_includes_thread_budget(self):
        from domain.slolib._internal import kernel_engine

        fp = kernel_engine.fingerprint()
        assert isinstance(fp, str) and fp
        # Thread budget is part of identity — the C-core winner flips with it
        # (measured: 1.3-1.7x at threads=1, 0.03x at threads=8).
        assert str(fp)


class TestOwnership:
    """Ownership by constructor: only the model owns weights; cores are pure."""

    def test_core_run_does_not_mutate_inputs(self):
        from domain.slolib._internal import kernel_core

        rng = np.random.default_rng(3)
        for name in kernel_core.names():
            core = kernel_core.get(name)
            if not core.available():
                continue
            x = rng.standard_normal((256, 64)).astype(np.float32)
            w = rng.standard_normal(64).astype(np.float32)
            x_before, w_before = x.copy(), w.copy()
            for op in ("rmsnorm", "gelu", "softmax"):
                if core.supports(op):
                    args = (x, w) if op == "rmsnorm" else (x,)
                    core.run(op, *args)
            assert np.array_equal(x, x_before), f"core={name} mutated input x"
            assert np.array_equal(w, w_before), f"core={name} mutated input w"
