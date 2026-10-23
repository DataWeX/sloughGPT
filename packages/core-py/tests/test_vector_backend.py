"""Tests for VectorBE compute backend with shared memory multiprocessing."""

import numpy as np
import pytest

from domain.infrastructure._internal.arch_config import ArchConfig
from domain.infrastructure._internal.vector_backend import VectorBE


def _make_arch():
    arch = ArchConfig(
        name="tiny_vector",
        norm="rms_norm",
        positional="rope",
        activation="swiglu",
        attention="gqa",
    )
    arch.n_layers = 1
    arch.n_heads = 4
    arch.n_kv_heads = 2
    arch.n_embed = 32
    arch.head_dim = 8
    arch.vocab_size = 64
    arch.rope_base = 10000.0
    arch.tied_weights = False
    arch.n_head = 4
    arch.n_kv_head = 2
    return arch


def _make_weights(arch):
    E = arch.n_embed
    H = arch.n_head
    KV = arch.n_kv_head
    D = arch.head_dim
    V = arch.vocab_size
    FF = 48
    weights = {}
    weights["embed_tokens.weight"] = np.random.randn(V, E).astype(np.float32) * 0.02
    i = 0
    weights[f"model.layers.{i}.self_attn.q_proj.weight"] = (
        np.random.randn(H * D, E).astype(np.float32) * 0.02
    )
    weights[f"model.layers.{i}.self_attn.k_proj.weight"] = (
        np.random.randn(KV * D, E).astype(np.float32) * 0.02
    )
    weights[f"model.layers.{i}.self_attn.v_proj.weight"] = (
        np.random.randn(KV * D, E).astype(np.float32) * 0.02
    )
    weights[f"model.layers.{i}.self_attn.o_proj.weight"] = (
        np.random.randn(E, H * D).astype(np.float32) * 0.02
    )
    weights[f"model.layers.{i}.mlp.gate_proj.weight"] = (
        np.random.randn(FF, E).astype(np.float32) * 0.02
    )
    weights[f"model.layers.{i}.mlp.up_proj.weight"] = (
        np.random.randn(FF, E).astype(np.float32) * 0.02
    )
    weights[f"model.layers.{i}.mlp.down_proj.weight"] = (
        np.random.randn(E, FF).astype(np.float32) * 0.02
    )
    weights[f"model.layers.{i}.input_layernorm.weight"] = np.ones(E, dtype=np.float32)
    weights[f"model.layers.{i}.post_attention_layernorm.weight"] = np.ones(E, dtype=np.float32)
    weights["model.norm.weight"] = np.ones(E, dtype=np.float32)
    weights["lm_head.weight"] = np.random.randn(V, E).astype(np.float32) * 0.02
    return weights


@pytest.fixture
def vector_be(monkeypatch):
    # Pin the suite's live pools to forkserver regardless of the
    # SLO_VECTOR_START_METHOD hatch: a live fork Pool inside the suite can
    # fork into leftover server threads and deadlock (class docstring of
    # TestPoolStartMethod, run-3 evidence), and default-assertions below
    # must not depend on the hatch being unset.
    monkeypatch.delenv("SLO_VECTOR_START_METHOD", raising=False)
    arch = _make_arch()
    weights = _make_weights(arch)
    be = VectorBE.from_weights(weights, arch)
    yield be
    be.__del__()


class TestVectorBEConstruction:
    def test_from_weights(self, vector_be):
        assert "vector" in vector_be.backend_name()
        assert vector_be.vocab_size() == 64
        assert vector_be.n_layers() == 1

    def test_arch_stored(self, vector_be):
        assert vector_be._arch.n_layers == 1
        assert vector_be._arch.n_head == 4


class TestVectorBESimpleOps:
    def test_silu(self, vector_be):
        x = np.array([0.0, 1.0, -1.0])
        result = vector_be.silu(x)
        assert result[0] == 0.0
        assert 0 < result[1] < 1

    def test_gelu(self, vector_be):
        x = np.array([0.0, 1.0, -1.0])
        result = vector_be.gelu(x)
        assert abs(result[0]) < 1e-6
        assert 0 < result[1] < 1

    def test_rope(self, vector_be):
        x = np.random.randn(1, 4, 4, 8).astype(np.float32)
        cos = np.ones((4, 1, 4), dtype=np.float32)
        sin = np.zeros((4, 1, 4), dtype=np.float32)
        result = vector_be.rope(x, cos, sin)
        assert result.shape == x.shape

    def test_repeat_kv(self, vector_be):
        x = np.random.randn(1, 2, 4, 8).astype(np.float32)
        result = vector_be.repeat_kv(x, 2)
        assert result.shape == (1, 4, 4, 8)

    def test_repeat_kv_noop(self, vector_be):
        x = np.random.randn(1, 4, 4, 8).astype(np.float32)
        result = vector_be.repeat_kv(x, 1)
        assert result is x

    def test_argmax(self, vector_be):
        x = np.array([1.0, 5.0, 3.0])
        assert vector_be.argmax(x) == 1

    def test_clip(self, vector_be):
        x = np.array([1.0, 5.0, 10.0])
        result = vector_be.clip(x, 2.0, 8.0)
        np.testing.assert_array_equal(result, [2.0, 5.0, 8.0])

    def test_from_to_numpy(self, vector_be):
        arr = np.array([1, 2, 3])
        assert vector_be.from_numpy(arr) is arr
        assert vector_be.to_numpy(arr) is arr


class TestVectorBESharedMemoryOps:
    def test_matmul_2d(self, vector_be):
        a = np.random.randn(4, 8).astype(np.float32)
        b = np.random.randn(8, 3).astype(np.float32)
        result = vector_be.matmul(a, b)
        np.testing.assert_allclose(result, a @ b, atol=1e-5)

    def test_matmul_fallback(self, vector_be):
        a = np.random.randn(4, 8, 3).astype(np.float32)
        b = np.random.randn(3, 2).astype(np.float32)
        result = vector_be.matmul(a, b)
        np.testing.assert_allclose(result, a @ b, atol=1e-5)

    def test_softmax_2d(self, vector_be):
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = vector_be.softmax(x)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_softmax_fallback(self, vector_be):
        x = np.random.randn(2, 3, 4).astype(np.float32)
        result = vector_be.softmax(x, axis=-1)
        np.testing.assert_allclose(result.sum(axis=-1), np.ones((2, 3)), atol=1e-6)

    def test_rmsnorm_2d(self, vector_be):
        x = np.random.randn(4, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        result = vector_be.rmsnorm(x, w)
        assert result.shape == x.shape
        assert result.dtype == np.float32

    def test_rmsnorm_1d_fallback(self, vector_be):
        x = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        w = np.ones(3, dtype=np.float32)
        result = vector_be.rmsnorm(x, w)
        assert result.shape == x.shape


class TestVectorBEForward:
    def test_forward_produces_logits(self, vector_be):
        tokens = np.array([[1, 2, 3]], dtype=np.int64)
        logits = vector_be.forward(tokens)
        assert logits.shape == (1, 3, 64)
        assert logits.dtype == np.float32

    def test_generate_yields_tokens(self, vector_be):
        tokens = np.array([[1]], dtype=np.int64)
        gen = list(vector_be.generate_stream(tokens, max_new_tokens=5, temperature=0.0))
        assert len(gen) <= 5
        assert all(isinstance(t, int) for t in gen)

    def test_generate_returns_metrics(self, vector_be):
        tokens = np.array([[1]], dtype=np.int64)
        all_ids, metrics = vector_be.generate(tokens, max_new_tokens=3, temperature=0.0)
        assert "n_tokens" in metrics
        assert "ttft_ms" in metrics
        assert "tokens_per_sec" in metrics
        assert all_ids.shape[1] >= 1

    def test_generate_eos_stops(self, vector_be):
        tokens = np.array([[1]], dtype=np.int64)
        all_ids, metrics = vector_be.generate(
            tokens, max_new_tokens=50, temperature=0.0, eos_token=1
        )
        assert metrics["n_tokens"] <= 50


class TestPoolStartMethod:
    """Pool start method: forkserver by default, `fork` escape via env (card a0ba949b).

    The bare default context = fork deadlocks workers when the parent is
    multithreaded (OpenBLAS) — fork-into-multithreaded, run-7/run-8 signature.
    The fork escape is asserted ctx-level only: constructing a live fork Pool
    here would fork into the suite's leftover server threads — workers deadlock
    at bootstrap and `VectorBE.__del__`'s join wedged run-3 (2 timeout blocks,
    3 orphaned workers). The real-pool wiring runs via `vector_be` on the safe
    forkserver default instead.
    """

    def test_helper_defaults_to_forkserver(self, monkeypatch):
        from domain.infrastructure._internal.vector_backend import _pool_start_method

        # A default is only a default with the hatch unset — prove it
        # regardless of the ambient SLO_VECTOR_START_METHOD.
        monkeypatch.delenv("SLO_VECTOR_START_METHOD", raising=False)
        assert _pool_start_method() == "forkserver"

    def test_helper_fork_escape_env(self, monkeypatch):
        from domain.infrastructure._internal.vector_backend import _pool_start_method

        monkeypatch.setenv("SLO_VECTOR_START_METHOD", "fork")
        assert _pool_start_method() == "fork"

    def test_pool_context_is_forkserver(self, vector_be):
        assert vector_be._pool._ctx.get_start_method() == "forkserver"

    def test_pool_context_fork_escape(self, monkeypatch):
        import multiprocessing as mp

        from domain.infrastructure._internal.vector_backend import _pool_start_method

        monkeypatch.setenv("SLO_VECTOR_START_METHOD", "fork")
        # Ctx-level only — no live Pool (see class docstring: live fork workers
        # deadlocked into the suite's server threads in run-3).
        assert mp.get_context(_pool_start_method()).get_start_method() == "fork"


class TestResourceTrackerHygiene:
    """card ccad389e: no /psm_* resource_tracker warning flood at exit.

    Subprocess by design: the warnings surface when pool WORKERS and the
    tracker daemon exit, and the class above documents that live fork pools
    deadlock the suite's server threads (run-3). One subprocess owns the
    whole lifecycle — construct -> dispatch -> teardown -> interpreter
    exit -> worker/daemon reap — with every descendant's stderr (the
    tracker daemons inherit fd 2) captured by a single pipe.

    Acceptance (card ccad389e): zero resource_tracker warnings, no leaked
    /dev/shm segments, BOTH start methods green.
    """

    _LIFECYCLE_SCRIPT = """
import importlib.util, sys, warnings
warnings.simplefilter("always")
sys.path.insert(0, ".")
sys.path.insert(0, "packages/core-py")
spec = importlib.util.spec_from_file_location(
    "tvb", "packages/core-py/tests/test_vector_backend.py"
)
tvb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tvb)
import numpy as np
from domain.infrastructure._internal.vector_backend import VectorBE
arch = tvb._make_arch()
weights = tvb._make_weights(arch)
be = VectorBE.from_weights(weights, arch)
be.forward(np.random.randint(0, arch.vocab_size, size=(1, 4)).astype(np.int64))
# The flood mechanism is WORKER-side attach (gh-82344): only the 2D
# pool-dispatched primitives (_matmul_worker/_softmax_worker/...) make
# workers call SharedMemory(name=...) — forward()'s 3D tensors stay
# in-process. Exercise them explicitly or the test is vacuous.
be.matmul(np.random.rand(256, 128).astype(np.float32),
          np.random.rand(128, 64).astype(np.float32))
be.softmax(np.random.rand(512, 64).astype(np.float32))
del be
import gc
gc.collect()
"""

    @pytest.mark.parametrize("method", ["fork", "forkserver"])
    def test_lifecycle_emits_no_resource_tracker_warnings(self, method):
        import os
        import subprocess
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[3]
        before = {p.name for p in Path("/dev/shm").glob("psm_*")}
        env = {
            **os.environ,
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": f"{root}{os.pathsep}{root / 'packages' / 'core-py'}",
            "SLO_VECTOR_START_METHOD": method,
        }
        proc = subprocess.run(
            [sys.executable, "-c", self._LIFECYCLE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=root,
            env=env,
        )
        assert proc.returncode == 0, f"[{method}] lifecycle crashed:\n{proc.stderr}"
        assert "resource_tracker" not in proc.stderr, (
            f"[{method}] resource_tracker warning flood:\n{proc.stderr[-2000:]}"
        )
        after = {p.name for p in Path("/dev/shm").glob("psm_*")}
        assert after == before, f"[{method}] leaked /dev/shm segments: {after - before}"
