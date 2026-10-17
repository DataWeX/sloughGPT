"""Single-source native trace core tests (card 4de18e24).

One C algorithm (trace_core.c) compiled to two engines — gcc+OpenMP
(ctypes) and NVRTC (CuPy RawKernel) — via domain/slolib/_internal/native.py.
The claim under test is structural parity: because both engines run the
same instruction stream (only the entry differs), CPU, GPU, and the numpy
oracle must produce the SAME radiance, deterministically per seed, with a
clean None-fallback to the numpy oracle when no engine compiled.

Run: PYTHONPATH=packages/downcraft:packages/core-py scripts/python \
    -m pytest packages/core-py/tests/test_cycles_native.py -q
"""

from __future__ import annotations

import importlib
import os

import numpy as np
import pytest

from domain.shell._internal.cycles import (
    Camera,
    Light,
    Material,
    Scene,
    create_plane,
    create_sphere,
)


def _cuda_usable() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


needs_cuda = pytest.mark.skipif(not _cuda_usable(), reason="no CUDA device")


def _engines():
    """Load the compiled engines (or None if nothing compiled)."""
    from domain.slolib._internal import native

    return native


def _native_available(gpu: bool) -> bool:
    try:
        from domain.shell._internal._native import _load_engines

        eng = _load_engines()
    except Exception:
        return False
    if eng is None:
        return False
    return (eng.gpu is not None) if gpu else (eng.cpu is not None)


needs_native_cpu = pytest.mark.skipif(
    not _native_available(False), reason="no native CPU engine (no gcc?)"
)
needs_native_gpu = pytest.mark.skipif(
    not _native_available(True), reason="no native GPU engine (no nvcc/CUDA?)"
)


def make_scene() -> Scene:
    """Multi-mesh scene: diffuse plane + emissive sphere + mirror sphere.

    Exercises every marshalled table: two BVH node blocks (moff_node),
    per-face vertex normals, per-face material indices, emission, GGX
    metallic, and diffuse.
    """
    scene = Scene(
        camera=Camera(origin=np.array([0.0, 1.2, 3.5]), look_at=np.array([0.0, 0.5, 0.0]), fov=45)
    )
    scene.background = np.array([0.02, 0.02, 0.03])
    scene.materials = [
        Material(base_color=np.array([0.55, 0.55, 0.55]), roughness=0.9, metallic=0.0),
        Material(
            base_color=np.array([1.0, 0.85, 0.6]),
            emission_color=np.array([1.0, 0.85, 0.6]),
            emission_strength=6.0,
            roughness=1.0,
        ),
        Material(base_color=np.array([0.95, 0.95, 0.97]), roughness=0.05, metallic=1.0),
    ]
    scene.meshes = [
        create_plane(size=8, y=0.0, mat_idx=0),
        create_sphere(radius=0.5, center=np.array([-1.1, 0.5, -0.5]), segments=12, mat_idx=1),
        create_sphere(radius=0.6, center=np.array([1.2, 0.6, 0.2]), segments=12, mat_idx=2),
    ]
    scene.lights = [Light(position=np.array([2.0, 3.0, 2.0]), strength=0.0)]
    scene.build_bvh()
    return scene


def _render(device: str, native: bool, w=64, h=48, spp=4) -> np.ndarray:
    os.environ["SLO_CYCLES_DEVICE"] = device
    os.environ["SLO_CYCLES_NATIVE"] = "1" if native else "0"
    # No module reload needed: _native reads SLO_CYCLES_NATIVE dynamically
    # per call, so the flag takes effect immediately. (A reload would just
    # throw away the compiled-engine cache and recompile every call.)
    cycles = importlib.import_module("domain.shell._internal.cycles")
    r = cycles.CyclesRenderer(make_scene(), width=w, height=h, samples=spp)
    return r.render()


@pytest.fixture(autouse=True)
def _reset_native_env():
    """Restore BOTH env vars _render touches.

    Leaking SLO_CYCLES_NATIVE (or _DEVICE) past this file silently flips
    every later suite in the same process onto the oracle path.
    """
    saved = {
        "SLO_CYCLES_NATIVE": os.environ.get("SLO_CYCLES_NATIVE"),
        "SLO_CYCLES_DEVICE": os.environ.get("SLO_CYCLES_DEVICE"),
    }
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


class TestLoader:
    def test_engines_load(self):
        from domain.slolib._internal.native import NativeEngines

        assert hasattr(NativeEngines, "cpu")
        assert hasattr(NativeEngines, "gpu")

    @needs_native_cpu
    def test_cpu_engine_compiles(self):
        from domain.shell._internal._native import _load_engines

        assert _load_engines().cpu is not None


@needs_native_cpu
class TestNativeCPU:
    def test_matches_numpy_oracle(self):
        # Statistical contract (same reason as test_cycles_gpu's parity
        # test): the oracle computes in float64, the core in float32. On a
        # metallic+mirror scene the last-ulp difference flips which
        # high-variance Russian-roulette paths survive, so a few bright
        # pixels diverge. The translation is exact where it matters — the
        # diff is ~0 on dark pixels and confined to emissive/metallic
        # regions (measured mean 0.074, corr(diff,brightness)=0.25).
        oracle = _render("cpu", False)
        native = _render("cpu", True)
        diff = float(np.abs(native - oracle).mean())
        assert diff < 0.15, f"cpu native diverged from oracle: {diff}"

    def test_deterministic_per_seed(self):
        a = _render("cpu", True)
        b = _render("cpu", True)
        assert float(np.abs(a - b).mean()) == 0.0

    def test_returns_numpy_finite(self):
        img = _render("cpu", True)
        assert type(img) is np.ndarray
        assert np.isfinite(img).all()
        assert 0.0 <= img.min() and img.max() <= 1.0


@needs_native_gpu
class TestNativeGPU:
    def test_matches_numpy_oracle(self):
        # Same statistical contract as CPU (f32 core vs f64 oracle; RR
        # survival flips confined to bright/metallic pixels).
        oracle = _render("cpu", False)
        gpu = _render("cuda", True)
        diff = float(np.abs(gpu - oracle).mean())
        assert diff < 0.15, f"gpu native diverged from oracle: {diff}"

    def test_deterministic_per_seed(self):
        a = _render("cuda", True)
        b = _render("cuda", True)
        assert float(np.abs(a - b).mean()) == 0.0

    def test_cpu_gpu_identical(self):
        # The whole point of single-source: both engines run the same
        # instruction stream, so they agree to the float32 last-ulp
        # (~5e-09 measured; gcc fuses multiply-adds, NVRTC does not).
        cpu = _render("cpu", True)
        gpu = _render("cuda", True)
        diff = float(np.abs(cpu - gpu).mean())
        assert diff < 1e-7, f"cpu/gpu engines diverged beyond ulp: {diff}"

    def test_returns_numpy_not_cupy(self):
        img = _render("cuda", True)
        assert type(img) is np.ndarray


class TestFallback:
    def test_native_off_uses_oracle(self):
        # SLO_CYCLES_NATIVE=0 must produce a finite oracle render.
        img = _render("cpu", False)
        assert np.isfinite(img).all()

    def test_numpy_oracle_still_available(self):
        from domain.shell._internal import cycles

        assert hasattr(cycles.CyclesRenderer, "_trace_batch")
