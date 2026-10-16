"""GPU tier tests for the cycles batch tracer (card e4f15b5a).

Device selection (cpu / cuda / auto + SLO_CYCLES_DEVICE), CPU<->GPU
statistical parity, determinism per seed, and the namespace contract:
render() and host-fed calls always hand back numpy — cupy never leaks
across the engine boundary.

Run: PYTHONPATH=packages/downcraft:packages/core-py scripts/python \
    -m pytest packages/core-py/tests/test_cycles_gpu.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from domain.shell._internal.cycles import (
    Camera,
    CyclesRenderer,
    Light,
    Material,
    Scene,
    create_plane,
    create_sphere,
    device_name,
    resolve_array_module,
)


def _cuda_usable() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


needs_cuda = pytest.mark.skipif(not _cuda_usable(), reason="no CUDA device")


def make_scene() -> Scene:
    scene = Scene(
        camera=Camera(origin=np.array([0.0, 2.0, 4.0]), look_at=np.array([0.0, 0.0, 0.0]), fov=50)
    )
    scene.materials = [
        Material(base_color=np.array([0.3, 0.3, 0.3]), roughness=0.8),
        Material(base_color=np.array([0.8, 0.1, 0.1]), roughness=0.3),
    ]
    scene.meshes = [
        create_plane(size=6, y=-1, mat_idx=0),
        create_sphere(radius=0.6, center=np.array([0, 0, 0]), segments=8, mat_idx=1),
    ]
    scene.lights = [
        Light(position=np.array([2, 3, 2]), color=np.array([1, 0.95, 0.9]), strength=6)
    ]
    scene.build_bvh()
    return scene


class TestDeviceSelection:
    def test_cpu_forces_numpy(self):
        assert resolve_array_module("cpu") is np
        assert device_name("cpu") == "cpu"

    def test_env_beats_auto(self, monkeypatch):
        monkeypatch.setenv("SLO_CYCLES_DEVICE", "cpu")
        assert resolve_array_module(None) is np

    def test_explicit_device_beats_env(self, monkeypatch):
        monkeypatch.setenv("SLO_CYCLES_DEVICE", "cuda")
        assert resolve_array_module("cpu") is np

    @needs_cuda
    def test_cuda_resolves_to_cupy(self):
        import cupy

        assert resolve_array_module("cuda") is cupy
        assert device_name("cuda") == "cuda"

    @needs_cuda
    def test_auto_resolves_on_this_machine(self):
        # This machine has a working device: auto must find it too.
        assert resolve_array_module("auto") is not np
        assert device_name() == "cuda"


class TestCPUBehavior:
    """GPU-less guarantees — these must hold on every machine."""

    def test_renderer_defaults_to_a_resolved_device(self):
        r = CyclesRenderer(make_scene(), width=8, height=6, samples=1)
        assert r._xp is not None
        assert r.device in ("auto", "cpu", "cuda")

    def test_zero_samples_returns_zeros(self):
        r = CyclesRenderer(make_scene(), width=8, height=6, samples=0, device="cpu")
        img = r.render()
        assert img.shape == (6, 8, 3)
        assert float(img.max()) == 0.0

    def test_generate_rays_origins_always_float(self):
        # An int Camera.origin used to broadcast int64 origins — every
        # bounce then C-truncated its updated origin to whole numbers.
        cam = Camera(origin=np.array([0, 2, 4]))
        origins, dirs = cam.generate_rays(8, 6)
        assert np.issubdtype(origins.dtype, np.floating)
        assert np.issubdtype(dirs.dtype, np.floating)

    def test_cpu_render_is_numpy_and_finite(self):
        r = CyclesRenderer(make_scene(), width=32, height=24, samples=2, device="cpu")
        img = r.render()
        assert type(img) is np.ndarray
        assert img.shape == (24, 32, 3)
        assert np.isfinite(img).all()
        assert 0.0 <= img.min() and img.max() <= 1.0


@needs_cuda
class TestGPURender:
    def test_render_returns_numpy_not_cupy(self):
        r = CyclesRenderer(make_scene(), width=32, height=24, samples=4, device="cuda")
        img = r.render()
        assert type(img) is np.ndarray  # not cupy, not a subclass
        assert img.dtype == np.float32
        assert np.isfinite(img).all()

    def test_parity_with_cpu(self):
        # Fused rendering keeps RNG draws host-side and identical, so the
        # two devices diverge only by fp rounding — which Russian roulette
        # can amplify on single pixels; statistically they must agree.
        rc = CyclesRenderer(make_scene(), width=48, height=32, samples=8, device="cpu")
        rg = CyclesRenderer(make_scene(), width=48, height=32, samples=8, device="cuda")
        img_c = rc.render()
        img_g = rg.render()
        mean_diff = float(np.abs(img_g - img_c).mean())
        assert mean_diff < 0.01, f"cpu/gpu statistical parity broke: {mean_diff}"

    def test_deterministic_per_seed_on_same_device(self):
        # Fresh renderer = fresh seed 42 = the identical stream (device
        # kernels are launch-deterministic: no atomics in this path).
        r1 = CyclesRenderer(make_scene(), width=32, height=24, samples=4, device="cuda")
        r2 = CyclesRenderer(make_scene(), width=32, height=24, samples=4, device="cuda")
        assert np.array_equal(r1.render(), r2.render())

    def test_host_fed_trace_stays_host(self):
        # Input-driven namespaces: a device renderer called with numpy
        # arrays runs the host path and returns numpy — no silent cupy.
        r = CyclesRenderer(make_scene(), width=16, height=12, samples=1, device="cuda")
        orig, dirs = r.scene.camera.generate_rays(16, 12)
        out = r._trace_batch(orig, dirs)
        assert type(out) is np.ndarray

    def test_device_fed_trace_stays_device_then_render_crosses(self):
        r = CyclesRenderer(make_scene(), width=16, height=12, samples=1, device="cuda")
        orig, dirs = r.scene.camera.generate_rays(16, 12)
        out = r._trace_batch(r._xp.asarray(orig), r._xp.asarray(dirs))
        assert type(out).__module__.split(".")[0] == "cupy"
        # ...and render() crosses back exactly once.
        assert type(r.render()) is np.ndarray

    def test_cuda_walk_matches_numpy_walk_and_scalar(self):
        """The fused RawKernel _mt_cuda + rewritten walk must agree with
        the numpy walk (bit-identical t/face) and the scalar walk
        (miss-mask exact, t within float32 output rounding)."""
        from domain.shell._internal.cycles import BVH

        scene = make_scene()
        bvh = BVH(scene.meshes[1], max_prims_per_node=8)
        rng = np.random.default_rng(11)
        n = 1500
        origins = rng.uniform(-2.5, 2.5, (n, 3))
        origins[:, 1] = rng.uniform(-0.9, 2.0, n)
        dirs = rng.normal(size=(n, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

        t_s = np.array([bvh.intersect(o, d)[0] for o, d in zip(origins, dirs)])
        t_np = np.asarray(bvh.intersect_batch(origins, dirs)[0])
        f_np = np.asarray(bvh.intersect_batch(origins, dirs)[1])

        import cupy as cp

        t_cp = cp.asnumpy(bvh.intersect_batch(cp.asarray(origins), cp.asarray(dirs))[0])
        f_cp = cp.asnumpy(bvh.intersect_batch(cp.asarray(origins), cp.asarray(dirs))[1])

        miss_s, miss_np, miss_cp = t_s >= 1e29, t_np >= 1e29, t_cp >= 1e29
        assert (miss_s == miss_np).all() and (miss_s == miss_cp).all()
        assert np.array_equal(t_np, t_cp) and np.array_equal(f_np, f_cp)
        hit = ~miss_s
        assert np.abs(t_np[hit] - t_s[hit]).max() < 1e-4
        assert np.abs(t_cp[hit] - t_s[hit]).max() < 1e-4
