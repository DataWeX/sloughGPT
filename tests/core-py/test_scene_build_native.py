"""Single-source scene-build tests (card bdfcb8cb).

scene_build.c — voxel mesher + BVH builder — compiled to both engines via
domain/slolib/_internal/native.py, the same compile-once loader as
trace_core.c. The claim under test is EXACT parity: the C reproduces the
Python's single-rounding float arithmetic and its (now stable) ordering
decisions, so native mesh/BVH output is byte-identical to the oracle's —
not merely "renders the same". When no engine compiled, or
SLO_CYCLES_NATIVE=0, the Python path runs unchanged.

Run: PYTHONPATH=packages/downcraft:packages/core-py scripts/python \
    -m pytest packages/core-py/tests/test_scene_build_native.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from domain.shell._internal import _native_build as nbuild
from domain.shell._internal.cycles import BVH, Mesh
from domain.shell._internal.simulation import WorldGrid
from domain.shell._internal.world_render import WorldToSceneMapper

# The same cube tables world_render._build_voxel_mesh emits. Cube soup
# test data gives centroid ties on every axis — the ordering edge case
# where an unstable sort and a stable sort would disagree.
_CUBE_VERTS = np.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ],
    dtype=np.float32,
)
_CUBE_FACES = np.array(
    [
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 4, 5],
        [0, 5, 1],
        [2, 6, 7],
        [2, 7, 3],
        [0, 3, 7],
        [0, 7, 4],
        [1, 5, 6],
        [1, 6, 2],
    ],
    dtype=np.int32,
)

_BVH_TABLES = (
    "_node_aabb_min",
    "_node_aabb_max",
    "_node_left",
    "_node_right",
    "_node_prim_count",
    "_node_prims",
    "_tri_v0",
    "_tri_e1",
    "_tri_e2",
)


def _cuda_usable() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


needs_cuda = pytest.mark.skipif(not _cuda_usable(), reason="no CUDA device")


def _native_build_available() -> bool:
    try:
        return nbuild._load("mesh") is not None and nbuild._load("bvh") is not None
    except Exception:
        return False


needs_native_build = pytest.mark.skipif(
    not _native_build_available(), reason="no native scene build engine (no gcc?)"
)


def _grid(seed: int, size=(9, 5, 7), density: float = 0.5) -> WorldGrid:
    rng = np.random.default_rng(seed)
    grid = WorldGrid(size=size)
    solid = rng.random(grid.total) < density
    grid.material[solid] = rng.integers(1, 5, size=int(solid.sum()))
    return grid


def _cube_soup(seed: int, n_cubes: int) -> Mesh:
    rng = np.random.default_rng(seed)
    centers = rng.integers(0, 8, size=(n_cubes, 3)).astype(np.float32)
    verts = np.concatenate([_CUBE_VERTS + (c + 0.5).astype(np.float32) for c in centers])
    faces = np.concatenate([_CUBE_FACES + 8 * i for i in range(n_cubes)])
    return Mesh(vertices=verts, faces=faces)


class TestMeshParity:
    @needs_native_build
    @pytest.mark.parametrize(
        "seed,size,density", [(0, (9, 5, 7), 0.5), (1, (13, 7, 11), 0.35), (2, (6, 3, 4), 0.9)]
    )
    def test_mesh_byte_identical(self, monkeypatch, seed, size, density):
        grid = _grid(seed, size=size, density=density)
        mapper = WorldToSceneMapper()

        monkeypatch.setenv("SLO_CYCLES_NATIVE", "0")
        py = mapper.grid_to_scene(grid).meshes[0]
        monkeypatch.setenv("SLO_CYCLES_NATIVE", "1")
        nat = mapper.grid_to_scene(grid).meshes[0]

        assert np.array_equal(py.vertices, nat.vertices)
        assert np.array_equal(py.faces, nat.faces)
        assert np.array_equal(py.material_idx, nat.material_idx)
        # Derived quantities flow from identical inputs — pin them anyway
        # (dtype/copy drift would show here first).
        assert np.array_equal(py.normals, nat.normals)
        assert np.array_equal(py._aabb_min, nat._aabb_min)
        assert np.array_equal(py._aabb_max, nat._aabb_max)


class TestBVHParity:
    @needs_native_build
    @pytest.mark.parametrize("seed", [0, 1])
    @pytest.mark.parametrize("max_prims", [1, 4, 8])
    def test_bvh_tables_byte_identical(self, monkeypatch, seed, max_prims):
        mesh = _cube_soup(seed, 12)

        monkeypatch.setenv("SLO_CYCLES_NATIVE", "0")
        py_bvh = BVH(mesh, max_prims)
        monkeypatch.setenv("SLO_CYCLES_NATIVE", "1")
        nat_bvh = BVH(mesh, max_prims)

        assert py_bvh._root == nat_bvh._root
        assert np.array_equal(
            np.asarray(py_bvh._prim_indices, dtype=np.int32),
            np.asarray(nat_bvh._prim_indices, dtype=np.int32),
        )
        for attr in _BVH_TABLES:
            assert np.array_equal(getattr(py_bvh, attr), getattr(nat_bvh, attr)), attr

    @needs_native_build
    def test_hits_identical(self, monkeypatch):
        """Behavioral pin: same rays, same (t, face) through both walks."""
        mesh = _cube_soup(3, 9)
        monkeypatch.setenv("SLO_CYCLES_NATIVE", "0")
        py_bvh = BVH(mesh, 4)
        monkeypatch.setenv("SLO_CYCLES_NATIVE", "1")
        nat_bvh = BVH(mesh, 4)

        rng = np.random.default_rng(4)
        orig = rng.uniform(-2, 10, size=(64, 3)).astype(np.float32)
        dire = rng.normal(size=(64, 3)).astype(np.float32)
        dire /= np.linalg.norm(dire, axis=1, keepdims=True).astype(np.float32)

        pt, pf = py_bvh.intersect_batch(orig, dire)
        nt, nf = nat_bvh.intersect_batch(orig, dire)
        assert np.array_equal(pt, nt)
        assert np.array_equal(pf, nf)
        for i in range(len(orig)):
            assert py_bvh.intersect(orig[i], dire[i]) == nat_bvh.intersect(orig[i], dire[i])


class TestFallback:
    def test_env_off_falls_back_to_python(self, monkeypatch):
        monkeypatch.setenv("SLO_CYCLES_NATIVE", "0")
        grid = _grid(4)
        scene = WorldToSceneMapper().grid_to_scene(grid)
        n_solid = int(np.count_nonzero(grid.material != 0))
        assert len(scene.meshes) == 1
        assert len(scene.meshes[0].faces) == n_solid * 12
        assert len(scene._bvh_list) >= 1
        assert len(scene._bvh_list[0].nodes) > 0

    def test_no_compiler_falls_back(self, monkeypatch):
        from domain.slolib._internal import native as native_mod

        monkeypatch.setattr(native_mod, "_find_compiler", lambda: None)
        monkeypatch.setattr(nbuild, "_MESH", None)
        monkeypatch.setattr(nbuild, "_BVH", None)
        assert nbuild._load("mesh") is None
        assert nbuild._load("bvh") is None
        assert (
            nbuild.build_mesh_native(
                np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int32), 1, 1, 1.0
            )
            is None
        )
        assert nbuild.build_bvh_native(None, None, 4) is None

    @needs_native_build
    def test_scene_end_to_end_parity(self, monkeypatch):
        """Whole pipeline: grid_to_scene native-on vs native-off — mesh,
        BVH tables, and material registration all identical."""
        grid = _grid(5, size=(11, 6, 9), density=0.55)

        monkeypatch.setenv("SLO_CYCLES_NATIVE", "0")
        py_scene = WorldToSceneMapper().grid_to_scene(grid)
        monkeypatch.setenv("SLO_CYCLES_NATIVE", "1")
        nat_scene = WorldToSceneMapper().grid_to_scene(grid)

        assert len(py_scene.materials) == len(nat_scene.materials)
        for pm, nm in zip(py_scene.materials, nat_scene.materials):
            assert pm.name == nm.name
            assert np.array_equal(pm.base_color, nm.base_color)
            assert pm.emission_strength == nm.emission_strength

        pmesh, nmesh = py_scene.meshes[0], nat_scene.meshes[0]
        assert np.array_equal(pmesh.vertices, nmesh.vertices)
        assert np.array_equal(pmesh.faces, nmesh.faces)
        assert np.array_equal(pmesh.material_idx, nmesh.material_idx)

        pbvh, nbvh = py_scene._bvh_list[0], nat_scene._bvh_list[0]
        assert pbvh._root == nbvh._root
        for attr in _BVH_TABLES:
            assert np.array_equal(getattr(pbvh, attr), getattr(nbvh, attr)), attr


def _python_mesh_oracle(flat: np.ndarray, grid, vs: float, matmap) -> Mesh:
    """The python mesher via the public off-switch (no monkeypatch)."""
    import os

    prev = os.environ.get("SLO_CYCLES_NATIVE")
    os.environ["SLO_CYCLES_NATIVE"] = "0"
    try:
        return WorldToSceneMapper()._build_voxel_mesh(flat.reshape(-1, 1), grid, vs, matmap)
    finally:
        if prev is None:
            os.environ.pop("SLO_CYCLES_NATIVE", None)
        else:
            os.environ["SLO_CYCLES_NATIVE"] = prev


class TestGPUEngine:
    @needs_cuda
    def test_gpu_mesh_entry_matches_python(self):
        """NVRTC compiles the same source; the GPU mesher must agree
        bit-for-bit with the python oracle (single-source honesty)."""
        import cupy as cp

        eng = nbuild._load("mesh")
        if eng is None:
            pytest.skip("no native CPU engine")
        gpu = eng.gpu
        assert gpu is not None

        grid = _grid(6, size=(7, 4, 5), density=0.6)
        flat = np.ascontiguousarray(np.flatnonzero(grid.material != 0), dtype=np.int64)
        mat_types = grid.material[flat]
        matmap = {int(m): int(m) for m in np.unique(mat_types)}
        mat = np.ascontiguousarray(np.vectorize(matmap.get)(mat_types), dtype=np.int32)
        vs = 1.5
        py = _python_mesh_oracle(flat, grid, vs, matmap)
        n = int(flat.size)

        out_v = cp.empty((n * 8, 3), dtype=cp.float32)
        out_f = cp.empty((n * 12, 3), dtype=cp.int32)
        out_m = cp.empty(n * 12, dtype=cp.int32)
        block = 256
        gpu(
            ((n + block - 1) // block, 1, 1),
            (block, 1, 1),
            (
                cp.asarray(flat),
                cp.asarray(mat),
                np.int64(n),
                np.int32(grid.size[1]),
                np.int32(grid.size[2]),
                np.float64(vs),
                out_v,
                out_f,
                out_m,
            ),
        )
        assert np.array_equal(py.vertices, cp.asnumpy(out_v))
        assert np.array_equal(py.faces, cp.asnumpy(out_f))
        assert np.array_equal(py.material_idx, cp.asnumpy(out_m))

    @needs_cuda
    def test_gpu_bvh_entry_matches_python(self):
        import cupy as cp

        eng = nbuild._load("bvh")
        if eng is None:
            pytest.skip("no native CPU engine")
        gpu = eng.gpu
        assert gpu is not None

        mesh = _cube_soup(7, 10)
        nf = len(mesh.faces)
        cap = 2 * nf + 1
        v = cp.asarray(np.ascontiguousarray(mesh.vertices, dtype=np.float32))
        f = cp.asarray(np.ascontiguousarray(mesh.faces, dtype=np.int32))
        o_min = cp.empty((cap, 3), dtype=cp.float32)
        o_max = cp.empty((cap, 3), dtype=cp.float32)
        o_l = cp.empty(cap, dtype=cp.int32)
        o_r = cp.empty(cap, dtype=cp.int32)
        o_ps = cp.empty(cap, dtype=cp.int32)
        o_pc = cp.empty(cap, dtype=cp.int32)
        perm = cp.empty(nf, dtype=cp.int32)
        tmp = cp.empty(nf, dtype=cp.int32)
        keys = cp.empty(nf, dtype=cp.float32)
        prims = cp.empty(nf, dtype=cp.int32)
        n_nodes = cp.zeros(1, dtype=cp.int32)
        n_prims = cp.zeros(1, dtype=cp.int64)

        gpu(
            (1, 1, 1),
            (1, 1, 1),
            (
                v,
                f,
                np.int64(nf),
                np.int32(4),
                o_min,
                o_max,
                o_l,
                o_r,
                o_ps,
                o_pc,
                perm,
                tmp,
                keys,
                prims,
                n_nodes,
                n_prims,
            ),
        )

        import os

        prev = os.environ.get("SLO_CYCLES_NATIVE")
        os.environ["SLO_CYCLES_NATIVE"] = "0"
        try:
            py = BVH(mesh, 4)
        finally:
            if prev is None:
                os.environ.pop("SLO_CYCLES_NATIVE", None)
            else:
                os.environ["SLO_CYCLES_NATIVE"] = prev

        nn = int(cp.asnumpy(n_nodes)[0])
        npr = int(cp.asnumpy(n_prims)[0])
        assert nn == len(py.nodes)
        assert npr == len(py._prim_indices)
        assert np.array_equal(cp.asnumpy(o_min)[:nn], py._node_aabb_min)
        assert np.array_equal(cp.asnumpy(o_max)[:nn], py._node_aabb_max)
        assert np.array_equal(cp.asnumpy(o_l)[:nn], py._node_left)
        assert np.array_equal(cp.asnumpy(o_r)[:nn], py._node_right)
        assert np.array_equal(cp.asnumpy(o_pc)[:nn], py._node_prim_count)
        assert np.array_equal(
            cp.asnumpy(o_ps)[:nn],
            np.array([node.primitive_start for node in py.nodes], dtype=np.int32),
        )
        assert np.array_equal(cp.asnumpy(prims)[:npr], np.asarray(py._prim_indices, dtype=np.int32))
