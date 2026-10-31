"""Compile-once scene-build bridge: voxel mesher + BVH builder.

`scene_build.c` is ONE source compiled by ``domain/slolib/_internal/native.py``
to both engines — the same contract `_native.py` holds for trace_core.c.
This module is the frame-path seam: `world_render` asks for native mesh /
BVH output and treats ``None`` as "engine unavailable, use the Python
oracle".

Engine choice is deliberate: the scene is host-resident (`Scene`/`Mesh`
hold numpy arrays and the frame uploads them), so the frame path calls
the CPU engine — a GPU build would spend its win on a PCIe round-trip.
The source is NVRTC-clean and the `scene_*_gpu` entries are compiled (and
run) by the test suite, keeping the single-source claim honest without
paying NVRTC compile latency in production.

`SLO_CYCLES_NATIVE=0` forces the Python oracle path — read per call (not
at import) so tests/CLIs can toggle without a module reload, exactly like
`_native._disabled()`.
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger("slo.cycles.native.build")

# Compiled engines, cached per process. False = "attempted, unavailable"
# so a missing compiler is not re-probed every frame; never cached when
# disabled, so re-enabling works immediately.
_MESH = None
_BVH = None


def _disabled() -> bool:
    """SLO_CYCLES_NATIVE=0 forces the python mesher/BVH path."""
    return os.environ.get("SLO_CYCLES_NATIVE", "1") == "0"


def _load(which: str):
    """Compile scene_build.c once (per entry pair) or return None."""
    global _MESH, _BVH
    if _disabled():
        return None  # off: never cached, so re-enabling works immediately
    slot = _MESH if which == "mesh" else _BVH
    if slot is not None:
        return slot or None
    try:
        from domain.slolib._internal.native import NativeEngines

        src = (Path(__file__).parent / "scene_build.c").read_text()
        if which == "mesh":
            eng = NativeEngines(src, "entry_mesh_cpu", "scene_mesh_gpu")
        else:
            eng = NativeEngines(src, "entry_bvh_cpu", "scene_bvh_gpu")
        # CPU only: the frame path is host-resident (see module docstring);
        # `.gpu` stays lazy for the tests that verify NVRTC parity.
        slot = eng if eng.cpu is not None else False
    except Exception as exc:  # noqa: BLE001 — any loader failure = python path
        logger.info("native scene build unavailable (%s); python oracle path", exc)
        slot = False
    if which == "mesh":
        _MESH = slot
    else:
        _BVH = slot
    return slot or None


def build_mesh_native(flat_idx, voxel_mat, ny, nz, vs):
    """Voxel mesh extraction through scene_build.c, or None (python path).

    Returns ``(verts, faces, fmat)`` — byte-identical to the Python loop
    in `WorldToSceneMapper._build_voxel_mesh` (same single-rounding
    arithmetic; verified by the parity gate).
    """
    eng = _load("mesh")
    if eng is None:
        return None
    n = int(flat_idx.size)
    if n == 0:
        return None
    flat = np.ascontiguousarray(flat_idx, dtype=np.int64)
    mat = np.ascontiguousarray(voxel_mat, dtype=np.int32)
    verts = np.empty((n * 8, 3), dtype=np.float32)
    faces = np.empty((n * 12, 3), dtype=np.int32)
    fmat = np.empty(n * 12, dtype=np.int32)

    fn = eng.cpu.entry_mesh_cpu
    fn.argtypes = [
        ctypes.c_void_p,  # flat_idx
        ctypes.c_void_p,  # mat_idx
        ctypes.c_long,  # n
        ctypes.c_int,  # ny
        ctypes.c_int,  # nz
        ctypes.c_double,  # vs
        ctypes.c_void_p,  # verts
        ctypes.c_void_p,  # faces
        ctypes.c_void_p,  # fmat
    ]
    fn.restype = None
    fn(
        flat.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
        mat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_long(n),
        ctypes.c_int(int(ny)),
        ctypes.c_int(int(nz)),
        ctypes.c_double(float(vs)),
        verts.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        faces.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        fmat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )
    return verts, faces, fmat


def build_bvh_native(vertices, faces, max_prims):
    """BVH construction through scene_build.c, or None (python path).

    Returns ``(nmin, nmax, left, right, pstart, pcnt, prims, root)`` with
    nodes in the Python recursion's pre-order and ``prims`` in
    ``_prim_indices`` order — byte-identical to `BVH._build_recursive`
    under the stable centroid ordering both sides share.
    """
    eng = _load("bvh")
    if eng is None:
        return None
    nf = int(len(faces))
    if nf == 0:
        return None  # python path raises on an empty mesh too; same behavior
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.int32)
    cap = 2 * nf + 1  # node cap 2n-1, one spare
    out_min = np.empty((cap, 3), dtype=np.float32)
    out_max = np.empty((cap, 3), dtype=np.float32)
    out_left = np.empty(cap, dtype=np.int32)
    out_right = np.empty(cap, dtype=np.int32)
    out_pstart = np.empty(cap, dtype=np.int32)
    out_pcnt = np.empty(cap, dtype=np.int32)
    perm = np.empty(nf, dtype=np.int32)
    tmp = np.empty(nf, dtype=np.int32)
    keys = np.empty(nf, dtype=np.float32)
    prim_out = np.empty(nf, dtype=np.int32)
    n_nodes = np.zeros(1, dtype=np.int32)
    n_prims = np.zeros(1, dtype=np.int64)

    fn = eng.cpu.entry_bvh_cpu
    fn.argtypes = [
        ctypes.c_void_p,  # verts
        ctypes.c_void_p,  # faces
        ctypes.c_long,  # n_faces
        ctypes.c_int,  # max_prims
        ctypes.c_void_p,  # out_min
        ctypes.c_void_p,  # out_max
        ctypes.c_void_p,  # out_left
        ctypes.c_void_p,  # out_right
        ctypes.c_void_p,  # out_pstart
        ctypes.c_void_p,  # out_pcnt
        ctypes.c_void_p,  # perm
        ctypes.c_void_p,  # tmp
        ctypes.c_void_p,  # keys
        ctypes.c_void_p,  # prim_out
        ctypes.c_void_p,  # out_n_nodes
        ctypes.c_void_p,  # out_n_prims
    ]
    fn.restype = None
    fn(
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        f.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_long(nf),
        ctypes.c_int(int(max_prims)),
        out_min.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_max.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_left.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_right.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_pstart.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_pcnt.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        perm.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        keys.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        prim_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        n_nodes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        n_prims.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
    )
    nn = int(n_nodes[0])
    npr = int(n_prims[0])
    return (
        out_min[:nn],
        out_max[:nn],
        out_left[:nn],
        out_right[:nn],
        out_pstart[:nn],
        out_pcnt[:nn],
        prim_out[:npr],
        0,  # root: the first appended node, like the recursion's return
    )
