"""Native trace bridge — packs the scene once, dispatches to one engine.

Card 4de18e24. This is the marshalling seam between the numpy ``Scene``
(BVH + Material python objects) and the single-source C core in
``trace_core.c``. It does three things and nothing else:

1. **Pack** — flatten every BVH's local tables into global arrays the C
   walk expects: node tables concatenated (``moff_node`` records where
   each mesh's nodes start), prim face-IDs re-based local->global
   (``fbase + fi``), triangle edges and per-face vertex-normals
   concatenated (``moff_face``), materials SoA, background. One numpy
   pack, reused every frame unless ``build_bvh`` rebuilds (keyed on the
   ``_bvh_list`` identity).

2. **Dispatch** — one ``trace(orig, dire, seed, ...)`` call. Picks the
   engine from the resolved device and what compiled: device=cpu → the
   gcc+OpenMP ctypes entry; device=cuda → the NVRTC RawKernel; neither
   compiled → the caller falls back to the numpy oracle (this returns
   None to signal that). Inputs host in → radiance host out, exactly the
   crossing contract ``render()`` already relies on.

3. **Hold** — the compiled engines (cached per source-hash by the
   slolib loader) and the packed tables (cached per scene build).

The counters: one thread per path on BOTH engines means the entire walk
+ shade + bounce runs inside a single thread — zero host dispatches per
bounce, zero alive-ray compaction. That is what kills the ~60us/call
dispatch floor that crippled the operator-by-operator CuPy walk.
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger("slo.cycles.native")

# One compiled source, cached across every renderer for the process.
# Only successful compiles are cached — "off" is evaluated per call.
_ENGINES = None


def _disabled() -> bool:
    """SLO_CYCLES_NATIVE=0 forces the numpy oracle path.

    Read dynamically (not at import) so tests/CLIs can toggle it at
    runtime without a module reload — an import-time constant would bake
    the value in and leak into every later suite in the same process.
    """
    return os.environ.get("SLO_CYCLES_NATIVE", "1") == "0"


def _load_engines():
    """Compile trace_core.c once (CPU + GPU) or return None."""
    global _ENGINES
    if _disabled():
        return None  # off: never cached, so re-enabling works immediately
    if _ENGINES is not None:
        return _ENGINES
    try:
        from pathlib import Path

        from domain.slolib._internal.native import NativeEngines

        src = (Path(__file__).parent / "trace_core.c").read_text()
        eng = NativeEngines(src, "entry_trace_cpu", "trace_paths_gpu")
        _ENGINES = eng if eng.any else False
    except Exception as exc:  # noqa: BLE001 — any loader failure = numpy path
        logger.info("native trace unavailable (%s); numpy oracle path", exc)
        _ENGINES = False
        return None
    if _ENGINES is not False:
        logger.info(
            "native trace core: cpu=%s gpu=%s",
            _ENGINES.cpu is not None,
            _ENGINES.gpu is not None,
        )
    return _ENGINES


class NativeTrace:
    """Packed scene + compiled engines. One per renderer, built lazily."""

    def __init__(self):
        self._cache = None  # (bvh_list_id, tables_dict)

    # ── pack ────────────────────────────────────────────────────────────
    def _tables(self, scene):
        """Flatten every BVH into global arrays. Cached per build."""
        bvh_list = scene._bvh_list
        key = id(bvh_list)
        if not bvh_list:
            return None  # empty scene: no tables to pack
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]

        na_min, na_max, nleft, nright, npcnt, nprims = [], [], [], [], [], []
        tv0, te1, te2, fnorm, fmat = [], [], [], [], []
        moff_node, moff_face = [], []
        node_cursor = face_cursor = 0
        K = 0

        for bvh in bvh_list:
            m = bvh.mesh
            n_nodes = len(bvh.nodes)
            K = max(K, int(bvh._node_prims.shape[1]))
            moff_node.append(node_cursor)
            moff_face.append(face_cursor)

            na_min.append(bvh._node_aabb_min)
            na_max.append(bvh._node_aabb_max)
            nleft.append(bvh._node_left)
            nright.append(bvh._node_right)
            npcnt.append(bvh._node_prim_count)
            nprims.append(bvh._node_prims)

            tv0.append(bvh._tri_v0)
            te1.append(bvh._tri_e1)
            te2.append(bvh._tri_e2)
            # Per-face vertex normals -> (F, 9): [n0 | n1 | n2].
            vn = m.normals[m.faces]  # (F, 3, 3)
            fnorm.append(vn.reshape(-1, 9).astype(np.float32))
            fmat.append(m.material_idx.astype(np.int32))

            node_cursor += n_nodes
            face_cursor += len(m.faces)

        # Prims hold LOCAL face IDs; the C walk re-bases with fbase+fi, so
        # we only pad the per-mesh block to a uniform K (the -1 pad ends a
        # row early). Uniform stride = one `K` for the whole flat table.
        prims = np.full((node_cursor, K), -1, dtype=np.int32)
        node_i = 0
        for block in nprims:
            nb = block.shape[0]
            prims[node_i : node_i + nb, : block.shape[1]] = block
            node_i += nb

        mats = scene.materials
        m_base = np.array([mm.base_color for mm in mats], dtype=np.float32)
        m_emis = np.array([mm.emission() for mm in mats], dtype=np.float32)
        m_prop = np.array(
            [[mm.transmission, mm.metallic, mm.ior, mm.roughness] for mm in mats],
            dtype=np.float32,
        )

        tables = {
            "nmin": np.ascontiguousarray(np.concatenate(na_min)),
            "nmax": np.ascontiguousarray(np.concatenate(na_max)),
            "left": np.ascontiguousarray(np.concatenate(nleft)),
            "right": np.ascontiguousarray(np.concatenate(nright)),
            "pcnt": np.ascontiguousarray(np.concatenate(npcnt)),
            "prims": prims,
            "K": np.int32(K),
            "tv0": np.ascontiguousarray(np.concatenate(tv0)),
            "te1": np.ascontiguousarray(np.concatenate(te1)),
            "te2": np.ascontiguousarray(np.concatenate(te2)),
            "fnorm": np.ascontiguousarray(np.concatenate(fnorm)),
            "fmat": np.ascontiguousarray(np.concatenate(fmat)),
            "m_base": m_base,
            "m_emis": m_emis,
            "m_prop": m_prop,
            "moff_node": np.ascontiguousarray(np.array(moff_node, dtype=np.int32)),
            "moff_face": np.ascontiguousarray(np.array(moff_face, dtype=np.int32)),
            "n_meshes": np.int32(len(bvh_list)),
            "bg": np.ascontiguousarray(scene.background.astype(np.float32)),
        }
        self._cache = (key, tables)
        return tables

    # ── dispatch ────────────────────────────────────────────────────────
    def trace(self, xp, scene, orig, dire, seed, max_depth, rr_start):
        """Trace a flattened ray batch. Returns (n, 3) f32 radiance or None.

        ``orig``/``dire`` are (n, 3). Host in → host out. None means no
        native engine for this device — the caller uses the numpy oracle.
        """
        eng = _load_engines()
        if eng is None:
            return None

        use_gpu = xp is not np
        if use_gpu and eng.gpu is None:
            return None
        if not use_gpu and eng.cpu is None:
            return None

        t = self._tables(scene)
        n = int(orig.shape[0])
        if n == 0:
            return np.zeros((0, 3), dtype=np.float32)
        # Empty scene (no BVHs): nothing to trace — let the numpy oracle
        # handle it (it returns the flat background for a mesh-less scene).
        if t is None:
            return None

        seed_u = np.uint32(seed & 0xFFFFFFFF)
        md = np.float32(max_depth)
        rr = np.float32(rr_start)

        if use_gpu:
            import cupy as cp

            o = cp.asarray(orig, dtype=cp.float32)
            d = cp.asarray(dire, dtype=cp.float32)
            dev = {k: cp.asarray(v) for k, v in t.items() if isinstance(v, np.ndarray)}
            dev = {k: cp.ascontiguousarray(v) for k, v in dev.items()}
            out = cp.empty((n, 3), dtype=cp.float32)
            args = self._arglist(dev, t, o, d, n, seed_u, md, rr, out, gpu=True)
            block = 256
            eng.gpu(((n + block - 1) // block, 1, 1), (block, 1, 1), args)
            return cp.asnumpy(out)

        o = np.ascontiguousarray(orig, dtype=np.float32)
        d = np.ascontiguousarray(dire, dtype=np.float32)
        out = np.empty((n, 3), dtype=np.float32)
        args = self._arglist(t, t, o, d, n, seed_u, md, rr, out, gpu=False)
        fn = eng.cpu.entry_trace_cpu
        fn.argtypes = self._argtypes()
        fn.restype = None
        fn(*args)
        return out

    def _arglist(self, pt, t, o, d, n, seed_u, md, rr, out, gpu):
        """Order MUST match SLO_TRACE_PARAMS in trace_core.c.

        Same logical args on both engines, different ABI: ctypes takes raw
        addresses (``c_void_p``); cupy RawKernel takes device arrays and
        device scalar pointers (``cp.float32(...)`` / ``cp.int32(...)``).
        """
        if not gpu:
            import ctypes

            def ptr(a):
                return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            def iptr(a):
                return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

            # Build in exact param order (see SLO_TRACE_PARAMS).
            args = []
            args += [ptr(pt["nmin"]), ptr(pt["nmax"])]
            args += [iptr(pt["left"]), iptr(pt["right"]), iptr(pt["pcnt"]), iptr(pt["prims"])]
            args += [ctypes.c_int(int(t["K"]))]
            args += [ptr(pt["tv0"]), ptr(pt["te1"]), ptr(pt["te2"])]
            args += [ptr(pt["fnorm"]), iptr(pt["fmat"])]
            args += [ptr(pt["m_base"]), ptr(pt["m_emis"]), ptr(pt["m_prop"])]
            args += [iptr(pt["moff_node"]), iptr(pt["moff_face"])]
            args += [ctypes.c_int(int(t["n_meshes"]))]
            args += [ptr(pt["bg"])]
            args += [ptr(o), ptr(d)]
            args += [ctypes.c_long(n)]
            args += [ctypes.c_uint(int(seed_u))]
            args += [ctypes.c_float(float(md)), ctypes.c_float(float(rr))]
            args += [ptr(out)]
            return args

        import cupy as cp

        def dev(a):
            return cp.asarray(a)

        scal = cp.float32
        iscal = cp.int32
        args = []
        args += [dev(pt["nmin"]), dev(pt["nmax"])]
        args += [dev(pt["left"]), dev(pt["right"]), dev(pt["pcnt"]), dev(pt["prims"])]
        args += [iscal(t["K"])]
        args += [dev(pt["tv0"]), dev(pt["te1"]), dev(pt["te2"])]
        args += [dev(pt["fnorm"]), dev(pt["fmat"])]
        args += [dev(pt["m_base"]), dev(pt["m_emis"]), dev(pt["m_prop"])]
        args += [dev(pt["moff_node"]), dev(pt["moff_face"])]
        args += [iscal(t["n_meshes"])]
        args += [dev(pt["bg"])]
        args += [o, d]
        args += [np.int64(n)]
        args += [np.uint32(seed_u)]
        args += [scal(md), scal(rr)]
        args += [out]
        return args

    def _argtypes(self):
        import ctypes

        f = ctypes.c_void_p
        i = ctypes.c_void_p
        return [
            f, f, i, i, i, i, ctypes.c_int, f, f, f, f, i, f, f, f, i, i,
            ctypes.c_int, f, f, f, ctypes.c_long, ctypes.c_uint, ctypes.c_float,
            ctypes.c_float, f,
        ]
