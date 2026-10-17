"""
Cycles — RAM-resident path tracer rendering engine.

A Cycles-style path tracer that runs on numpy by default and on the
GPU (CuPy, through the accelerator layer) when a CUDA device is
present. Output is always numpy; scene state tensors stay host
arrays. Selection: ``device=`` on the renderer or
``SLO_CYCLES_DEVICE=cpu|cuda|auto`` — auto asks the accelerator.

Supports:
  - Mesh geometry (vertices, faces, normals)
  - Materials: diffuse, glossy (GGX), glass, emission, subsurface
  - Area + point lights
  - BVH acceleration structure
  - Path tracing with Russian roulette termination
  - Output: rendered image as numpy tensor (H, W, 3)

The engine is designed to run as a VM device — each frame produces
state tensors (depth, normals, albedo, emission) that the NPU
can process for neural rendering, denoising, or scene understanding.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("slo.cycles")

EPSILON = 1e-6
MAX_DEPTH = 8
RR_START = 3


# ── Array module (numpy everywhere; cupy when the accelerator says CUDA) ────

_XP_CACHE: dict = {}


def _xp(a):
    """Array module (numpy or cupy) that owns ``a``.

    Input-driven by design: the hot helpers are called from both sides
    of the device boundary, so the namespace follows the data — host
    arrays keep the whole host path, device arrays keep the whole
    device path. A helper never decides the device on its own.
    """
    if type(a).__module__.split(".")[0] == "cupy":
        import cupy

        return cupy
    return np


def resolve_array_module(device: str | None = None):
    """numpy unless ``device`` resolves to a working CUDA accelerator.

    ``device``: ``"auto"`` (default — ask the accelerator layer),
    ``"cuda"`` (require GPU; falls back to CPU with a warning if the
    probe fails — ``is_available`` only proves the package imports),
    ``"cpu"`` (force numpy). An explicit argument beats
    ``SLO_CYCLES_DEVICE``; the result is cached per device string
    because a backend cannot change under a running process.
    """
    device = device or os.environ.get("SLO_CYCLES_DEVICE") or "auto"
    if device == "cpu":
        return np
    if device in _XP_CACHE:
        return _XP_CACHE[device]
    xp = np
    try:
        from domain.slolib._internal.gpu import get_accelerator

        cand = get_accelerator().array_module()
        if cand is not np:
            probe = cand.arange(8, dtype=cand.float32) * 2.0
            float(probe.sum())  # sync: a kernel must actually run
            xp = cand
    except Exception as exc:
        if device == "cuda":
            logger.warning("cycles: device=cuda probe failed (%s); staying on CPU", exc)
        xp = np
    if device == "cuda" and xp is np:
        logger.warning("cycles: device=cuda requested but no usable CUDA backend; staying on CPU")
    _XP_CACHE[device] = xp
    return xp


def device_name(device: str | None = None) -> str:
    """``"cuda"`` or ``"cpu"`` — which array module this selection resolves to."""
    return "cpu" if resolve_array_module(device) is np else "cuda"


# ── Fused CUDA Möller–Trumbore kernel (device path of BVH._mt_batch) ─────────
#
# Every Python-level cupy call pays a ~60us dispatch/launch floor on a
# consumer card (measured on the GTX 1050), so the operator-by-operator
# translation of the Möller–Trumbore test — ~75 calls per invocation,
# once per leaf iteration of the frontier walk — burned ~4.5ms of pure
# dispatch per iteration. This is the SAME math as ``BVH._mt_numpy``
# compiled into one launch: identical EPSILON rules, identical 1e30
# miss sentinel, identical float32 output cast. ``__T__`` is replaced
# with the promoted compute type (double wins when dtypes mix).

_MT_CUDA_SRC = r"""
extern "C" __global__
void mt_batch(const __T__* __restrict__ v0, const __T__* __restrict__ e1,
              const __T__* __restrict__ e2, const __T__* __restrict__ orig,
              const __T__* __restrict__ dire, float* __restrict__ t_out,
              const long n) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long j = 3 * i;
  const __T__ dx = dire[j], dy = dire[j + 1], dz = dire[j + 2];
  const __T__ e2x = e2[j], e2y = e2[j + 1], e2z = e2[j + 2];
  const __T__ hx = dy * e2z - dz * e2y;
  const __T__ hy = dz * e2x - dx * e2z;
  const __T__ hz = dx * e2y - dy * e2x;
  const __T__ e1x = e1[j], e1y = e1[j + 1], e1z = e1[j + 2];
  const __T__ a = e1x * hx + e1y * hy + e1z * hz;
  __T__ t = 1e30;
  if (fabs((double)a) > 1e-6) {
    const __T__ f = (__T__)1.0 / a;
    const __T__ sx = orig[j] - v0[j];
    const __T__ sy = orig[j + 1] - v0[j + 1];
    const __T__ sz = orig[j + 2] - v0[j + 2];
    const __T__ u = f * (sx * hx + sy * hy + sz * hz);
    const __T__ qx = sy * e1z - sz * e1y;
    const __T__ qy = sz * e1x - sx * e1z;
    const __T__ qz = sx * e1y - sy * e1x;
    const __T__ v = f * (dx * qx + dy * qy + dz * qz);
    const __T__ tt = f * (e2x * qx + e2y * qy + e2z * qz);
    if (u >= (__T__)0.0 && v >= (__T__)0.0 && (u + v) <= (__T__)1.0 && tt > (__T__)1e-6)
      t = tt;
  }
  t_out[i] = (float)t;
}
"""

_MT_KERNELS: dict = {}


def _mt_cuda_kernel(cp, dtype):
    """Compiled-once RawKernel for the promoted dtype; cached per dtype."""
    kern = _MT_KERNELS.get(dtype.str)
    if kern is None:
        ctype = "double" if dtype == np.float64 else "float"
        kern = cp.RawKernel(_MT_CUDA_SRC.replace("__T__", ctype), "mt_batch")
        _MT_KERNELS[dtype.str] = kern
    return kern


# ── Vector math (pure numpy, no dependencies) ────────────────────────────────


def _normalize(v: np.ndarray) -> np.ndarray:
    xp = _xp(v)
    n = xp.linalg.norm(v, axis=-1, keepdims=True)
    return xp.where(n > EPSILON, v / n, 0.0)


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _xp(a).sum(a * b, axis=-1)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _xp(a).cross(a, b)


def _reflect(I: np.ndarray, N: np.ndarray) -> np.ndarray:
    return I - 2.0 * _dot(I, N)[..., np.newaxis] * N


def _refract(I: np.ndarray, N: np.ndarray, eta: float | np.ndarray) -> tuple[np.ndarray, bool | np.ndarray]:
    xp = _xp(I)
    eta = xp.asarray(eta, dtype=np.float64)
    cos_i = -_dot(I, N)
    if eta.ndim == 1:
        # Per-ray eta: columnarize BOTH eta and cos_i so every product
        # stays (k, 1) x (k, 3) -> (k, 3); mixing (k,1) with (k,) would
        # broadcast to (k, k).
        eta = eta[:, np.newaxis]
        cos_i = cos_i[:, np.newaxis]
    sin2_t = eta**2 * (1.0 - cos_i**2)
    mask = sin2_t < 1.0
    cos_t = xp.sqrt(xp.clip(1.0 - sin2_t, 0, 1))
    result = eta * I + (eta * cos_i - cos_t) * N
    return result, mask


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class Material:
    """PBR material with multiple lobes."""

    name: str = "default"
    base_color: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 0.8]))
    metallic: float = 0.0
    roughness: float = 0.5
    ior: float = 1.45
    emission_strength: float = 0.0
    emission_color: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    transmission: float = 0.0
    subsurface: float = 0.0
    subsurface_radius: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.2, 0.1]))

    def eval(self, wi: np.ndarray, wo: np.ndarray, N: np.ndarray) -> np.ndarray:
        """Evaluate BSDF. Returns (3,) color contribution."""
        diffuse = self.base_color / math.pi

        H = _normalize(wi + wo)
        NdotH = np.maximum(_dot(N, H), 0.0)
        NdotV = np.maximum(_dot(N, wo), 0.0)
        NdotL = np.maximum(_dot(N, wi), 0.0)

        if NdotL < EPSILON or NdotV < EPSILON:
            return np.zeros(3)

        alpha = self.roughness**2
        alpha2 = alpha**2

        D = _ggx_distribution(NdotH, alpha2)
        G = _ggx_geometry(NdotV, NdotL, alpha2)
        F0 = (1.0 - self.metallic) * np.array([0.04]) + self.metallic * self.base_color
        F = _fresnel_schlick(_dot(H, wo), F0)

        specular = (D * G * F) / (4.0 * NdotV * NdotL + EPSILON)

        kD = (1.0 - F) * (1.0 - self.metallic)
        color = kD * diffuse + specular * self.metallic + (1 - self.metallic) * specular
        return color * NdotL

    def emission(self) -> np.ndarray:
        return self.emission_color * self.emission_strength


@dataclass
class Mesh:
    """Triangle mesh geometry."""

    vertices: np.ndarray  # (V, 3)
    faces: np.ndarray  # (F, 3) indices
    normals: np.ndarray | None = None  # (V, 3) or None = compute
    material_idx: np.ndarray | None = None  # (F,) per-face material index
    _aabb_min: np.ndarray | None = None
    _aabb_max: np.ndarray | None = None

    def __post_init__(self):
        if self.normals is None:
            self._compute_normals()
        if self.material_idx is None:
            self.material_idx = np.zeros(len(self.faces), dtype=np.int32)
        self._aabb_min = self.vertices.min(axis=0)
        self._aabb_max = self.vertices.max(axis=0)

    def _compute_normals(self):
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        face_normals = _normalize(_cross(v1 - v0, v2 - v0))
        vertex_normals = np.zeros_like(self.vertices)
        for i in range(3):
            np.add.at(vertex_normals, self.faces[:, i], face_normals)
        self.normals = _normalize(vertex_normals)


@dataclass
class Light:
    """Area or point light."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    color: np.ndarray = field(default_factory=lambda: np.ones(3))
    strength: float = 1.0
    size: float = 0.5  # area light radius


@dataclass
class Camera:
    """Pinhole camera."""

    origin: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 3.0]))
    look_at: np.ndarray = field(default_factory=lambda: np.zeros(3))
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    fov: float = 50.0  # degrees

    def generate_rays(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate ray origins and directions for every pixel.

        Returns:
            origins: (H, W, 3)
            directions: (H, W, 3)
        """
        aspect = width / height
        half_h = math.tan(math.radians(self.fov) / 2)
        half_w = half_h * aspect

        forward = _normalize(self.look_at - self.origin)
        right = _normalize(_cross(forward, self.up))
        up = _cross(right, forward)

        u = np.linspace(-half_w, half_w, width)
        v = np.linspace(half_h, -half_h, height)
        uu, vv = np.meshgrid(u, v)

        dirs = (
            forward[np.newaxis, np.newaxis, :]
            + uu[..., np.newaxis] * right[np.newaxis, np.newaxis, :]
            + vv[..., np.newaxis] * up[np.newaxis, np.newaxis, :]
        )
        dirs = _normalize(dirs)
        # Force float: an int-typed Camera.origin (e.g. np.array([0, 2, 4]))
        # would broadcast int64 origins here, and every bounce would then
        # C-truncate its updated origin back to whole numbers.
        origins = np.broadcast_to(np.asarray(self.origin, dtype=np.float64), dirs.shape)
        return origins, dirs


# ── GGX BSDF helpers ─────────────────────────────────────────────────────────


def _ggx_distribution(NdotH: np.ndarray, alpha2: float) -> np.ndarray:
    denom = NdotH**2 * (alpha2 - 1.0) + 1.0
    return alpha2 / (math.pi * denom**2 + EPSILON)


def _ggx_geometry(NdotV: float, NdotL: float, alpha2: float) -> float:
    def _smith(n: float) -> float:
        return 2.0 * n / (n + math.sqrt(alpha2 + (1 - alpha2) * n**2))

    return _smith(NdotV) * _smith(NdotL)


def _fresnel_schlick(cos: float, F0: np.ndarray) -> np.ndarray:
    return F0 + (1.0 - F0) * (1.0 - cos) ** 5


# ── Vectorized BSDF batch helpers (mirror the scalar methods over (n,) arrays) ─


def _ggx_geometry_batch(NdotV: np.ndarray, NdotL: np.ndarray, alpha2: float) -> np.ndarray:
    xp = _xp(NdotV)

    def _smith(n: np.ndarray) -> np.ndarray:
        return 2.0 * n / (n + xp.sqrt(alpha2 + (1 - alpha2) * n**2))

    return _smith(NdotV) * _smith(NdotL)


def _eval_bsdf_batch(
    wi: np.ndarray, wo: np.ndarray, N: np.ndarray, mat: Material
) -> np.ndarray:
    """Material.eval over a batch — identical math, array ops, masked zeros."""
    xp = _xp(wi)
    diffuse = xp.asarray(mat.base_color / math.pi)

    H = _normalize(wi + wo)
    NdotH = xp.maximum(_dot(N, H), 0.0)
    NdotV = xp.maximum(_dot(N, wo), 0.0)
    NdotL = xp.maximum(_dot(N, wi), 0.0)

    alpha = mat.roughness**2
    alpha2 = alpha**2

    D = _ggx_distribution(NdotH, alpha2)
    G = _ggx_geometry_batch(NdotV, NdotL, alpha2)
    # F0 stays a (3,) color — built on the host, then owned by `xp`
    # so the fresnel mix below never straddles the device boundary.
    F0 = xp.asarray((1.0 - mat.metallic) * np.array([0.04]) + mat.metallic * mat.base_color)
    F = _fresnel_schlick(_dot(H, wo)[..., np.newaxis], F0)

    specular = ((D * G)[..., np.newaxis] * F) / (4.0 * NdotV * NdotL + EPSILON)[..., np.newaxis]

    kD = (1.0 - F) * (1.0 - mat.metallic)
    color = kD * diffuse + specular * mat.metallic + (1 - mat.metallic) * specular
    color = color * NdotL[..., np.newaxis]
    dead = (NdotL < EPSILON) | (NdotV < EPSILON)
    return xp.where(dead[..., np.newaxis], 0.0, color)


def _bsdf_pdf_batch(
    wo: np.ndarray,
    wi: np.ndarray,
    N: np.ndarray,
    mat: Material,
    lobe: np.ndarray,
) -> np.ndarray:
    """PDF per ray for the lobe actually sampled (0=trans, 1=gloss, 2=diffuse).

    The scalar version re-draws RNG to guess the lobe; the batch version
    uses the branch that was genuinely sampled — sampling and pdf stay
    paired, which is the correct MIS form.
    """
    xp = _xp(wo)
    pdf = xp.maximum(_dot(N, wi), 0.0) / math.pi  # diffuse default
    gloss = lobe == 1
    if gloss.any():
        H = _normalize(wo[gloss] + wi[gloss])
        alpha2 = mat.roughness**4
        NdotH = xp.maximum(_dot(N[gloss], H), 0.0)
        D = _ggx_distribution(NdotH, alpha2)
        pdf[gloss] = D * NdotH / (4 * xp.maximum(_dot(H, wo[gloss]), 0.0) + EPSILON)
    trans = lobe == 0
    pdf[trans] = 1.0 / (4 * math.pi)
    return pdf


def _sample_ggx_batch(r1: np.ndarray, r2: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """GGX importance sample half-vectors; alpha per ray."""
    xp = _xp(r1)
    a2 = alpha**2
    cos2 = (1 - r2) / (1 + (a2 - 1) * r2)
    sin = xp.sqrt(xp.maximum(1 - cos2, 0))
    phi = 2 * math.pi * r1
    return xp.stack(
        [sin * xp.cos(phi), sin * xp.sin(phi), xp.sqrt(xp.clip(cos2, 0, 1))], axis=-1
    )


def _tangent_frame_batch(N: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-ray tangent frame; up-vector flips near |N.y| == 1 like scalar."""
    xp = _xp(N)
    up = xp.where(
        (xp.abs(N[:, 1]) < 0.999)[:, np.newaxis],
        xp.array([0.0, 1.0, 0.0]),
        xp.array([1.0, 0.0, 0.0]),
    )
    t = _normalize(xp.cross(up, N))
    b = xp.cross(N, t)
    return t, b


# ── BVH (bounding volume hierarchy) ──────────────────────────────────────────


@dataclass
class BVHNode:
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    left: int = -1
    right: int = -1
    primitive_start: int = -1
    primitive_count: int = 0


class BVH:
    """Simple SAH-based BVH for mesh triangles."""

    def __init__(self, mesh: Mesh, max_prims_per_node: int = 4):
        self.mesh = mesh
        self.nodes: list[BVHNode] = []
        self._prim_indices: list[int] = []
        self._build(max_prims_per_node)

    def _build_recursive(self, indices: np.ndarray, max_prims: int) -> int:
        node_idx = len(self.nodes)
        v0 = self.mesh.vertices[self.mesh.faces[indices, 0]]
        v1 = self.mesh.vertices[self.mesh.faces[indices, 1]]
        v2 = self.mesh.vertices[self.mesh.faces[indices, 2]]

        all_verts = np.stack([v0, v1, v2], axis=1).reshape(-1, 3)
        aabb_min = all_verts.min(axis=0)
        aabb_max = all_verts.max(axis=0)

        self.nodes.append(
            BVHNode(
                aabb_min=aabb_min, aabb_max=aabb_max, primitive_start=node_idx, primitive_count=0
            )
        )

        if len(indices) <= max_prims:
            self.nodes[node_idx].primitive_start = len(self._prim_indices)
            self.nodes[node_idx].primitive_count = len(indices)
            self._prim_indices.extend(indices.tolist())
            return node_idx

        extent = aabb_max - aabb_min
        axis = int(np.argmax(extent))
        centroids = (v0 + v1 + v2) / 3.0
        order = centroids[:, axis].argsort()
        indices = indices[order]

        mid = len(indices) // 2
        left = self._build_recursive(indices[:mid], max_prims)
        right = self._build_recursive(indices[mid:], max_prims)
        self.nodes[node_idx].left = left
        self.nodes[node_idx].right = right
        return node_idx

    def _build(self, max_prims: int):
        self._prim_indices: list[int] = []
        n_faces = len(self.mesh.faces)
        indices = np.arange(n_faces)
        self._root = self._build_recursive(indices, max_prims)
        self._prepare_batch_tables(max_prims)

    def _prepare_batch_tables(self, max_prims: int):
        """Materialize node/triangle data as numpy tables for batch tracing.

        Built once per BVH so ``intersect_batch`` runs the whole ray batch
        with vectorized ops instead of a per-ray Python stack walk.
        """
        n_nodes = len(self.nodes)
        self._node_aabb_min = np.stack([n.aabb_min for n in self.nodes]).astype(np.float32)
        self._node_aabb_max = np.stack([n.aabb_max for n in self.nodes]).astype(np.float32)
        self._node_left = np.array([n.left for n in self.nodes], dtype=np.int32)
        self._node_right = np.array([n.right for n in self.nodes], dtype=np.int32)
        self._node_prim_count = np.array([n.primitive_count for n in self.nodes], dtype=np.int32)

        # Padded per-node primitive table (leaf prim counts are <= max_prims).
        prims = np.asarray(self._prim_indices, dtype=np.int32)
        self._node_prims = np.full((n_nodes, int(max_prims)), -1, dtype=np.int32)
        for ni in range(n_nodes):
            node = self.nodes[ni]
            if node.primitive_count > 0:
                self._node_prims[ni, : node.primitive_count] = prims[
                    node.primitive_start : node.primitive_start + node.primitive_count
                ]

        # Triangle edge tables for the vectorized Möller–Trumbore test.
        faces = self.mesh.faces
        v = self.mesh.vertices
        self._tri_v0 = v[faces[:, 0]].astype(np.float32)
        self._tri_e1 = (v[faces[:, 1]] - self._tri_v0).astype(np.float32)
        self._tri_e2 = (v[faces[:, 2]] - self._tri_v0).astype(np.float32)

    def _dev_tables(self, xp):
        """Batch tables in ``xp``'s namespace — numpy returns them as-is,
        cupy converts once and caches per BVH.

        The cache key is the module itself; ``Scene.build_bvh`` builds
        fresh BVHs every frame, so a device copy can never outlive the
        scene that produced it (tables are one mesh: nodes + per-triangle
        edges, sub-megabyte).
        """
        host = (
            self._node_aabb_min,
            self._node_aabb_max,
            self._node_left,
            self._node_right,
            self._node_prim_count,
            self._node_prims,
            self._tri_v0,
            self._tri_e1,
            self._tri_e2,
        )
        if xp is np:
            return host
        cache = getattr(self, "_dev_cache", None)
        if cache is not None and cache[0] is xp:
            return cache[1]
        dev = tuple(xp.asarray(t) for t in host)
        self._dev_cache = (xp, dev)
        return dev

    def intersect_batch(
        self, orig: np.ndarray, dire: np.ndarray, t_max: float = 1e30
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized ray-BVH intersection for a batch of rays.

        Frontier-based walk: every ray owns a slot in an explicit int
        stack; each while-loop iteration pops one node per live ray and
        processes the survivors with array ops (slab test, leaf prim
        gather, Möller–Trumbore). Closest hit per ray is kept, exactly
        like the scalar ``intersect``.

        Returns ``(t_hit (n,), face_idx (n,))`` — misses are ``1e30, -1``.

        Input-driven namespace: host arrays in → host tables and host
        results out; device arrays in → cached device tables, device
        results (the while-loop only probes ``.size`` — host metadata —
        so the whole walk runs without per-pop host round-trips).
        """
        xp = _xp(orig)
        n = len(orig)
        t_hit = xp.full(n, 1e30, dtype=np.float32)
        face_hit = xp.full(n, -1, dtype=np.int32)
        if n == 0 or not self.nodes:
            return t_hit, face_hit

        (
            na_min,
            na_max,
            node_left,
            node_right,
            node_pcount,
            node_prims,
            tri_v0,
            tri_e1,
            tri_e2,
        ) = self._dev_tables(xp)

        stack_depth = 64
        stack = xp.zeros((n, stack_depth), dtype=np.int32)
        sp = xp.ones(n, dtype=np.int32)
        stack[:, 0] = self._root
        inv = 1.0 / (dire + 1e-30)
        slot_off = xp.array([0, 1], dtype=np.int32)

        while True:
            active = xp.nonzero(sp > 0)[0]
            if active.size == 0:
                break
            sp[active] -= 1
            ni = stack[active, sp[active]]

            # Vectorized slab test for the popped nodes. A node whose entry
            # distance is beyond the ray's best hit so far cannot contain a
            # closer hit — prune it (exact; the scalar walk never pruned).
            o = orig[active]
            inv_a = inv[active]
            t1 = (na_min[ni] - o) * inv_a
            t2 = (na_max[ni] - o) * inv_a
            tnear = xp.minimum(t1, t2).max(axis=1)
            tfar = xp.maximum(t1, t2).min(axis=1)
            box = (tnear <= xp.minimum(tfar, t_hit[active])) & (tfar > 0)
            act = active[box]
            if act.size == 0:
                continue
            ni = ni[box]

            # Leaves only for the prim test: inner rows carry a -1 prim
            # row, so testing them is pure waste — merged wholesale it
            # quadrupled CPU walk time (numpy mt runs real flops per
            # pair, and inner pops outnumber leaves). One nonzero buys
            # the leaf subset; the push below needs no subset at all
            # (leaves are excluded by their -1 children).
            leaf = node_pcount[ni] > 0
            lrows = xp.nonzero(leaf)[0]
            if lrows.size:
                rays = act[lrows]
                nodes = ni[lrows]
                prims = node_prims[nodes]  # (m, K)
                m, k = prims.shape
                rr = xp.repeat(rays, k)
                ff = prims.reshape(-1)
                valid_pair = ff >= 0
                safe = xp.maximum(ff, 0)
                t = self._mt_batch(
                    tri_v0[safe],
                    tri_e1[safe],
                    tri_e2[safe],
                    orig[rr],
                    dire[rr],
                )
                t[~valid_pair] = 1e30
                t = t.reshape(m, k)
                best = t.argmin(axis=1)
                ar = xp.arange(m)
                bt = t[ar, best]
                closer = bt < t_hit[rays]
                sel = rays[closer]
                t_hit[sel] = bt[closer]
                face_hit[sel] = prims[ar[closer], best[closer]]

            # Push both children of every popped node in one pass (leaf
            # rows included — their -1 children fail `kids >= 0`, so the
            # old irows nonzero + gathers are gone). Slots match the
            # sequential order exactly — right at sp, left at sp+1, each
            # gated on stack room; blocked slots are rewritten with their
            # old value, so the scatter is a no-op for them.
            kids = xp.stack([node_right[ni], node_left[ni]], axis=1)
            sp_a = sp[act]
            slot = sp_a[:, None] + slot_off
            can = (kids >= 0) & (slot < stack_depth)
            rows2 = xp.repeat(act, 2)
            slot_cl = xp.minimum(slot, stack_depth - 1).reshape(-1)
            prev = stack[rows2, slot_cl]
            stack[rows2, slot_cl] = xp.where(can.reshape(-1), kids.reshape(-1), prev)
            sp[act] = sp_a + can.sum(axis=1)

        over = t_hit > t_max
        t_hit[over] = 1e30
        face_hit[over] = -1
        return t_hit, face_hit

    @staticmethod
    def _mt_batch(v0, e1, e2, orig, dire):
        """Vectorized Möller–Trumbore; returns t (float32) or 1e30 per pair.

        Host input runs the operator-by-operator numpy version (cross
        products written out explicitly — ``np.cross`` costs ~3x more
        on small (n, 3) batches, measured in the frame profile). Device
        input launches the single fused CUDA kernel instead: the
        per-call Python dispatch floor made the operator version cost
        ~4.5ms per leaf iteration on its own. Both compute identical
        math — same EPSILON rules, same 1e30 sentinel, same float32
        output cast. Namespace follows ``orig`` (see ``_xp``).
        """
        xp = _xp(orig)
        if xp is np:
            return BVH._mt_numpy(v0, e1, e2, orig, dire)
        return BVH._mt_cuda(v0, e1, e2, orig, dire)

    @staticmethod
    def _mt_cuda(v0, e1, e2, orig, dire):
        """One launch of the fused kernel (module-level ``_MT_CUDA_SRC``)."""
        import cupy as cp

        dt = np.result_type(v0.dtype, e1.dtype, e2.dtype, orig.dtype, dire.dtype)
        n = int(orig.shape[0])
        out = cp.empty(n, dtype=np.float32)
        if n:
            kern = _mt_cuda_kernel(cp, dt)
            a, b, c, o, d = (x.astype(dt, copy=False) for x in (v0, e1, e2, orig, dire))
            kern(((n + 127) // 128,), (128,), (a, b, c, o, d, out, n))
        return out

    @staticmethod
    def _mt_numpy(v0, e1, e2, orig, dire):
        xp = _xp(orig)
        # h = cross(dire, e2)
        h = xp.stack(
            [
                dire[:, 1] * e2[:, 2] - dire[:, 2] * e2[:, 1],
                dire[:, 2] * e2[:, 0] - dire[:, 0] * e2[:, 2],
                dire[:, 0] * e2[:, 1] - dire[:, 1] * e2[:, 0],
            ],
            axis=-1,
        )
        a = xp.sum(e1 * h, axis=-1)
        ok = xp.abs(a) > EPSILON
        f = xp.where(ok, 1.0 / xp.where(ok, a, 1.0), 0.0)
        s = orig - v0
        u = f * xp.sum(s * h, axis=-1)
        # q = cross(s, e1)
        q = xp.stack(
            [
                s[:, 1] * e1[:, 2] - s[:, 2] * e1[:, 1],
                s[:, 2] * e1[:, 0] - s[:, 0] * e1[:, 2],
                s[:, 0] * e1[:, 1] - s[:, 1] * e1[:, 0],
            ],
            axis=-1,
        )
        v = f * xp.sum(dire * q, axis=-1)
        t = f * xp.sum(e2 * q, axis=-1)
        hit = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > EPSILON)
        return xp.where(hit, t, 1e30).astype(np.float32)

    def intersect(
        self, orig: np.ndarray, dire: np.ndarray, t_max: float = 1e30
    ) -> tuple[float, int]:
        """Ray-BVH intersection. Returns (t_hit, face_index) or (inf, -1)."""
        t_min = np.float32(1e30)
        hit_face = -1
        stack = [self._root]

        while stack and len(stack) < 64:
            node = self.nodes[stack.pop()]

            if not self._aabb_intersect(orig, dire, node.aabb_min, node.aabb_max):
                continue

            if node.primitive_count > 0:
                for i in range(node.primitive_start, node.primitive_start + node.primitive_count):
                    fi = self._prim_indices[i]
                    t = self._ray_triangle(orig, dire, fi)
                    if EPSILON < t < t_min:
                        t_min = t
                        hit_face = fi
            else:
                if node.left >= 0:
                    stack.append(node.left)
                if node.right >= 0:
                    stack.append(node.right)

        return t_min, hit_face

    def _aabb_intersect(self, orig, dire, bmin, bmax) -> bool:
        inv = 1.0 / (dire + 1e-30)
        t1 = (bmin - orig) * inv
        t2 = (bmax - orig) * inv
        t_near = np.minimum(t1, t2).max()
        t_far = np.maximum(t1, t2).min()
        return t_near <= t_far and t_far > 0

    def _ray_triangle(self, orig, dire, fi) -> float:
        v0 = self.mesh.vertices[self.mesh.faces[fi, 0]]
        v1 = self.mesh.vertices[self.mesh.faces[fi, 1]]
        v2 = self.mesh.vertices[self.mesh.faces[fi, 2]]

        e1 = v1 - v0
        e2 = v2 - v0
        h = _cross(dire, e2)
        a = _dot(e1, h)
        if abs(a) < EPSILON:
            return 1e30

        f = 1.0 / a
        s = orig - v0
        u = f * _dot(s, h)
        if u < 0.0 or u > 1.0:
            return 1e30

        q = _cross(s, e1)
        v = f * _dot(dire, q)
        if v < 0.0 or u + v > 1.0:
            return 1e30

        t = f * _dot(e2, q)
        return t if t > EPSILON else 1e30


# ── Scene ────────────────────────────────────────────────────────────────────


@dataclass
class Scene:
    meshes: list[Mesh] = field(default_factory=list)
    materials: list[Material] = field(default_factory=lambda: [Material()])
    lights: list[Light] = field(default_factory=list)
    camera: Camera = field(default_factory=Camera)
    background: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.05, 0.05]))
    _bvh_list: list[BVH] = field(default_factory=list)

    def build_bvh(self):
        # 8 prims/leaf (BVH default is 4): halves node count at identical
        # hit results; frontier-walk frame time is pop-bound, not
        # depth-bound (measured 170ms vs 172ms per pass — keep the smaller
        # tree for the memory win).
        self._bvh_list = [BVH(mesh, max_prims_per_node=8) for mesh in self.meshes]

    def add_mesh(self, mesh: Mesh):
        self.meshes.append(mesh)
        self._bvh_list.append(BVH(mesh))


# ── Path tracer ──────────────────────────────────────────────────────────────


def _mesh_dev(mesh: Mesh, xp):
    """Mesh arrays in ``xp``'s namespace; the device copy caches on the
    mesh itself, keyed by (module, vertices identity) — a rebuilt mesh
    re-uploads, and CPU calls never touch the cache."""
    if xp is np:
        return mesh.vertices, mesh.faces, mesh.normals, mesh.material_idx
    cache = getattr(mesh, "_dev_cache", None)
    if cache is not None and cache[0] is xp and cache[1] is mesh.vertices:
        return cache[2]
    dev = (
        xp.asarray(mesh.vertices),
        xp.asarray(mesh.faces),
        xp.asarray(mesh.normals),
        xp.asarray(mesh.material_idx),
    )
    mesh._dev_cache = (xp, mesh.vertices, dev)
    return dev


class CyclesRenderer:
    """Path tracer with a device tier.

    Scene state is built as host numpy; a CUDA accelerator (selected via
    ``device=`` or ``SLO_CYCLES_DEVICE=cpu|cuda|auto``) keeps the frame's
    arrays resident on the GPU for the whole trace and crosses PCIe
    exactly once — ``render`` always returns a numpy (H, W, 3) tensor.
    """

    def __init__(
        self,
        scene: Scene,
        width: int = 640,
        height: int = 480,
        samples: int = 64,
        device: str | None = None,
    ):
        self.scene = scene
        self.width = width
        self.height = height
        self.samples = samples
        # Explicit device beats SLO_CYCLES_DEVICE beats auto (accelerator).
        self.device = device or os.environ.get("SLO_CYCLES_DEVICE") or "auto"
        self._xp = resolve_array_module(self.device)
        self._rng = np.random.default_rng(42)
        # Seed for the native core's counter RNG (keyed per path/depth/slot),
        # pinned to the same 42 the numpy path uses so both are deterministic
        # per seed. The bridge packs the scene + dispatches to the compiled
        # engine; None-fallback to the numpy oracle lives in render().
        self._seed = 42
        from domain.shell._internal._native import NativeTrace

        self._native = NativeTrace()

    def render(self) -> np.ndarray:
        """Render the scene. Returns (H, W, 3) float32 image tensor.

        All ``samples`` subpixel-jittered passes trace as ONE batch of
        S*H*W rays, not S separate H*W traces. That fusion is what makes
        the device tier pay: the frontier walk's host syncs are per call,
        not per ray, so one big call amortizes them across every sample
        (GTX 1050, 160x120x16: fused 2.95s vs 34.6s per-sample — slower
        than CPU unfused, 2.4x faster fused). Always returns numpy: a
        device render crosses PCIe exactly once, at the return below.
        """
        if not self.scene._bvh_list:
            self.scene.build_bvh()

        s = max(int(self.samples), 0)
        if s == 0:
            return np.zeros((self.height, self.width, 3), dtype=np.float32)

        xp = self._xp
        origins, _ = self.scene.camera.generate_rays(self.width, self.height)

        # Every sample's jitter drawn as one host RNG block up front —
        # CPU and device renders consume the identical stream. (The draw
        # order differs from the old interleave-with-trace loop, so images
        # differ from the pre-fusion renderer — deterministic per seed.)
        offset = self._rng.uniform(-0.5, 0.5, size=(s, self.height, self.width, 2))
        u = np.linspace(-0.5, 0.5, self.width)
        v = np.linspace(0.5, -0.5, self.height)
        uu, vv = np.meshgrid(u, v)
        aspect = self.width / self.height
        half_h = math.tan(math.radians(self.scene.camera.fov) / 2)
        half_w = half_h * aspect

        jitter_u = (uu[np.newaxis] + offset[..., 0] / self.width) * half_w * 2
        jitter_v = (vv[np.newaxis] + offset[..., 1] / self.height) * half_h * 2

        forward = _normalize(self.scene.camera.look_at - self.scene.camera.origin)
        right = _normalize(_cross(forward, self.scene.camera.up))
        up = _cross(right, forward)

        dirs_all = (
            forward[np.newaxis, np.newaxis, np.newaxis, :]
            + jitter_u[..., np.newaxis] * right[np.newaxis, np.newaxis, np.newaxis, :]
            + jitter_v[..., np.newaxis] * up[np.newaxis, np.newaxis, np.newaxis, :]
        )
        dirs_all = _normalize(dirs_all)  # (s, H, W, 3), host

        # One batch, one upload, one trace, one download for the frame:
        # _trace_batch only reads shape[:2], so samples stack as extra
        # rows — (s*H, W, 3) pretends s frames stacked vertically.
        # Native single-source core first (card 4de18e24): one C algorithm
        # compiled to gcc+OpenMP (cpu) or NVRTC (cuda). It traces the whole
        # fused batch as one flat ray set in a single call — no per-bounce
        # dispatch, no alive-ray compaction. Host in -> radiance host out,
        # so it skips the upload below. None => no engine for this device
        # -> fall through to the numpy oracle (identical math, slower).
        flat_o = np.broadcast_to(origins, (s, self.height, self.width, 3)).reshape(-1, 3)
        flat_d = dirs_all.reshape(-1, 3)
        rad = self._native.trace(
            xp, self.scene, flat_o, flat_d, self._seed, MAX_DEPTH, RR_START
        )
        if rad is not None:
            image = rad.reshape(s, self.height, self.width, 3).mean(axis=0)
            return np.clip(image, 0, 1)

        o = xp.asarray(
            np.broadcast_to(origins, (s, self.height, self.width, 3)).reshape(
                s * self.height, self.width, 3
            )
        )
        d = xp.asarray(dirs_all.reshape(s * self.height, self.width, 3))
        image = self._trace_batch(o, d).reshape(s, self.height, self.width, 3).mean(axis=0)
        if xp is not np:
            image = xp.asnumpy(image)
        return np.clip(image, 0, 1)

    def _trace_batch(self, origins: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        """Trace a batch of rays. Returns (H, W, 3) color.

        Input-driven namespace (see ``_xp``): the whole trace runs
        wherever the arrays live. ``render`` passes device arrays so a
        GPU frame never leaves the GPU; direct host calls stay host.
        """
        xp = _xp(origins)
        H, W = origins.shape[:2]
        N = H * W
        # Background + per-material tables: built from python objects on
        # the host once per call, then owned by `xp`. Gathering by device
        # index replaces the old per-ray Material-field list comps —
        # iterating a device array would sync once per element.
        bg = xp.asarray(self.scene.background)
        mats = self.scene.materials
        prop_t = xp.asarray(
            np.array(
                [[mm.transmission, mm.metallic, mm.ior, mm.roughness] for mm in mats],
                dtype=np.float64,
            )
        )
        emis_t = xp.asarray(np.array([mm.emission() for mm in mats], dtype=np.float64))
        throughput = xp.ones((N, 3), dtype=np.float32)
        radiance = xp.zeros((N, 3), dtype=np.float32)
        orig = origins.reshape(-1, 3).copy()
        dire = dirs.reshape(-1, 3).copy()
        alive = xp.ones(N, dtype=bool)

        for depth in range(MAX_DEPTH):
            if not alive.any():
                break

            act_orig = orig[alive]
            act_dire = dire[alive]
            t_hit, face_idx, hit_point, hit_normal, hit_mesh_idx, hit_mat_idx = (
                self._intersect_scene(act_orig, act_dire)
            )

            # Map results back to full arrays
            t_full = xp.full(N, 1e30, dtype=np.float32)
            fi_full = xp.full(N, -1, dtype=np.int32)
            hp_full = xp.zeros((N, 3), dtype=np.float32)
            hn_full = xp.zeros((N, 3), dtype=np.float32)
            mm_full = xp.full(N, -1, dtype=np.int32)
            mi_full = xp.full(N, -1, dtype=np.int32)

            act_idx = xp.where(alive)[0]
            t_full[act_idx] = t_hit
            fi_full[act_idx] = face_idx
            hp_full[act_idx] = hit_point
            hn_full[act_idx] = hit_normal
            mm_full[act_idx] = hit_mat_idx
            mi_full[act_idx] = hit_mesh_idx

            # Missed rays -> background
            missed = alive & (t_full >= 1e30)
            radiance[missed] += throughput[missed] * bg
            alive[missed] = False

            if not alive.any():
                break

            # Hit mask for this bounce
            hit = alive & (fi_full >= 0)

            # Emission — vectorized: gather per-material emission colors
            # once and index them by each hit ray's material.
            if hit.any():
                hi = xp.nonzero(hit)[0]
                radiance[hi] += throughput[hi] * emis_t[mm_full[hi]]

            # BSDF bounce — fully vectorized over the hit rays.
            if depth < MAX_DEPTH - 1 and hit.any():
                hi = xp.nonzero(hit)[0]
                m = hi.size
                wo = -dire[hi]
                nrm = hn_full[hi]
                P = hp_full[hi]
                mid = mm_full[hi]
                trans_v = prop_t[mid, 0]
                metal_v = prop_t[mid, 1]
                ior_v = prop_t[mid, 2]
                rough_v = prop_t[mid, 3]

                # Fixed RNG draw count per ray per bounce (deterministic
                # under the seed; draw order differs from the old scalar
                # sampler, so pixel-exact parity is not expected). Host
                # RNG, uploaded per bounce: same stream as the CPU path,
                # so CPU and GPU renders stay comparable.
                r1 = xp.asarray(self._rng.random(m))
                r2 = xp.asarray(self._rng.random(m))
                r3 = xp.asarray(self._rng.random(m))
                r4 = xp.asarray(self._rng.random(m))

                lobe = xp.full(m, 2, dtype=np.int8)  # diffuse
                do_trans = r3 < trans_v
                lobe[do_trans] = 0
                do_gloss = (~do_trans) & (r4 < metal_v)
                lobe[do_gloss] = 1

                wi = xp.empty((m, 3), dtype=np.float64)

                tr = do_trans
                if tr.any():
                    # Scalar form: eta = 1/ior if dot(N, wo) > 0 else ior,
                    # refract(-wo, N, eta) — per-ray eta via the batch path.
                    eta = xp.where(_dot(nrm[tr], wo[tr]) > 0, 1.0 / ior_v[tr], ior_v[tr])
                    refr, _ = _refract(-wo[tr], nrm[tr], eta)
                    wi[tr] = _normalize(refr)

                gl = do_gloss
                if gl.any():
                    h = _sample_ggx_batch(r1[gl], r2[gl], rough_v[gl] ** 2)
                    wi[gl] = _reflect(-wo[gl], h)

                df = ~(do_trans | do_gloss)
                if df.any():
                    ang = 2 * math.pi * r1[df]
                    s = xp.sqrt(r2[df])
                    local = xp.stack(
                        [xp.cos(ang) * s, xp.sin(ang) * s, xp.sqrt(1 - r2[df])], axis=-1
                    )
                    t, b = _tangent_frame_batch(nrm[df])
                    wi[df] = local[:, 0:1] * t + local[:, 1:2] * b + local[:, 2:3] * nrm[df]

                flip = _dot(wi, nrm) < 0
                wi[flip] = -wi[flip]

                alive_new = alive.copy()
                # .tolist(): one sync, python ints — list indexing and the
                # group masks below work unchanged on either namespace.
                for k in xp.unique(mid).tolist():
                    g = xp.nonzero(mid == k)[0]
                    mat = mats[k]
                    bsdf_val = _eval_bsdf_batch(wi[g], wo[g], nrm[g], mat)
                    pdf = _bsdf_pdf_batch(wo[g], wi[g], nrm[g], mat, lobe[g])
                    good = pdf > EPSILON
                    gk = g[good]
                    throughput[gk] *= bsdf_val[good] / pdf[good, np.newaxis]
                    dire[gk] = wi[gk]
                    orig[gk] = P[gk] + nrm[gk] * EPSILON * 2
                    alive_new[g[~good]] = False

                if depth >= RR_START:
                    p = xp.maximum(throughput[hi].max(axis=1), 0.05)
                    surv = xp.asarray(self._rng.random(m)) <= p
                    throughput[hi[surv]] /= p[surv, np.newaxis]
                    alive_new[hi[~surv]] = False

                alive = alive_new
            else:
                alive[:] = False

        return radiance.reshape(H, W, 3)

    def _intersect_scene(self, orig: np.ndarray, dire: np.ndarray):
        """Intersect all meshes for the whole ray batch at once.

        Per-mesh ``intersect_batch`` (vectorized frontier walk) keeps the
        closest hit; hit points, barycentrics, and normals are gathered
        with array ops grouped by mesh — no per-ray Python loops.
        Namespace follows ``orig``: device in → device tables
        (``_mesh_dev``) and device results out.
        """
        xp = _xp(orig)
        n = len(orig)
        t_hit = xp.full(n, 1e30, dtype=np.float32)
        face_idx = xp.full(n, -1, dtype=np.int32)
        mesh_idx = xp.full(n, -1, dtype=np.int32)

        for mi, bvh in enumerate(self.scene._bvh_list):
            t, fi = bvh.intersect_batch(orig, dire)
            closer = t < t_hit
            t_hit[closer] = t[closer]
            face_idx[closer] = fi[closer]
            mesh_idx[closer] = mi

        # Compute hit points and normals
        hit_point = orig + dire * t_hit[..., np.newaxis]
        hit_normal = xp.zeros_like(hit_point)
        hit_mat_idx = xp.zeros(n, dtype=np.int32)

        valid = xp.nonzero(face_idx >= 0)[0]
        if valid.size:
            for mi in xp.unique(mesh_idx[valid]).tolist():
                sel = valid[mesh_idx[valid] == mi]
                mesh = self.scene.meshes[mi]
                v_a, f_a, n_a, mi_a = _mesh_dev(mesh, xp)
                fi_a = face_idx[sel]
                v0 = v_a[f_a[fi_a, 0]]
                v1 = v_a[f_a[fi_a, 1]]
                v2 = v_a[f_a[fi_a, 2]]

                o = orig[sel]
                d = dire[sel]
                t = t_hit[sel]
                p = o + d * t[:, np.newaxis]
                e1, e2 = v1 - v0, v2 - v0
                w = p - v0
                d00 = _dot(e1, e1)
                d01 = _dot(e1, e2)
                d11 = _dot(e2, e2)
                d20 = _dot(w, e1)
                d21 = _dot(w, e2)
                denom = d00 * d11 - d01 * d01
                safe = xp.abs(denom) >= EPSILON
                den_s = xp.where(safe, denom, 1.0)
                w1 = (d11 * d20 - d01 * d21) / den_s
                w2 = (d00 * d21 - d01 * d20) / den_s
                bary = xp.clip(xp.stack([1.0 - w1 - w2, w1, w2], axis=-1), 0, 1)
                bary[~safe] = 1.0 / 3.0

                n0 = n_a[f_a[fi_a, 0]]
                n1 = n_a[f_a[fi_a, 1]]
                n2 = n_a[f_a[fi_a, 2]]
                hit_normal[sel] = _normalize(
                    bary[:, 0:1] * n0 + bary[:, 1:2] * n1 + bary[:, 2:3] * n2
                )
                hit_mat_idx[sel] = mi_a[fi_a]

        return t_hit, face_idx, hit_point, hit_normal, mesh_idx, hit_mat_idx

    def _triangle_barycentric(self, orig, dire, t, v0, v1, v2):
        """Barycentric weights of the hit point within triangle (v0, v1, v2)."""
        p = orig + dire * t
        e1, e2 = v1 - v0, v2 - v0
        w = p - v0
        d00 = _dot(e1, e1)
        d01 = _dot(e1, e2)
        d11 = _dot(e2, e2)
        d20 = _dot(w, e1)
        d21 = _dot(w, e2)
        denom = d00 * d11 - d01 * d01
        if abs(denom) < EPSILON:
            return np.array([1 / 3, 1 / 3, 1 / 3])
        w1 = (d11 * d20 - d01 * d21) / denom
        w2 = (d00 * d21 - d01 * d20) / denom
        bary = np.array([1.0 - w1 - w2, w1, w2])
        return np.clip(bary, 0, 1)

    def _sample_bsdf(self, wo, N, mat) -> np.ndarray:
        """Sample BSDF direction."""
        r1 = self._rng.random()
        r2 = self._rng.random()

        if self._rng.random() < mat.transmission:
            # Refraction
            eta = 1.0 / mat.ior if _dot(N, wo) > 0 else mat.ior
            wi, _ = _refract(-wo, N, eta)
            return _normalize(wi)

        # Diffuse + glossy mix
        if self._rng.random() < mat.metallic:
            # Glossy GGX sample
            alpha = mat.roughness**2
            h = self._sample_ggx(r1, r2, alpha)
            wi = _reflect(-wo, h)
        else:
            # Cosine-weighted hemisphere
            r1 = 2 * math.pi * r1
            r2 = r2
            s = math.sqrt(r2)
            wi_local = np.array([math.cos(r1) * s, math.sin(r1) * s, math.sqrt(1 - r2)])
            t, b = self._tangent_frame(N)
            wi = wi_local[0] * t + wi_local[1] * b + wi_local[2] * N

        if _dot(wi, N) < 0:
            wi = -wi
        return _normalize(wi)

    def _sample_ggx(self, r1, r2, alpha) -> np.ndarray:
        """GGX importance sample half-vector."""
        a2 = alpha**2
        cos2 = (1 - r2) / (1 + (a2 - 1) * r2)
        sin = math.sqrt(max(1 - cos2, 0))
        phi = 2 * math.pi * r1
        return np.array([sin * math.cos(phi), sin * math.sin(phi), math.sqrt(cos2)])

    def _tangent_frame(self, N) -> tuple[np.ndarray, np.ndarray]:
        """Build tangent frame from normal."""
        up = np.array([0, 1, 0]) if abs(N[1]) < 0.999 else np.array([1, 0, 0])
        t = _normalize(_cross(up, N))
        b = _cross(N, t)
        return t, b

    def _bsdf_pdf(self, wo, wi, N, mat) -> float:
        """PDF for BSDF sample."""
        if self._rng.random() < mat.transmission:
            return 1.0 / (4 * math.pi)

        if self._rng.random() < mat.metallic:
            H = _normalize(wo + wi)
            alpha = mat.roughness**2
            NdotH = max(_dot(N, H), 0)
            alpha2 = alpha**2
            D = _ggx_distribution(NdotH, alpha2)
            return D * max(_dot(N, H), 0) / (4 * max(_dot(H, wo), 0) + EPSILON)

        return max(_dot(N, wi), 0) / math.pi

    # ── State tensor output (for NPU) ────────────────────────────────────────

    def render_state_tensors(self) -> dict[str, np.ndarray]:
        """Render and return state tensors for neural processing.

        Returns dict with:
          - 'image': (H, W, 3) rendered image
          - 'depth': (H, W) linear depth
          - 'normal': (H, W, 3) world-space normals
          - 'albedo': (H, W, 3) base color
          - 'emission': (H, W, 3) emission pass
          - 'mask': (H, W) object ID mask
        """
        image = self.render()
        depth = np.zeros((self.height, self.width), dtype=np.float32)
        normal = np.zeros((self.height, self.width, 3), dtype=np.float32)
        albedo = np.zeros((self.height, self.width, 3), dtype=np.float32)
        emission = np.zeros((self.height, self.width, 3), dtype=np.float32)
        mask = np.zeros((self.height, self.width), dtype=np.int32)

        origins, dirs = self.scene.camera.generate_rays(self.width, self.height)
        flat_orig = origins.reshape(-1, 3)
        flat_dir = dirs.reshape(-1, 3)

        for i in range(len(flat_orig)):
            t, fi, mesh_i = self._trace_single(flat_orig[i], flat_dir[i])
            if fi >= 0:
                h = i // self.width
                w = i % self.width
                depth[h, w] = t
                mask[h, w] = mesh_i + 1

                mesh = self.scene.meshes[mesh_i]
                face = mesh.faces[fi]
                bary = self._triangle_barycentric(
                    flat_orig[i],
                    flat_dir[i],
                    t,
                    mesh.vertices[face[0]],
                    mesh.vertices[face[1]],
                    mesh.vertices[face[2]],
                )
                n0, n1, n2 = mesh.normals[face]
                normal[h, w] = _normalize(bary[0] * n0 + bary[1] * n1 + bary[2] * n2)

                mat = self.scene.materials[mesh.material_idx[fi]]
                albedo[h, w] = mat.base_color
                emission[h, w] = mat.emission()

        return {
            "image": image,
            "depth": depth,
            "normal": normal,
            "albedo": albedo,
            "emission": emission,
            "mask": mask,
        }

    def _trace_single(self, orig, dire) -> tuple[float, int, int]:
        """Single ray trace — returns (t, face_idx, mesh_idx)."""
        t_min = 1e30
        fi_min = -1
        mi_min = -1
        for mi, bvh in enumerate(self.scene._bvh_list):
            t, fi = bvh.intersect(orig, dire)
            if t < t_min:
                t_min = t
                fi_min = fi
                mi_min = mi
        return t_min, fi_min, mi_min


# ── Primitive builders ───────────────────────────────────────────────────────


def create_plane(size: float = 2.0, y: float = -1.0, mat_idx: int = 0) -> Mesh:
    """Create a ground plane mesh."""
    s = size / 2
    vertices = np.array([[-s, y, -s], [s, y, -s], [s, y, s], [-s, y, s]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    normals = np.array([[0, 1, 0]] * 4)
    material_idx = np.array([mat_idx, mat_idx])
    return Mesh(vertices=vertices, faces=faces, normals=normals, material_idx=material_idx)


def create_sphere(
    radius: float = 0.5, center: np.ndarray = None, segments: int = 16, mat_idx: int = 0
) -> Mesh:
    """Create a UV sphere mesh."""
    if center is None:
        center = np.zeros(3)
    rings = segments
    sectors = segments * 2
    verts = []
    norms = []
    faces = []

    for i in range(rings + 1):
        theta = math.pi * i / rings
        for j in range(sectors):
            phi = 2 * math.pi * j / sectors
            x = radius * math.sin(theta) * math.cos(phi) + center[0]
            y = radius * math.cos(theta) + center[1]
            z = radius * math.sin(theta) * math.sin(phi) + center[2]
            verts.append([x, y, z])
            norms.append(
                [math.sin(theta) * math.cos(phi), math.cos(theta), math.sin(theta) * math.sin(phi)]
            )

    for i in range(rings):
        for j in range(sectors):
            i0 = i * sectors + j
            i1 = i0 + 1
            i2 = i0 + sectors
            i3 = i2 + 1
            if j == sectors - 1:
                i1 = i * sectors
                i3 = i2
            if i < rings - 1:
                faces.append([i0, i1, i3])
                faces.append([i0, i3, i2])

    mat_idx_arr = np.full(len(faces), mat_idx, dtype=np.int32)
    return Mesh(
        vertices=np.array(verts, dtype=np.float64),
        faces=np.array(faces, dtype=np.int32),
        normals=np.array(norms, dtype=np.float64),
        material_idx=mat_idx_arr,
    )


def create_cube(size: float = 1.0, center: np.ndarray = None, mat_idx: int = 0) -> Mesh:
    """Create a unit cube mesh."""
    if center is None:
        center = np.zeros(3)
    s = size / 2
    c = center
    vertices = np.array(
        [
            [c[0] - s, c[1] - s, c[2] - s],
            [c[0] + s, c[1] - s, c[2] - s],
            [c[0] + s, c[1] + s, c[2] - s],
            [c[0] - s, c[1] + s, c[2] - s],
            [c[0] - s, c[1] - s, c[2] + s],
            [c[0] + s, c[1] - s, c[2] + s],
            [c[0] + s, c[1] + s, c[2] + s],
            [c[0] - s, c[1] + s, c[2] + s],
        ]
    )
    faces = np.array(
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
        ]
    )
    mat_idx_arr = np.full(len(faces), mat_idx, dtype=np.int32)
    mesh = Mesh(vertices=vertices, faces=faces, material_idx=mat_idx_arr)
    return mesh
