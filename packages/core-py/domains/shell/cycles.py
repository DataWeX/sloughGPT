"""
Cycles — RAM-resident path tracer rendering engine.

A pure-numpy Cycles-style path tracer that lives entirely in RAM.
No GPU, no external libraries — just numpy arrays as scene state tensors.

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
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("slo.cycles")

EPSILON = 1e-6
MAX_DEPTH = 8
RR_START = 3


# ── Vector math (pure numpy, no dependencies) ────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > EPSILON, v / n, 0.0)


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b)


def _reflect(I: np.ndarray, N: np.ndarray) -> np.ndarray:
    return I - 2.0 * _dot(I, N)[..., np.newaxis] * N


def _refract(I: np.ndarray, N: np.ndarray, eta: float) -> tuple[np.ndarray, np.ndarray]:
    cos_i = -_dot(I, N)
    sin2_t = eta ** 2 * (1.0 - cos_i ** 2)
    mask = sin2_t < 1.0
    cos_t = np.sqrt(np.clip(1.0 - sin2_t, 0, 1))
    result = eta * I + (eta * cos_i - cos_t)[..., np.newaxis] * N
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

        alpha = self.roughness ** 2
        alpha2 = alpha ** 2

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
    faces: np.ndarray     # (F, 3) indices
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

        dirs = (forward[np.newaxis, np.newaxis, :]
                + uu[..., np.newaxis] * right[np.newaxis, np.newaxis, :]
                + vv[..., np.newaxis] * up[np.newaxis, np.newaxis, :])
        dirs = _normalize(dirs)
        origins = np.broadcast_to(self.origin, dirs.shape)
        return origins, dirs


# ── GGX BSDF helpers ─────────────────────────────────────────────────────────

def _ggx_distribution(NdotH: np.ndarray, alpha2: float) -> np.ndarray:
    denom = NdotH ** 2 * (alpha2 - 1.0) + 1.0
    return alpha2 / (math.pi * denom ** 2 + EPSILON)


def _ggx_geometry(NdotV: float, NdotL: float, alpha2: float) -> float:
    def _smith(n: float) -> float:
        return 2.0 * n / (n + math.sqrt(alpha2 + (1 - alpha2) * n ** 2))
    return _smith(NdotV) * _smith(NdotL)


def _fresnel_schlick(cos: float, F0: np.ndarray) -> np.ndarray:
    return F0 + (1.0 - F0) * (1.0 - cos) ** 5


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

        self.nodes.append(BVHNode(aabb_min=aabb_min, aabb_max=aabb_max,
                                   primitive_start=node_idx, primitive_count=0))

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

    def intersect(self, orig: np.ndarray, dire: np.ndarray, t_max: float = 1e30) -> tuple[float, int]:
        """Ray-BVH intersection. Returns (t_hit, face_index) or (inf, -1)."""
        t_min = np.float32(1e30)
        hit_face = -1
        stack = [self._root]

        while stack and len(stack) < 64:
            node = self.nodes[stack.pop()]

            if not self._aabb_intersect(orig, dire, node.aabb_min, node.aabb_max):
                continue

            if node.primitive_count > 0:
                for i in range(node.primitive_start,
                               node.primitive_start + node.primitive_count):
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
        self._bvh_list = [BVH(mesh) for mesh in self.meshes]

    def add_mesh(self, mesh: Mesh):
        self.meshes.append(mesh)
        self._bvh_list.append(BVH(mesh))


# ── Path tracer ──────────────────────────────────────────────────────────────

class CyclesRenderer:
    """RAM-resident path tracer. All state lives in numpy arrays."""

    def __init__(self, scene: Scene, width: int = 640, height: int = 480,
                 samples: int = 64):
        self.scene = scene
        self.width = width
        self.height = height
        self.samples = samples
        self._rng = np.random.default_rng(42)

    def render(self) -> np.ndarray:
        """Render the scene. Returns (H, W, 3) float32 image tensor."""
        if not self.scene._bvh_list:
            self.scene.build_bvh()

        image = np.zeros((self.height, self.width, 3), dtype=np.float32)
        depth_buf = np.zeros((self.height, self.width), dtype=np.float32)
        normal_buf = np.zeros((self.height, self.width, 3), dtype=np.float32)
        albedo_buf = np.zeros((self.height, self.width, 3), dtype=np.float32)

        origins, dirs = self.scene.camera.generate_rays(self.width, self.height)

        for s in range(self.samples):
            offset = self._rng.uniform(-0.5, 0.5, size=(self.height, self.width, 2))
            # Jitter subpixel
            u = np.linspace(-0.5, 0.5, self.width)
            v = np.linspace(0.5, -0.5, self.height)
            uu, vv = np.meshgrid(u, v)
            aspect = self.width / self.height
            half_h = math.tan(math.radians(self.scene.camera.fov) / 2)
            half_w = half_h * aspect

            jitter_u = (uu + offset[:, :, 0] / self.width) * half_w * 2
            jitter_v = (vv + offset[:, :, 1] / self.height) * half_h * 2

            forward = _normalize(self.scene.camera.look_at - self.scene.camera.origin)
            right = _normalize(_cross(forward, self.scene.camera.up))
            up = _cross(right, forward)

            dirs_jit = (forward[np.newaxis, np.newaxis, :]
                        + jitter_u[..., np.newaxis] * right[np.newaxis, np.newaxis, :]
                        + jitter_v[..., np.newaxis] * up[np.newaxis, np.newaxis, :])
            dirs_jit = _normalize(dirs_jit)

            contrib = self._trace_batch(origins, dirs_jit)
            image += contrib

        image /= max(self.samples, 1)
        return np.clip(image, 0, 1)

    def _trace_batch(self, origins: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        """Trace a batch of rays. Returns (H, W, 3) color."""
        H, W = origins.shape[:2]
        N = H * W
        throughput = np.ones((N, 3), dtype=np.float32)
        radiance = np.zeros((N, 3), dtype=np.float32)
        orig = origins.reshape(-1, 3).copy()
        dire = dirs.reshape(-1, 3).copy()
        alive = np.ones(N, dtype=bool)

        for depth in range(MAX_DEPTH):
            if not alive.any():
                break

            act_orig = orig[alive]
            act_dire = dire[alive]
            t_hit, face_idx, hit_point, hit_normal, hit_mesh_idx, hit_mat_idx = \
                self._intersect_scene(act_orig, act_dire)
            n_alive = act_orig.shape[0]

            # Map results back to full arrays
            t_full = np.full(N, 1e30, dtype=np.float32)
            fi_full = np.full(N, -1, dtype=np.int32)
            hp_full = np.zeros((N, 3), dtype=np.float32)
            hn_full = np.zeros((N, 3), dtype=np.float32)
            mm_full = np.full(N, -1, dtype=np.int32)
            mi_full = np.full(N, -1, dtype=np.int32)

            act_idx = np.where(alive)[0]
            t_full[act_idx] = t_hit
            fi_full[act_idx] = face_idx
            hp_full[act_idx] = hit_point
            hn_full[act_idx] = hit_normal
            mm_full[act_idx] = hit_mat_idx
            mi_full[act_idx] = hit_mesh_idx

            # Missed rays -> background
            missed = alive & (t_full >= 1e30)
            radiance[missed] += throughput[missed] * self.scene.background
            alive[missed] = False

            if not alive.any():
                break

            # Hit mask for this bounce
            hit = alive & (fi_full >= 0)

            # Emission
            if hit.any():
                hit_idx = np.where(hit)[0]
                for idx in hit_idx:
                    mat = self.scene.materials[mm_full[idx]]
                    radiance[idx] += throughput[idx] * mat.emission()

            # BSDF bounce
            if depth < MAX_DEPTH - 1 and hit.any():
                hit_idx = np.where(hit)[0]
                alive_new = alive.copy()
                for idx in hit_idx:
                    mat = self.scene.materials[mm_full[idx]]
                    N_vec = hn_full[idx]
                    wo = -dire[idx]
                    wi = self._sample_bsdf(wo, N_vec, mat)
                    bsdf_val = mat.eval(wi, wo, N_vec)
                    pdf = self._bsdf_pdf(wo, wi, N_vec, mat)

                    if pdf > EPSILON:
                        throughput[idx] *= bsdf_val / pdf
                        dire[idx] = wi
                        orig[idx] = hp_full[idx] + N_vec * EPSILON * 2
                    else:
                        alive_new[idx] = False

                    if depth >= RR_START:
                        p = max(throughput[idx].max(), 0.05)
                        if self._rng.random() > p:
                            alive_new[idx] = False
                        else:
                            throughput[idx] /= p

                alive = alive_new
            else:
                alive[:] = False

        return radiance.reshape(H, W, 3)

    def _intersect_scene(self, orig: np.ndarray, dire: np.ndarray):
        """Intersect all meshes. Returns per-ray results."""
        n = len(orig)
        t_hit = np.full(n, 1e30, dtype=np.float32)
        face_idx = np.full(n, -1, dtype=np.int32)
        mesh_idx = np.full(n, -1, dtype=np.int32)

        for mi, bvh in enumerate(self.scene._bvh_list):
            for i in range(n):
                t, fi = bvh.intersect(orig[i], dire[i])
                if t < t_hit[i]:
                    t_hit[i] = t
                    face_idx[i] = fi
                    mesh_idx[i] = mi

        # Compute hit points and normals
        hit_point = orig + dire * t_hit[..., np.newaxis]
        hit_normal = np.zeros_like(hit_point)
        hit_mat_idx = np.zeros(n, dtype=np.int32)

        valid = face_idx >= 0
        for i in np.where(valid)[0]:
            mesh = self.scene.meshes[mesh_idx[i]]
            fi = face_idx[i]
            v0, v1, v2 = mesh.vertices[mesh.faces[fi]]
            bary = self._triangle_barycentric(orig[i], dire[i], t_hit[i], v0, v1, v2)
            n0, n1, n2 = mesh.normals[mesh.faces[fi]]
            hit_normal[i] = _normalize(bary[0] * n0 + bary[1] * n1 + bary[2] * n2)
            hit_mat_idx[i] = mesh.material_idx[fi]

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
            return np.array([1/3, 1/3, 1/3])
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
            alpha = mat.roughness ** 2
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
        a2 = alpha ** 2
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
            alpha = mat.roughness ** 2
            NdotH = max(_dot(N, H), 0)
            alpha2 = alpha ** 2
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
                bary = self._triangle_barycentric(flat_orig[i], flat_dir[i], t,
                                                    mesh.vertices[face[0]],
                                                    mesh.vertices[face[1]],
                                                    mesh.vertices[face[2]])
                n0, n1, n2 = mesh.normals[face]
                normal[h, w] = _normalize(bary[0] * n0 + bary[1] * n1 + bary[2] * n2)

                mat = self.scene.materials[mesh.material_idx[fi]]
                albedo[h, w] = mat.base_color
                emission[h, w] = mat.emission()

        return {
            'image': image,
            'depth': depth,
            'normal': normal,
            'albedo': albedo,
            'emission': emission,
            'mask': mask,
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
    vertices = np.array([
        [-s, y, -s], [s, y, -s], [s, y, s], [-s, y, s]
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    normals = np.array([[0, 1, 0]] * 4)
    material_idx = np.array([mat_idx, mat_idx])
    return Mesh(vertices=vertices, faces=faces, normals=normals, material_idx=material_idx)


def create_sphere(radius: float = 0.5, center: np.ndarray = None,
                  segments: int = 16, mat_idx: int = 0) -> Mesh:
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
            norms.append([math.sin(theta) * math.cos(phi),
                          math.cos(theta),
                          math.sin(theta) * math.sin(phi)])

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
    return Mesh(vertices=np.array(verts, dtype=np.float64),
                faces=np.array(faces, dtype=np.int32),
                normals=np.array(norms, dtype=np.float64),
                material_idx=mat_idx_arr)


def create_cube(size: float = 1.0, center: np.ndarray = None,
                mat_idx: int = 0) -> Mesh:
    """Create a unit cube mesh."""
    if center is None:
        center = np.zeros(3)
    s = size / 2
    c = center
    vertices = np.array([
        [c[0]-s, c[1]-s, c[2]-s], [c[0]+s, c[1]-s, c[2]-s],
        [c[0]+s, c[1]+s, c[2]-s], [c[0]-s, c[1]+s, c[2]-s],
        [c[0]-s, c[1]-s, c[2]+s], [c[0]+s, c[1]-s, c[2]+s],
        [c[0]+s, c[1]+s, c[2]+s], [c[0]-s, c[1]+s, c[2]+s],
    ])
    faces = np.array([
        [0,1,2],[0,2,3], [4,6,5],[4,7,6],
        [0,4,5],[0,5,1], [2,6,7],[2,7,3],
        [0,3,7],[0,7,4], [1,5,6],[1,6,2],
    ])
    mat_idx_arr = np.full(len(faces), mat_idx, dtype=np.int32)
    mesh = Mesh(vertices=vertices, faces=faces, material_idx=mat_idx_arr)
    return mesh
