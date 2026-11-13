/* scene_build.c — single-source scene-build core: voxel mesh extraction +
 * BVH construction. The per-frame build_scene() in world_render.py was
 * pure Python (per-voxel cube emission, recursive BVH subdivision) and,
 * after trace_core.c made tracing ~27x faster, became the majority of
 * every world frame (card bdfcb8cb: build_scene 57% of the native frame).
 *
 * Compiled by domain/slolib/_internal/native.py to BOTH engines, same
 * contract as trace_core.c:
 *   CPU: gcc -O3 -fopenmp -> ctypes.  entry_mesh_cpu: OpenMP per voxel.
 *   GPU: NVRTC -> RawKernel. scene_mesh_gpu: one thread per voxel;
 *        scene_bvh_gpu: one thread runs the serial BVH build.
 *
 * SLO_Q is the ONLY target-specific token (empty on CPU, __device__ on
 * NVRTC) — no #ifdef in the algorithms themselves. No libc, no recursion
 * (NVRTC forbids both): the BVH walk is an explicit-stack, pre-order
 * iterative build that appends nodes in exactly the order the Python
 * recursion did, and the centroid sort is a bottom-up merge sort whose
 * (value, arrival-position) total order equals numpy's
 * argsort(kind="stable") — so the Python oracle and this file produce
 * byte-identical trees, not merely equivalent ones.
 *
 * Float parity: every value the Python path rounds exactly once from
 * double to float32 (voxel centers: (coord + 0.5) * vs) is rounded once
 * here too; centroids stay float32 throughout (v0 + v1 + v2) / 3.0f with
 * no fma-able pattern, so gcc and NVRTC agree with numpy bit-for-bit.
 */

/* SLO_Q is the single target-specific token: NVRTC defines __CUDACC__,
 * so helpers compile as __device__ there and as plain static inline
 * under gcc. The entry points diverge once, at the bottom. */
#ifdef __CUDACC__
#define SLO_Q __device__
#else
#define SLO_Q
#endif

/* Flattened to explicit pointers/scalars like SLO_TRACE_PARAMS: a struct
 * does not marshal reliably across ctypes (by-value layout) and NVRTC
 * (device pointer params). */

#define SLO_MESH_PARAMS \
    const long *flat_idx, const int *mat_idx, long n, \
    int ny, int nz, double vs, \
    float *verts, int *faces, int *fmat

#define SLO_BVH_PARAMS \
    const float *verts, const int *faces, long n_faces, int max_prims, \
    float *out_min, float *out_max, int *out_left, int *out_right, \
    int *out_pstart, int *out_pcnt, \
    int *perm, int *tmp, float *keys, int *prim_out, \
    int *out_n_nodes, long *out_n_prims

/* ── voxel mesh ─────────────────────────────────────────────────────────── */

/* cube_verts from world_render._build_voxel_mesh, as sign offsets so the
 * per-voxel multiply is exact for any voxel_size (offset * half == the
 * numpy value: (float)(+-vs * 0.5), sign included). */
SLO_Q static const float slo_cube_off[8][3] = {
    {-1.0f, -1.0f, -1.0f}, { 1.0f, -1.0f, -1.0f},
    { 1.0f,  1.0f, -1.0f}, {-1.0f,  1.0f, -1.0f},
    {-1.0f, -1.0f,  1.0f}, { 1.0f, -1.0f,  1.0f},
    { 1.0f,  1.0f,  1.0f}, {-1.0f,  1.0f,  1.0f},
};

/* cube_faces verbatim from world_render._build_voxel_mesh. */
SLO_Q static const int slo_cube_faces[12][3] = {
    {0, 1, 2}, {0, 2, 3}, {4, 6, 5}, {4, 7, 6},
    {0, 4, 5}, {0, 5, 1}, {2, 6, 7}, {2, 7, 3},
    {0, 3, 7}, {0, 7, 4}, {1, 5, 6}, {1, 6, 2},
};

/* One voxel -> its 8 vertices, 12 faces, 12 face materials. `i` is the
 * voxel's position in the solid list (vertex offset i*8, like vi in the
 * Python loop); `fi` is its flat grid cell. Coords match
 * WorldGrid.coords: x-major (x = fi / (ny*nz), z fastest). */
SLO_Q static void slo_mesh_voxel(long i, long fi, int m, int ny, int nz,
                                 double vs, float *verts, int *faces, int *fmat) {
    long yz = (long)ny * nz;
    long x = fi / yz;
    long r1 = fi % yz;
    long y = r1 / nz;
    long z = r1 % nz;
    /* Python computes the center in double, rounds to float32 once. */
    float c[3];
    c[0] = (float)(((double)x + 0.5) * vs);
    c[1] = (float)(((double)y + 0.5) * vs);
    c[2] = (float)(((double)z + 0.5) * vs);
    float half = (float)(vs * 0.5);
    long vi = i * 8;
    long fo = i * 12;
    int k, f, c3;
    for (k = 0; k < 8; ++k)
        for (c3 = 0; c3 < 3; ++c3)
            verts[(vi + k) * 3 + c3] = slo_cube_off[k][c3] * half + c[c3];
    for (f = 0; f < 12; ++f)
        for (c3 = 0; c3 < 3; ++c3)
            faces[(fo + f) * 3 + c3] = slo_cube_faces[f][c3] + (int)vi;
    for (f = 0; f < 12; ++f)
        fmat[fo + f] = m;
}

/* ── BVH ────────────────────────────────────────────────────────────────── */

typedef struct { long lo, hi; int node, parent; } SloBvhFrame;
/* parent >= 0 marks a right-child frame: the node is appended on POP (the
 * pre-order point where the Python right recursion would have entered),
 * after the entire left subtree has been built. parent == -1 means the
 * node already exists (`node`). */

SLO_Q static int slo_bvh_append(const float *verts, const int *faces,
                                const int *perm, long lo, long hi,
                                float *out_min, float *out_max,
                                int *out_left, int *out_right,
                                int *out_pstart, int *out_pcnt, int *n_nodes) {
    int ni = (*n_nodes)++;
    float mn[3], mx[3];
    int c3, k;
    for (c3 = 0; c3 < 3; ++c3) { mn[c3] = 1e30f; mx[c3] = -1e30f; }
    for (long j = lo; j < hi; ++j) {
        const int *fc = faces + 3L * perm[j];
        for (k = 0; k < 3; ++k) {
            const float *v = verts + 3L * fc[k];
            for (c3 = 0; c3 < 3; ++c3) {
                if (v[c3] < mn[c3]) mn[c3] = v[c3];
                if (v[c3] > mx[c3]) mx[c3] = v[c3];
            }
        }
    }
    for (c3 = 0; c3 < 3; ++c3) {
        out_min[ni * 3 + c3] = mn[c3];
        out_max[ni * 3 + c3] = mx[c3];
    }
    out_left[ni] = -1;
    out_right[ni] = -1;
    out_pstart[ni] = ni;   /* python placeholder; only leaf pstart is read */
    out_pcnt[ni] = 0;
    return ni;
}

/* np.argmax(extent): first index of the max (strict > keeps the first). */
SLO_Q static int slo_bvh_axis(const float *mn, const float *mx) {
    int axis = 0;
    float best = mx[0] - mn[0];
    float e = mx[1] - mn[1];
    if (e > best) { best = e; axis = 1; }
    e = mx[2] - mn[2];
    if (e > best) { axis = 2; }
    return axis;
}

/* centroids[:, axis] as float32: (a + b + c) / 3.0f, same single-rounding
 * chain as the numpy expression on float32 vertex rows. */
SLO_Q static void slo_bvh_centroids(const float *verts, const int *faces,
                                    const int *perm, long lo, long hi,
                                    int axis, float *keys) {
    for (long j = lo; j < hi; ++j) {
        const int *fc = faces + 3L * perm[j];
        const float *a = verts + 3L * fc[0];
        const float *b = verts + 3L * fc[1];
        const float *c = verts + 3L * fc[2];
        keys[perm[j]] = (a[axis] + b[axis] + c[axis]) / 3.0f;
    }
}

/* Bottom-up merge sort of perm[lo:hi) by (keys[value], arrival position).
 * The arrival-position tiebreak makes the permutation a total order equal
 * to numpy's argsort(kind="stable") — any correct sort under this
 * comparator yields the identical permutation. */
SLO_Q static void slo_bvh_sort(int *perm, int *tmp, const float *keys,
                               long lo, long hi) {
    long n = hi - lo;
    for (long step = 1; step < n; step *= 2) {
        for (long s = lo; s < hi; s += 2 * step) {
            long m = s + step;
            if (m >= hi) {                      /* tail run: copy through */
                for (long p = s; p < hi; ++p) tmp[p] = perm[p];
                break;
            }
            long e = s + 2 * step;
            if (e > hi) e = hi;
            long i = s, j = m, k = s;
            while (i < m && j < e) {
                if (keys[perm[j]] < keys[perm[i]]) tmp[k++] = perm[j++];
                else tmp[k++] = perm[i++];      /* equal -> left first */
            }
            while (i < m) tmp[k++] = perm[i++];
            while (j < e) tmp[k++] = perm[j++];
        }
        for (long p = lo; p < hi; ++p) perm[p] = tmp[p];
    }
}

/* Iterative pre-order build appending nodes in exactly the order
 * BVH._build_recursive did: parent, then the whole left subtree, then the
 * right node. perm[] holds the working face-id order (segments are
 * contiguous ranges); prim_out collects leaves in pre-order like
 * _prim_indices. Node cap 2*n_faces - 1, prim count n_faces. */
SLO_Q static void slo_bvh_build(const float *verts, const int *faces,
                                long n_faces, int max_prims,
                                float *out_min, float *out_max,
                                int *out_left, int *out_right,
                                int *out_pstart, int *out_pcnt,
                                int *perm, int *tmp, float *keys,
                                int *prim_out, int *out_n_nodes,
                                long *out_n_prims) {
    SloBvhFrame stack[96];
    int sp = 0;
    int n_nodes = 0;
    long n_prims = 0;
    long i;

    for (i = 0; i < n_faces; ++i) perm[i] = (int)i;

    int root = slo_bvh_append(verts, faces, perm, 0, n_faces, out_min, out_max,
                              out_left, out_right, out_pstart, out_pcnt,
                              &n_nodes);
    stack[sp].lo = 0; stack[sp].hi = n_faces;
    stack[sp].node = root; stack[sp].parent = -1; sp++;

    while (sp > 0) {
        SloBvhFrame fr = stack[--sp];
        int ni = fr.node;
        if (fr.parent >= 0) {
            /* right marker: append after the left subtree (pre-order) */
            ni = slo_bvh_append(verts, faces, perm, fr.lo, fr.hi, out_min,
                                out_max, out_left, out_right, out_pstart,
                                out_pcnt, &n_nodes);
            out_right[fr.parent] = ni;
        }
        long l = fr.hi - fr.lo;
        if (l <= (long)max_prims) {
            out_pstart[ni] = (int)n_prims;
            out_pcnt[ni] = (int)l;
            for (i = 0; i < l; ++i) prim_out[n_prims++] = perm[fr.lo + i];
            continue;
        }
        int axis = slo_bvh_axis(out_min + 3L * ni, out_max + 3L * ni);
        slo_bvh_centroids(verts, faces, perm, fr.lo, fr.hi, axis, keys);
        slo_bvh_sort(perm, tmp, keys, fr.lo, fr.hi);
        long mid = l / 2;
        int ln = slo_bvh_append(verts, faces, perm, fr.lo, fr.lo + mid,
                                out_min, out_max, out_left, out_right,
                                out_pstart, out_pcnt, &n_nodes);
        out_left[ni] = ln;
        /* right pushed first so left pops first (median split bounds the
         * live stack at ceil(log2(n_faces)) + 1 frames; 96 is ample) */
        stack[sp].lo = fr.lo + mid; stack[sp].hi = fr.hi;
        stack[sp].node = -1; stack[sp].parent = ni; sp++;
        stack[sp].lo = fr.lo; stack[sp].hi = fr.lo + mid;
        stack[sp].node = ln; stack[sp].parent = -1; sp++;
    }
    *out_n_nodes = n_nodes;
    *out_n_prims = n_prims;
}

/* ── entries: the ONLY place the two engines diverge ────────────────────── */

#ifdef __CUDACC__

/* One thread per voxel: independent writes, no sharing. */
extern "C" __global__ void scene_mesh_gpu(SLO_MESH_PARAMS) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    slo_mesh_voxel(i, flat_idx[i], mat_idx[i], ny, nz, vs, verts, faces, fmat);
}

/* The BVH build is a serial dependency chain (each split feeds its
 * children); one thread runs it — still ~100x the Python recursion, and
 * the source stays one. */
extern "C" __global__ void scene_bvh_gpu(SLO_BVH_PARAMS) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    slo_bvh_build(verts, faces, n_faces, max_prims, out_min, out_max,
                  out_left, out_right, out_pstart, out_pcnt, perm, tmp, keys,
                  prim_out, out_n_nodes, out_n_prims);
}

#else

#ifdef _OPENMP
#include <omp.h>
#endif

/* CPU entries: the mesher fans out across cores; ctypes releases the GIL
 * for the call, so the team runs wide. */
extern void entry_mesh_cpu(SLO_MESH_PARAMS) {
    #pragma omp parallel for schedule(dynamic, 256)
    for (long i = 0; i < n; ++i) {
        slo_mesh_voxel(i, flat_idx[i], mat_idx[i], ny, nz, vs, verts, faces,
                       fmat);
    }
}

extern void entry_bvh_cpu(SLO_BVH_PARAMS) {
    slo_bvh_build(verts, faces, n_faces, max_prims, out_min, out_max, out_left,
                  out_right, out_pstart, out_pcnt, perm, tmp, keys, prim_out,
                  out_n_nodes, out_n_prims);
}

#endif
