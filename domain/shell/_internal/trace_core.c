/* trace_core.c — ONE path-tracing algorithm, TWO engines.
 *
 * Card 4de18e24. This file is the single-source answer to the
 * two-implementations liability: there is no numpy walk AND a CUDA
 * walk to keep in sync. There is THIS file, compiled twice by
 * domain/slolib/_internal/native.py:
 *
 *   CPU: gcc -O3 -fopenmp -> ctypes.  entry_trace_cpu: OpenMP
 *        parallel-for over paths. ctypes releases the GIL; OpenMP
 *        fans out across cores.
 *   GPU: NVRTC -> cupy RawKernel.     entry_trace_gpu: one thread
 *        per path.
 *
 * SLO_Q is the ONLY target-specific token (empty on CPU, __device__ on
 * GPU). Same math, same order, same RNG => parity is structural, not
 * tested. Both engines run one thread per path: the entire walk +
 * shade + bounce lives inside that thread, so there are ZERO host
 * dispatches per bounce and ZERO alive-ray compactions. The ~60us/call
 * dispatch floor that crippled the operator-by-operator CuPy walk does
 * not exist here, on either engine — which is what closes the
 * small-batch crossover on BOTH sides.
 *
 * Semantics mirror domain/shell/_internal/cycles.py exactly:
 *   - EPSILON = 1e-6, MAX_DEPTH = 8, RR_START = 3
 *   - Moller-Trumbore with the same EPSILON guard, same 1e30 miss
 *   - GGX D/G/F, lobe selection, PDF (trans=1/4pi), throughput/=pdf
 *   - miss -> background; emission -> radiance; RR with p=max(keep)
 * The numpy _trace_batch stays as the oracle AND the no-compiler
 * fallback; this file is the engine both devices now share.
 *
 * RNG: counter-based, keyed by (seed, path, depth, slot). No state is
 * carried between paths or bounces, so results are deterministic per
 * seed, identical across engines, and immune to thread/schedule order.
 */

/* SLO_Q is the single target-specific token: NVRTC defines __CUDACC__
 * (so every shared function becomes __device__); gcc defines neither
 * (so it expands to nothing). Everything else in this file is the same
 * instruction stream on both engines. */
#ifdef __CUDACC__
#define SLO_Q __device__
#else
#define SLO_Q
#endif

#define SLO_EPSILON 1e-6
#define SLO_PI 3.14159265358979323846
#define SLO_MISS 1e30f
#define SLO_MAX_STACK 64

/* ── counter RNG (shared by both engines; identical integer math) ────────── */

SLO_Q static unsigned int slo_hash(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

/* Uniform float in [0,1) from (seed, path, depth, slot). */
SLO_Q static float slo_rng(unsigned int seed, long path, int depth, int slot) {
    unsigned int h = slo_hash(seed);
    h = slo_hash(h ^ (unsigned int)path);
    h = slo_hash(h ^ (unsigned int)(depth * 7919 + slot));
    return (h & 0x00ffffffu) * (1.0f / 16777216.0f);
}

/* ── vector helpers (explicit components; no vec lib, no aliasing) ──────── */

SLO_Q static void slo_normalize3(float v[3]) {
    float n = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if (n > SLO_EPSILON) {
        float inv = 1.0f / sqrtf(n);
        v[0] *= inv; v[1] *= inv; v[2] *= inv;
    } else {
        v[0] = v[1] = v[2] = 0.0f;
    }
}

SLO_Q static float slo_dot3(const float a[3], const float b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

SLO_Q static void slo_cross3(const float a[3], const float b[3], float o[3]) {
    o[0] = a[1] * b[2] - a[2] * b[1];
    o[1] = a[2] * b[0] - a[0] * b[2];
    o[2] = a[0] * b[1] - a[1] * b[0];
}

/* ── Möller–Trumbore: IDENTICAL guard/order to BVH._mt_numpy ────────────── */

SLO_Q static float slo_mt(const float v0[3], const float e1[3], const float e2[3],
                          const float o[3], const float d[3]) {
    float h[3], q[3], s[3];
    slo_cross3(d, e2, h);
    float a = slo_dot3(e1, h);
    if (fabsf(a) <= SLO_EPSILON) return SLO_MISS;
    float f = 1.0f / a;
    s[0] = o[0] - v0[0]; s[1] = o[1] - v0[1]; s[2] = o[2] - v0[2];
    float u = f * slo_dot3(s, h);
    if (u < 0.0f || u > 1.0f) return SLO_MISS;
    slo_cross3(s, e1, q);
    float v = f * slo_dot3(d, q);
    if (v < 0.0f || u + v > 1.0f) return SLO_MISS;
    float t = f * slo_dot3(e2, q);
    return (t > SLO_EPSILON) ? t : SLO_MISS;
}

/* ── scene tables (flattened, engine-agnostic; caller packs pointers) ────── */

typedef struct {
    /* BVH node tables, flat across every mesh in the scene. */
    const float *nmin;      /* (n_nodes, 3) */
    const float *nmax;      /* (n_nodes, 3) */
    const int   *left;      /* (n_nodes,) */
    const int   *right;     /* (n_nodes,) */
    const int   *pcnt;      /* (n_nodes,) primitive_count */
    const int   *prims;     /* (n_nodes, K) padded face ids, -1 = pad */
    int          K;         /* prims per node row */
    /* triangle edge tables, flat across every mesh. */
    const float *tv0;       /* (n_tris, 3) */
    const float *te1;       /* (n_tris, 3) */
    const float *te2;       /* (n_tris, 3) */
    /* per-face vertex-normal interpolation (matches numpy exactly).
     * fnorm holds the THREE vertex normals per face laid out (F,9):
     * [n0x n0y n0z n1x n1y n1z n2x n2y n2z]. Barycentrics from the hit
     * point reproduce the smooth shading numpy produces, so normals
     * (and every downstream BSDF term) agree to float precision. */
    const float *fnorm;     /* (n_faces, 9) = 3 vertex normals */
    const int   *fmat;      /* (n_faces,) material index */
    /* materials, SoA. */
    const float *m_base;    /* (n_mats, 3) */
    const float *m_emis;    /* (n_mats, 3) emission_color*strength */
    const float *m_prop;    /* (n_mats, 4) trans, metal, ior, rough */
    /* multi-mesh offsets: mesh i's nodes start at moff_node[i],
     * its tris/faces at moff_face[i]. */
    const int   *moff_node; /* (n_meshes,) */
    const int   *moff_face; /* (n_meshes,) */
    int          n_meshes;
    float        bg[3];
} SloScene;

/* ── single-mesh BVH walk: one thread, own stack, closest hit ─────────────
 * Prune by entry t (safe: a node entering beyond the best hit so far
 * cannot hold a closer hit). Push right then left; order only affects
 * visitation, closest hit is kept regardless. */

SLO_Q static void slo_walk(const SloScene *sc, int mesh,
                           const float o[3], const float d[3],
                           float *out_t, int *out_face) {
    int base = sc->moff_node[mesh];
    int fbase = sc->moff_face[mesh];
    float inv[3];
    inv[0] = 1.0f / (d[0] + 1e-30f);
    inv[1] = 1.0f / (d[1] + 1e-30f);
    inv[2] = 1.0f / (d[2] + 1e-30f);

    int stack[SLO_MAX_STACK];
    int sp = 0;
    stack[sp++] = 0; /* local root */

    float best = SLO_MISS;
    int best_face = -1;

    while (sp > 0) {
        int ni = base + stack[--sp];
        /* slab test */
        float t1x = (sc->nmin[3 * ni]     - o[0]) * inv[0];
        float t1y = (sc->nmin[3 * ni + 1] - o[1]) * inv[1];
        float t1z = (sc->nmin[3 * ni + 2] - o[2]) * inv[2];
        float t2x = (sc->nmax[3 * ni]     - o[0]) * inv[0];
        float t2y = (sc->nmax[3 * ni + 1] - o[1]) * inv[1];
        float t2z = (sc->nmax[3 * ni + 2] - o[2]) * inv[2];
        float txmin = fminf(t1x, t2x), txmax = fmaxf(t1x, t2x);
        float tymin = fminf(t1y, t2y), tymax = fmaxf(t1y, t2y);
        float tzmin = fminf(t1z, t2z), tzmax = fmaxf(t1z, t2z);
        float tnear = fmaxf(fmaxf(txmin, tymin), tzmin);
        float tfar  = fminf(fminf(txmax, tymax), tzmax);
        if (tnear > tfar || tfar <= 0.0f || tnear > best) continue;

        if (sc->pcnt[ni] > 0) {
            const int *pr = sc->prims + (long)ni * sc->K;
            for (int j = 0; j < sc->K; ++j) {
                int fi = pr[j];
                if (fi < 0) break; /* -1 pad ends the row */
                long t3 = 3L * (fbase + fi);
                float t = slo_mt(sc->tv0 + t3, sc->te1 + t3, sc->te2 + t3, o, d);
                if (t < best) { best = t; best_face = fbase + fi; }
            }
        } else {
            int l = sc->left[ni], r = sc->right[ni];
            if (r >= 0 && sp < SLO_MAX_STACK) stack[sp++] = r;
            if (l >= 0 && sp < SLO_MAX_STACK) stack[sp++] = l;
        }
    }
    *out_t = best;
    *out_face = best_face;
}

/* ── GGX BSDF eval + lobe sampling (mirror the batch numpy helpers) ─────── */

SLO_Q static float slo_ggx_D(float ndoth, float alpha2) {
    float denom = ndoth * ndoth * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (SLO_PI * denom * denom + SLO_EPSILON);
}

SLO_Q static float slo_smith_g(float ndotv, float ndotl, float alpha2) {
    float sv = 2.0f * ndotv / (ndotv + sqrtf(alpha2 + (1.0f - alpha2) * ndotv * ndotv));
    float sl = 2.0f * ndotl / (ndotl + sqrtf(alpha2 + (1.0f - alpha2) * ndotl * ndotl));
    return sv * sl;
}

/* Evaluate the full BSDF (diffuse + spec*metal + (1-metal)*spec), same
 * masking as _eval_bsdf_batch. Returns rgb into out[3]. */
SLO_Q static void slo_bsdf_eval(const float wi[3], const float wo[3], const float N[3],
                                const float base[3], float metal, float rough,
                                float out[3]) {
    float diffuse[3] = { base[0] / (float)SLO_PI, base[1] / (float)SLO_PI, base[2] / (float)SLO_PI };
    float H[3] = { wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2] };
    slo_normalize3(H);
    float ndoth = fmaxf(slo_dot3(N, H), 0.0f);
    float ndotv = fmaxf(slo_dot3(N, wo), 0.0f);
    float ndotl = fmaxf(slo_dot3(N, wi), 0.0f);
    float alpha = rough * rough;
    float alpha2 = alpha * alpha;
    float D = slo_ggx_D(ndoth, alpha2);
    float G = slo_smith_g(ndotv, ndotl, alpha2);
    float f0r = (1.0f - metal) * 0.04f + metal * base[0];
    float f0g = (1.0f - metal) * 0.04f + metal * base[1];
    float f0b = (1.0f - metal) * 0.04f + metal * base[2];
    float ch5 = powf(1.0f - slo_dot3(H, wo), 5.0f);
    float Fr = f0r + (1.0f - f0r) * ch5;
    float Fg = f0g + (1.0f - f0g) * ch5;
    float Fb = f0b + (1.0f - f0b) * ch5;
    float spec = (D * G) / (4.0f * ndotv * ndotl + SLO_EPSILON);
    float kr = (1.0f - Fr) * (1.0f - metal);
    float kg = (1.0f - Fg) * (1.0f - metal);
    float kb = (1.0f - Fb) * (1.0f - metal);
    float cr = kr * diffuse[0] + spec * Fr * metal + (1.0f - metal) * spec * Fr;
    float cg = kg * diffuse[1] + spec * Fg * metal + (1.0f - metal) * spec * Fg;
    float cb = kb * diffuse[2] + spec * Fb * metal + (1.0f - metal) * spec * Fb;
    if (ndotl < SLO_EPSILON || ndotv < SLO_EPSILON) { cr = cg = cb = 0.0f; }
    out[0] = cr * ndotl; out[1] = cg * ndotl; out[2] = cb * ndotl;
}

/* PDF for the lobe actually sampled (0=trans,1=gloss,2=diffuse). */
SLO_Q static float slo_bsdf_pdf(const float wo[3], const float wi[3], const float N[3],
                                float rough, int lobe) {
    if (lobe == 0) return 1.0f / (4.0f * (float)SLO_PI);
    float ndotl = fmaxf(slo_dot3(N, wi), 0.0f);
    float pdf = ndotl / (float)SLO_PI; /* diffuse default */
    if (lobe == 1) {
        float H[3] = { wo[0] + wi[0], wo[1] + wi[1], wo[2] + wi[2] };
        slo_normalize3(H);
        float a2 = rough * rough; a2 = a2 * a2; /* rough^4 */
        float ndoth = fmaxf(slo_dot3(N, H), 0.0f);
        float D = slo_ggx_D(ndoth, a2);
        pdf = D * ndoth / (4.0f * fmaxf(slo_dot3(H, wo), 0.0f) + SLO_EPSILON);
    }
    return pdf;
}

/* ── GGX half-vector sample (alpha = rough^2, per _sample_ggx_batch) ─────── */

SLO_Q static void slo_sample_ggx(float r1, float r2, float alpha, float out[3]) {
    float a2 = alpha * alpha;
    float cos2 = (1.0f - r2) / (1.0f + (a2 - 1.0f) * r2);
    cos2 = fminf(fmaxf(cos2, 0.0f), 1.0f);
    float sin_ = sqrtf(fmaxf(1.0f - cos2, 0.0f));
    float phi = 2.0f * (float)SLO_PI * r1;
    out[0] = sin_ * cosf(phi);
    out[1] = sin_ * sinf(phi);
    out[2] = sqrtf(cos2);
}

/* ── tangent frame (up flips near |N.y|==0.999, like the batch helper) ──── */

SLO_Q static void slo_tangent_frame(const float N[3], float t[3], float b[3]) {
    float up[3];
    if (fabsf(N[1]) < 0.999f) { up[0] = 0.0f; up[1] = 1.0f; up[2] = 0.0f; }
    else                      { up[0] = 1.0f; up[1] = 0.0f; up[2] = 0.0f; }
    slo_cross3(up, N, t);
    slo_normalize3(t);
    slo_cross3(N, t, b);
}

/* ── one full path: walk + shade + bounce, entirely inside this thread ──── */

SLO_Q static void slo_trace_path(const SloScene *sc, unsigned int seed, long path,
                                 const float o0[3], const float d0[3],
                                 float max_depth, float rr_start, float out[3]) {
    float ro[3] = { o0[0], o0[1], o0[2] };
    float rd[3] = { d0[0], d0[1], d0[2] };
    float thr[3] = { 1.0f, 1.0f, 1.0f };
    float rad[3] = { 0.0f, 0.0f, 0.0f };
    int depth_max = (int)max_depth;
    int rr0 = (int)rr_start;

    for (int depth = 0; depth < depth_max; ++depth) {
        /* closest hit across every mesh */
        float bt = SLO_MISS;
        int bf = -1;
        for (int m = 0; m < sc->n_meshes; ++m) {
            float t; int fi;
            slo_walk(sc, m, ro, rd, &t, &fi);
            if (t < bt) { bt = t; bf = fi; }
        }

        if (bf < 0) { /* miss -> background */
            rad[0] += thr[0] * sc->bg[0];
            rad[1] += thr[1] * sc->bg[1];
            rad[2] += thr[2] * sc->bg[2];
            break;
        }

        /* hit point (numpy: orig + dire * t_hit) */
        float P[3] = { ro[0] + rd[0] * bt, ro[1] + rd[1] * bt, ro[2] + rd[2] * bt };
        int mat = sc->fmat[bf];
        const float *base = sc->m_base + 3L * mat;
        const float *emis = sc->m_emis + 3L * mat;
        const float *prop = sc->m_prop + 4L * mat;
        float trans = prop[0], metal = prop[1], ior = prop[2], rough = prop[3];

        /* barycentric-interpolated vertex normal — same formula as numpy
         * _intersect_scene: w=p-v0; d00..d21; denom=d00*d11-d01*d01;
         * guard on |denom|>=EPS; clip bary to [0,1]; fallback 1/3. */
        float n0[3], n1[3], n2[3], N[3];
        {
            const float *fn = sc->fnorm + 9L * bf;
            float w1, w2;
            if (bt < SLO_MISS) {
                const float *e1 = sc->te1 + 3L * bf;
                const float *e2 = sc->te2 + 3L * bf;
                const float *v0 = sc->tv0 + 3L * bf;
                float w[3] = { P[0] - v0[0], P[1] - v0[1], P[2] - v0[2] };
                float d00 = slo_dot3(e1, e1);
                float d01 = slo_dot3(e1, e2);
                float d11 = slo_dot3(e2, e2);
                float d20 = slo_dot3(w, e1);
                float d21 = slo_dot3(w, e2);
                float denom = d00 * d11 - d01 * d01;
                if (fabsf(denom) >= SLO_EPSILON) {
                    w1 = (d11 * d20 - d01 * d21) / denom;
                    w2 = (d00 * d21 - d01 * d20) / denom;
                } else {
                    w1 = w2 = 1.0f / 3.0f;
                }
                w1 = fminf(fmaxf(w1, 0.0f), 1.0f);
                w2 = fminf(fmaxf(w2, 0.0f), 1.0f);
            } else {
                w1 = w2 = 1.0f / 3.0f;
            }
            float b0 = 1.0f - w1 - w2;
            n0[0] = fn[0]; n0[1] = fn[1]; n0[2] = fn[2];
            n1[0] = fn[3]; n1[1] = fn[4]; n1[2] = fn[5];
            n2[0] = fn[6]; n2[1] = fn[7]; n2[2] = fn[8];
            N[0] = b0 * n0[0] + w1 * n1[0] + w2 * n2[0];
            N[1] = b0 * n0[1] + w1 * n1[1] + w2 * n2[1];
            N[2] = b0 * n0[2] + w1 * n1[2] + w2 * n2[2];
            slo_normalize3(N);
        }

        /* emission */
        rad[0] += thr[0] * emis[0];
        rad[1] += thr[1] * emis[1];
        rad[2] += thr[2] * emis[2];

        if (depth == depth_max - 1) break; /* last depth: emit only */

        float wo[3] = { -rd[0], -rd[1], -rd[2] };
        float r1 = slo_rng(seed, path, depth, 0);
        float r2 = slo_rng(seed, path, depth, 1);
        float r3 = slo_rng(seed, path, depth, 2);
        float r4 = slo_rng(seed, path, depth, 3);

        /* lobe select, same draw->lobe mapping as the batch path */
        int lobe = 2; /* diffuse */
        if (r3 < trans) lobe = 0;
        else if (r4 < metal) lobe = 1;

        float wi[3];
        if (lobe == 0) { /* refraction */
            float eta = (slo_dot3(N, wo) > 0.0f) ? (1.0f / ior) : ior;
            float I[3] = { -wo[0], -wo[1], -wo[2] };
            float cosi = -slo_dot3(I, N);
            float sin2t = eta * eta * (1.0f - cosi * cosi);
            float cost = sqrtf(fminf(fmaxf(1.0f - sin2t, 0.0f), 1.0f));
            wi[0] = eta * I[0] + (eta * cosi - cost) * N[0];
            wi[1] = eta * I[1] + (eta * cosi - cost) * N[1];
            wi[2] = eta * I[2] + (eta * cosi - cost) * N[2];
            slo_normalize3(wi);
        } else if (lobe == 1) { /* glossy */
            float h[3];
            slo_sample_ggx(r1, r2, rough * rough, h);
            float I[3] = { -wo[0], -wo[1], -wo[2] };
            float d2 = 2.0f * slo_dot3(I, h);
            wi[0] = I[0] - d2 * h[0];
            wi[1] = I[1] - d2 * h[1];
            wi[2] = I[2] - d2 * h[2];
        } else { /* diffuse: cosine-weighted hemisphere */
            float ang = 2.0f * (float)SLO_PI * r1;
            float s = sqrtf(r2);
            float local[3] = { cosf(ang) * s, sinf(ang) * s, sqrtf(1.0f - r2) };
            float t[3], b[3];
            slo_tangent_frame(N, t, b);
            wi[0] = local[0] * t[0] + local[1] * b[0] + local[2] * N[0];
            wi[1] = local[0] * t[1] + local[1] * b[1] + local[2] * N[1];
            wi[2] = local[0] * t[2] + local[1] * b[2] + local[2] * N[2];
        }
        if (slo_dot3(wi, N) < 0.0f) { wi[0] = -wi[0]; wi[1] = -wi[1]; wi[2] = -wi[2]; }

        /* bsdf / pdf; kill the path on a degenerate pdf */
        float bsdf[3];
        slo_bsdf_eval(wi, wo, N, base, metal, rough, bsdf);
        float pdf = slo_bsdf_pdf(wo, wi, N, rough, lobe);
        if (pdf <= SLO_EPSILON) break;

        thr[0] *= bsdf[0] / pdf;
        thr[1] *= bsdf[1] / pdf;
        thr[2] *= bsdf[2] / pdf;
        rd[0] = wi[0]; rd[1] = wi[1]; rd[2] = wi[2];
        ro[0] = P[0] + N[0] * SLO_EPSILON * 2.0f;
        ro[1] = P[1] + N[1] * SLO_EPSILON * 2.0f;
        ro[2] = P[2] + N[2] * SLO_EPSILON * 2.0f;

        /* russian roulette */
        if (depth >= rr0) {
            float p = fmaxf(fmaxf(thr[0], fmaxf(thr[1], thr[2])), 0.05f);
            float r5 = slo_rng(seed, path, depth, 4);
            if (r5 > p) break;
            thr[0] /= p; thr[1] /= p; thr[2] /= p;
        }
    }
    out[0] = rad[0]; out[1] = rad[1]; out[2] = rad[2];
}

/* ── entry parameter list (ONE definition, both engines) ──────────────────
 * Flattened to explicit pointers/scalars: a struct-of-pointers does not
 * marshal reliably across ctypes (by-value layout) and NVRTC (device
 * pointer params), so the shared list is a macro. Every arg is a plain
 * pointer or scalar both ABIs accept identically. */

#define SLO_TRACE_PARAMS \
    const float *nmin, const float *nmax, \
    const int *left, const int *right, const int *pcnt, const int *prims, int K, \
    const float *tv0, const float *te1, const float *te2, \
    const float *fnorm, const int *fmat, \
    const float *m_base, const float *m_emis, const float *m_prop, \
    const int *moff_node, const int *moff_face, int n_meshes, \
    const float *bg, \
    const float *orig, const float *dire, long n, \
    unsigned int seed, float max_depth, float rr_start, float *out

#define SLO_TRACE_FILL \
    SloScene sc; \
    sc.nmin = nmin; sc.nmax = nmax; \
    sc.left = left; sc.right = right; sc.pcnt = pcnt; sc.prims = prims; \
    sc.K = K; \
    sc.tv0 = tv0; sc.te1 = te1; sc.te2 = te2; \
    sc.fnorm = fnorm; sc.fmat = fmat; \
    sc.m_base = m_base; sc.m_emis = m_emis; sc.m_prop = m_prop; \
    sc.moff_node = moff_node; sc.moff_face = moff_face; \
    sc.n_meshes = n_meshes; \
    sc.bg[0] = bg[0]; sc.bg[1] = bg[1]; sc.bg[2] = bg[2];

/* ── entries: the ONLY place the two engines diverge ────────────────────── */

#ifdef __CUDACC__
extern "C" __global__ void trace_paths_gpu(SLO_TRACE_PARAMS) {
    SLO_TRACE_FILL
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float o[3], d[3], rgb[3];
    o[0] = orig[3 * i]; o[1] = orig[3 * i + 1]; o[2] = orig[3 * i + 2];
    d[0] = dire[3 * i]; d[1] = dire[3 * i + 1]; d[2] = dire[3 * i + 2];
    slo_trace_path(&sc, seed, i, o, d, max_depth, rr_start, rgb);
    out[3 * i] = rgb[0]; out[3 * i + 1] = rgb[1]; out[3 * i + 2] = rgb[2];
}
#else
#ifdef _OPENMP
#include <omp.h>
#endif
/* CPU entry: one OpenMP thread per path. ctypes releases the GIL for
 * this call, so the team runs wide across cores. */
extern void entry_trace_cpu(SLO_TRACE_PARAMS) {
    SLO_TRACE_FILL
    #pragma omp parallel for schedule(dynamic, 64)
    for (long i = 0; i < n; ++i) {
        float o[3], d[3], rgb[3];
        o[0] = orig[3 * i]; o[1] = orig[3 * i + 1]; o[2] = orig[3 * i + 2];
        d[0] = dire[3 * i]; d[1] = dire[3 * i + 1]; d[2] = dire[3 * i + 2];
        slo_trace_path(&sc, seed, i, o, d, max_depth, rr_start, rgb);
        out[3 * i] = rgb[0]; out[3 * i + 1] = rgb[1]; out[3 * i + 2] = rgb[2];
    }
}
#endif
