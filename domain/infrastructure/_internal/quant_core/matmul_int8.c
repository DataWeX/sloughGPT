/**
 * matmul_int8.c — int8 × int8 → int32 GEMM accelerated with AVX2/AVX-512.
 *
 * Computes  C[i,j] = Σₖ A[i,k] · B[j,k]
 *
 * Layout:
 *   A:  M×K  int8_t (row-major)
 *   B:  N×K  int8_t (row-major)
 *   C:  M×N  int32_t (row-major, zeroed then written)
 *
 * Cache blocking: the B matrix is processed in column blocks (j-blocks) of at
 * most ~256KB. Within a block the rows of A (M×K, small) stay resident in
 * L1/L2 and are reused across every B row in the block, so B is streamed from
 * DRAM exactly once regardless of M. Without this, an M×N×K GEMM re-streams
 * B from DRAM M times (e.g. M=128 makes a 590KB matrix cost 75MB of traffic).
 *
 * Two SIMD kernels back the GEMM: AVX2 (_dot16) and AVX-512 BW + VNNI
 * (_mm512_dpbusd_epi32, the fused int8 dot-product llama.cpp uses. The +128
 * activation bias makes the first operand unsigned so VNNI computes the signed
 * product with a one-time O(1) correction against precomputed per-row weight
 * sums). Neither wins at every shape, so the runtime picks per call:
 * AVX2 wins single-token decode (M=1, latency-bound — measured 1.31–1.84× on
 * this host) while AVX-512 wins batched prefill (M≥8, throughput-bound, 64
 * MAC/instruction vs AVX2's 32). The kernel for the next GEMM is chosen by
 * matmul_int8_select_kernel(); the Python wrapper routes by batch size from a
 * host-calibrated table (see wrapper.py "smart per-shape kernel dispatch").
 * Kernel selection: matmul_int8_select_kernel(k) with
 *   * 0 = hardware default (AVX-512 when CPUID says it is supported)
 *   * 1 = force AVX2
 *   * 2 = force AVX-512 VNNI
 * MAN_QUANT_KERNEL=2|512|auto still selects at load time and is honoured as
 * the "auto" default when no explicit per-call selection is made. The library
 * is compiled with AVX-512 flags when the toolchain allows, so the binary is
 * portable across CPU generations (AVX-512 path is runtime-gated by CPUID).
 *
 * Threading: for large GEMMs (B bytes ≥ _THREAD_MIN_BYTES) the j-block loop is
 * spread across threads. B is split into contiguous j-block slices, each thread
 * streams its own slice (disjoint C columns), so the result is bit-identical to
 * the single-threaded path. Thread count: MAN_GEMM_THREADS env var, else online
 * CPU count (max 64). pthread_create failure degrades to running inline.
 *
 * Build (AVX-512 + VNNI when supported, else AVX2-only):
 *   gcc -O3 -mavx2 -mavx512f -mavx512bw -mavx512vnni -fno-tree-vectorize \
 *       -shared -fPIC -pthread -o matmul_int8.so matmul_int8.c
 */

#include <immintrin.h>
#include <math.h>
#include <cpuid.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <pthread.h>
#include <unistd.h>

/** Target size of a B j-block in bytes — keep A + B-block in L2. */
#define _JB_TARGET_BYTES 262144L  /* 256KB */
#define _JB_MIN 32
#define _JB_MAX 4096

/** Minimum B bytes (N×K) before the j-block loop is spread across threads.
 *  Below this the thread-spawn cost (~0.1-0.2ms on 8 threads) exceeds the
 *  bandwidth win (measured neutral at 4.2MB, 1.5x at 8.3MB). */
#define _THREAD_MIN_BYTES 6291456L  /* 6MB */

static int _gemm_threads(void) {
    /* Serial by default. j-block threading only pays off when the weight
     * matrix is streamed from DRAM by real cores; on this 4C/8T i5-9300H the
     * isolated big-GEMM gains (up to 2.5x on lm_head) do NOT transfer to real
     * decode: warm weights are already near DRAM bandwidth, hyperthreads share
     * ports/LLC and add memory-controller contention, and spawning threads
     * also stalls the numpy attention ops and heats the chip into throttle
     * (measured 59ms vs 124-130ms bimodal at 4 threads vs stable 54-67ms
     * serial). Set MAN_GEMM_THREADS=N (1..256) to enable per-call. */
    const char *e = getenv("MAN_GEMM_THREADS");
    if (e == NULL) return 1;
    long t = strtol(e, NULL, 10);
    if (t <= 0 || t > 256) return 1;
    return (int)t;
}

/* ── AVX2 implementation ────────────────────────────────────────── */

#if defined(__AVX2__)

/** Horizontal sum of 8 int32 lanes → single int32. */
static inline int32_t _hsum8(__m256i v) {
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i s  = _mm_add_epi32(lo, hi);          /* 4 int32 */
    s = _mm_hadd_epi32(s, s);                    /* 2 int32 */
    s = _mm_hadd_epi32(s, s);                    /* 1 int32 */
    return _mm_cvtsi128_si32(s);
}

/**
 * Dot-product 16 int8 pairs → 8 int32 partial sums.
 *
 * Widen both operands to int16 and let madd_epi16 pair the products into int32
 * in one instruction — one fewer op per 16 bytes than the old widen+mullo+madd
 * chain. Products stay within int16 range (127·127 = 16129), so madd is exact
 * and needs no sign-bias correction: the AVX2 path stays pure signed int8.
 */
static inline __m256i _dot16(const __m128i a, const __m128i b) {
    return _mm256_madd_epi16(_mm256_cvtepi8_epi16(a), _mm256_cvtepi8_epi16(b));
}

/** Dot product of two length-K int8 rows → int32. */
static inline int32_t _dot_row_avx2(const int8_t *a, const int8_t *b, int K) {
    __m256i s0 = _mm256_setzero_si256();
    __m256i s1 = _mm256_setzero_si256();
    __m256i s2 = _mm256_setzero_si256();
    __m256i s3 = _mm256_setzero_si256();
    int k = 0;

    /* 64 bytes/iteration across 4 independent accumulators so the madd chains
     * interleave instead of serializing onto one add dependency (hides the
     * ~5-cycle madd latency and keeps both 256-bit ALU ports busy). */
    for (; k + 64 <= K; k += 64) {
        s0 = _mm256_add_epi32(s0, _dot16(_mm_loadu_si128((const __m128i *)(a + k)),
                                          _mm_loadu_si128((const __m128i *)(b + k))));
        s1 = _mm256_add_epi32(s1, _dot16(_mm_loadu_si128((const __m128i *)(a + k + 16)),
                                          _mm_loadu_si128((const __m128i *)(b + k + 16))));
        s2 = _mm256_add_epi32(s2, _dot16(_mm_loadu_si128((const __m128i *)(a + k + 32)),
                                          _mm_loadu_si128((const __m128i *)(b + k + 32))));
        s3 = _mm256_add_epi32(s3, _dot16(_mm_loadu_si128((const __m128i *)(a + k + 48)),
                                          _mm_loadu_si128((const __m128i *)(b + k + 48))));
    }
    /* Remaining 16-element block. */
    for (; k + 16 <= K; k += 16) {
        s0 = _mm256_add_epi32(s0, _dot16(_mm_loadu_si128((const __m128i *)(a + k)),
                                          _mm_loadu_si128((const __m128i *)(b + k))));
    }

    __m256i acc = _mm256_add_epi32(_mm256_add_epi32(s0, s1), _mm256_add_epi32(s2, s3));
    int32_t total = _hsum8(acc);

    /* Scalar remainder (< 16 elements) */
    for (; k < K; k++) {
        total += (int32_t)a[k] * (int32_t)b[k];
    }
    return total;
}

/** Compute one B j-block for all M rows of A (B block reused from cache).
 *  A 4×4 register-blocked variant was benchmarked and measured within ±2% of
 *  this form at every shape — the loop is throughput-bound, so reload reuse
 *  has no headroom; keep the simple version. */
static inline void _block_j_avx2(const int8_t *A, const int8_t *B,
                                 int32_t *C, int M, int N, int K,
                                 int jb, int j_end) {
    for (int i = 0; i < M; i++) {
        const int8_t *a_row = A + (size_t)i * K;
        int32_t       *c_row = C + (size_t)i * N;
        for (int j = jb; j < j_end; j++) {
            c_row[j] = _dot_row_avx2(a_row, B + (size_t)j * K, K);
        }
    }
}

/* ── AVX-512 VNNI implementation ────────────────────────────────────
 *
 * Uses _mm512_dpbusd_epi32 (VNNI): a single fused multiply-accumulate that
 * consumes 64 int8 operands per 512-bit lane and produces 16 int32 partial
 * sums — the exact kernel llama.cpp uses for int8 GEMM. The ZMM register
 * doubles the per-instruction throughput of the AVX2 YMM path.
 *
 * _mm512_dpbusd_epi32 treats operand (a) as unsigned and (b) as signed, so
 * for signed activations we compensate: a_s8 = a_u8 - 256·sign(a). We sum the
 * signed b entries where a is negative and subtract 256× that sum.
 */

#if defined(__AVX512BW__) && defined(__AVX512VNNI__)

static int _use_avx512(void);  /* forward decl */

/** Horizontal sum of all 16 int32 lanes → single int32. */
static inline int32_t _hsum16(__m512i v) {
    __m512i s = _mm512_add_epi32(v, _mm512_permutexvar_epi32(
        _mm512_setr_epi32(8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7), v));
    s = _mm512_add_epi32(s, _mm512_permutexvar_epi32(
        _mm512_setr_epi32(4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7), s));
    s = _mm512_add_epi32(s, _mm512_permutexvar_epi32(
        _mm512_setr_epi32(2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3), s));
    s = _mm512_add_epi32(s, _mm512_permutexvar_epi32(
        _mm512_setr_epi32(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1), s));
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(s));
}

/** Dot product of two length-K int8 rows → int32 (AVX-512 VNNI). */
static inline int32_t _dot_row_avx512(const int8_t *a, const int8_t *b,
                                      int32_t b_sum, int K) {
    /* dpbusd(a+128, b) = Σ (a_i+128)·b_i = Σ a_i·b_i + 128·Σ b_i.
       a+128 makes the activation unsigned (dpbusd's first operand is u8);
       the 128·Σb_i term is the precomputed weight row sum, subtracted once
       below — O(1) per output instead of an in-loop sign correction. */
    const __m512i plus128 = _mm512_set1_epi8(128);
    __m512i acc = _mm512_setzero_si512();
    int k = 0;
    for (; k + 64 <= K; k += 64) {
        __m512i va = _mm512_loadu_si512((const __m512i *)(a + k));
        __m512i vb = _mm512_loadu_si512((const __m512i *)(b + k));
        acc = _mm512_dpbusd_epi32(acc, _mm512_add_epi8(va, plus128), vb);
    }
    int32_t total = _hsum16(acc);
    for (; k < K; k++) {
        total += ((int32_t)a[k] + 128) * (int32_t)b[k];
    }
    return total - 128 * b_sum;
}

/** Sum of a signed int8 row → int32 (vectorized with VNNI). */
static inline int32_t _rowsum_i8(const int8_t *row, int K) {
#if defined(__AVX512BW__) && defined(__AVX512VNNI__)
    if (_use_avx512()) {
        const __m512i ones = _mm512_set1_epi8(1);
        __m512i acc = _mm512_setzero_si512();
        int k = 0;
        for (; k + 64 <= K; k += 64) {
            acc = _mm512_dpbusd_epi32(acc, ones,
                _mm512_loadu_si512((const __m512i *)(row + k)));
        }
        int32_t s = _hsum16(acc);
        for (; k < K; k++) s += (int32_t)row[k];
        return s;
    }
#endif
    int32_t s = 0;
    for (int k = 0; k < K; k++) s += (int32_t)row[k];
    return s;
}

#endif /* AVX512BW && AVX512VNNI */

/** Runtime CPU/vendor check for AVX-512 BW + VNNI. */
static int _has_avx512(void) {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return 0;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return 0;
    int avx512f   = (ebx >> 16) & 1;
    int avx512bw  = (ebx >> 30) & 1;
    int avx512vnni= (ecx >> 11) & 1;
    return avx512f && avx512bw && avx512vnni;
}

/** Cache whether the AVX-512 path is compiled IN and the CPU supports it. */
static int _use_avx512(void) {
    static int cached = -1;
    if (cached < 0) {
#if defined(__AVX512BW__) && defined(__AVX512VNNI__)
        cached = _has_avx512();
#else
        cached = 0;
#endif
    }
    /* Force a specific kernel for A/B benchmarking and deployment choice:
     * MAN_QUANT_KERNEL=512 → AVX-512, =2 → AVX2, =auto → hardware default.
     * The forced value also propagates through quant_core_has_avx512() so the
     * Python adaptive-crossover logic follows whatever kernel actually runs.
     * "2" only disables a path that was compiled in; on a non-AVX512 build the
     * AVX-512 code does not exist and _use_avx512() stays 0 regardless. */
    const char *e = getenv("MAN_QUANT_KERNEL");
    if (e != NULL) {
        if (strcmp(e, "512") == 0) return (cached > 0) ? 1 : 0;
        if (strcmp(e, "2") == 0) return 0;
        /* "auto" (and any unknown value) → hardware default. */
    }
    return cached;
}

/** Runtime-selected int8 kernel for the next GEMM: 0 = auto (hardware default,
 *  with MAN_QUANT_KERNEL respected), 1 = force AVX2, 2 = force AVX-512 VNNI.
 *  The Python smart dispatcher calibrates per shape on the host and sets this
 *  between calls, so a single binary serves both M=1 decode (AVX2 wins) and
 *  M≥8 prefill (AVX-512 wins) — the right kernel for each shape, chosen by
 *  measurement instead of one fixed bootstrap decision. */
static int _forced_kernel = 0;

void matmul_int8_select_kernel(int which) { _forced_kernel = which; }

int matmul_int8_kernel(void) {
#if defined(__AVX2__)
    if (_forced_kernel == 1) return 1;
    if (_forced_kernel == 2) return 2;
    return _use_avx512() ? 2 : 1;
#else
    return 0;
#endif
}

/** Dispatcher: AVX-512 VNNI when selected, else AVX2. b_sum feeds the VNNI
 *  bias correction and is ignored on the AVX2 path. */
static inline int32_t _dot_row(const int8_t *a, const int8_t *b,
                               int32_t b_sum, int K) {
#if defined(__AVX512BW__) && defined(__AVX512VNNI__)
    if (_forced_kernel == 2) return _dot_row_avx512(a, b, b_sum, K);
    if (_forced_kernel == 1) { (void)b_sum; return _dot_row_avx2(a, b, K); }
    if (_use_avx512()) return _dot_row_avx512(a, b, b_sum, K);
#endif
    (void)b_sum;
    return _dot_row_avx2(a, b, K);
}

/** Compute one B j-block for all M rows of A using the best available kernel. */
static inline void _block_j(const int8_t *A, const int8_t *B,
                            const int32_t *B_rowsum,
                            int32_t *C, int M, int N, int K,
                            int jb, int j_end) {
    for (int i = 0; i < M; i++) {
        const int8_t *a_row = A + (size_t)i * K;
        int32_t       *c_row = C + (size_t)i * N;
        for (int j = jb; j < j_end; j++) {
            int32_t bs = B_rowsum ? B_rowsum[j] : 0;
            c_row[j] = _dot_row(a_row, B + (size_t)j * K, bs, K);
        }
    }
}

/* ── Thread pool helpers ────────────────────────────────────────────
 *
 * The B matrix is split into contiguous j-block slices, one per thread.
 * Each slice writes a disjoint column range of C/Acc, so the result is
 * bit-identical to the single-threaded path regardless of schedule.
 * pthread_create failure degrades gracefully: the failed slice runs on the
 * caller thread inline.
 */

typedef struct {
    const int8_t *A;
    const int8_t *B;
    const int32_t *B_rowsum;
    int32_t *C;
    int M, N, K;
    int jb_start;   /* first j-block index */
    int jb_stop;    /* one past last j-block index (in j-block units) */
    int jblock;
} gemm_worker_arg;

static void *_gemm_worker(void *p) {
    gemm_worker_arg *a = p;
    int stop = a->jb_stop * a->jblock;
    if (stop > a->N) stop = a->N;
    for (int jb = a->jb_start * a->jblock; jb < stop; jb += a->jblock) {
        int j_end = (jb + a->jblock < a->N) ? jb + a->jblock : a->N;
        _block_j(a->A, a->B, a->B_rowsum, a->C, a->M, a->N, a->K, jb, j_end);
    }
    return NULL;
}

static void _gemm_run_threads(const int8_t *A, const int8_t *B,
                              const int32_t *B_rowsum, int32_t *C,
                              int M, int N, int K, int nblk, int jblock) {
    int nthreads = _gemm_threads();
    if (nthreads > nblk) nthreads = nblk;
    if (nthreads < 1) nthreads = 1;
    if (nthreads <= 1) {
        for (int jb = 0; jb < N; jb += jblock) {
            int j_end = (jb + jblock < N) ? jb + jblock : N;
            _block_j(A, B, B_rowsum, C, M, N, K, jb, j_end);
        }
        return;
    }
    pthread_t *th = (pthread_t *)malloc((size_t)nthreads * sizeof(pthread_t));
    gemm_worker_arg *args = (gemm_worker_arg *)malloc(
        (size_t)nthreads * sizeof(gemm_worker_arg));
    if (th == NULL || args == NULL) {
        free(th); free(args);
        for (int jb = 0; jb < N; jb += jblock) {
            int j_end = (jb + jblock < N) ? jb + jblock : N;
            _block_j(A, B, B_rowsum, C, M, N, K, jb, j_end);
        }
        return;
    }
    int spawned = 0;
    for (int t = 0; t < nthreads; t++) {
        args[t] = (gemm_worker_arg){
            A, B, B_rowsum, C, M, N, K,
            (int)((long)t * nblk / nthreads),
            (int)((long)(t + 1) * nblk / nthreads),
            jblock,
        };
        if (pthread_create(&th[spawned], NULL, _gemm_worker, &args[t]) == 0)
            spawned++;
        else
            _gemm_worker(&args[t]);
    }
    for (int t = 0; t < spawned; t++) pthread_join(th[t], NULL);
    free(th); free(args);
}

#endif /* __AVX2__ */

/* ── Fused float→int8 quantize + GEMM + dequantize + bias ────────────
 *
 * Single pass over A that mirrors the Python hot path of
 * quantized_linear() (per-token symmetric W8A8):
 *
 *   scale_i   = max(|A[i,:]|) / 127   (or 1.0 for an all-zero row)
 *   Aq[i,k]   = clip(round(A[i,k] / scale_i), -128, 127)   (ties-to-even)
 *   Acc[i,j]  = Σₖ Aq[i,k] · B[j,k]                          (int32 GEMM)
 *   C[i,j]    = Acc[i,j] · scale_i · B_scale[j] + bias[j]
 *
 * Rounding matches numpy: round-to-nearest-even for half-way values, so the
 * quantized activations are bit-identical to quantize_activation() and the
 * float32 output matches int8_matmul()'s dequantize within 1 ULP.
 *
 * B_scale is a pointer to N floats (per-row) when b_scale_per_row is set,
 * otherwise to a single float used for every output column. bias may be NULL.
 * On failure to allocate scratch memory the output is left undefined and the
 * call returns; the Python wrapper falls back to the unfused path first.
 * ───────────────────────────────────────────────────────────────────── */

static inline float _quantize_row_f32_scalar(const float *a, int8_t *aq, int K) {
    float row_max = 0.0f;
    for (int k = 0; k < K; k++) {
        float f = fabsf(a[k]);
        if (f > row_max) row_max = f;
    }
    float scale = (row_max > 0.0f) ? row_max / 127.0f : 1.0f;
    for (int k = 0; k < K; k++) {
        float q = rintf(a[k] / scale);
        if (q > 127.0f) q = 127.0f;
        else if (q < -128.0f) q = -128.0f;
        aq[k] = (int8_t)q;
    }
    return scale;
}

#if defined(__AVX2__)

/** AVX2 per-row quantize: max|A|/127 scale + vectorized round/clamp→int8. */
static inline float _quantize_row_f32(const float *a, int8_t *aq, int K) {
    __m256 maxv = _mm256_setzero_ps();
    int k = 0;
    for (; k + 8 <= K; k += 8) {
        __m256 v = _mm256_loadu_ps(a + k);
        maxv = _mm256_max_ps(maxv, _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v));
    }
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(maxv), _mm256_extractf128_ps(maxv, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 1, 1, 1)));
    float row_max = _mm_cvtss_f32(m);
    for (; k < K; k++) {
        float f = fabsf(a[k]);
        if (f > row_max) row_max = f;
    }
    float scale = (row_max > 0.0f) ? row_max / 127.0f : 1.0f;

    __m256 s = _mm256_set1_ps(scale);
    const __m256 lo = _mm256_set1_ps(-128.0f);
    const __m256 hi = _mm256_set1_ps(127.0f);
    k = 0;
    /* 16 elements per iteration. Two packs of (lo4,hi4) reorder each 8-int32
     * group back to natural order: packs_epi32(lo, hi) yields
     * [a0..a3, a4..a7], so joining two groups with packs_epi16 keeps the
     * byte order contiguous — _mm256_packs_epi32 on four separate 8-vectors
     * interleaves the low 4-lanes and scrambles the output. */
    for (; k + 16 <= K; k += 16) {
        __m256 r0 = _mm256_div_ps(_mm256_loadu_ps(a + k), s);
        __m256 r1 = _mm256_div_ps(_mm256_loadu_ps(a + k + 8), s);
        __m256i v0 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(r0, lo), hi));
        __m256i v1 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(r1, lo), hi));
        __m128i p0 = _mm_packs_epi32(_mm256_castsi256_si128(v0), _mm256_extracti128_si256(v0, 1));
        __m128i p1 = _mm_packs_epi32(_mm256_castsi256_si128(v1), _mm256_extracti128_si256(v1, 1));
        _mm_storeu_si128((__m128i *)(aq + k), _mm_packs_epi16(p0, p1));
    }
    for (; k < K; k++) {
        float q = rintf(a[k] / scale);
        if (q > 127.0f) q = 127.0f;
        else if (q < -128.0f) q = -128.0f;
        aq[k] = (int8_t)q;
    }
    return scale;
}

#endif /* __AVX2__ */

static inline void _dequantize_f32_range(float *C, const int32_t *Acc,
                                         const float *A_scale,
                                         const float *B_scale,
                                         const float *bias, int M, int N,
                                         int b_scale_per_row,
                                         int j_start, int j_end) {
#if defined(__AVX2__)
    for (int i = 0; i < M; i++) {
        float as = A_scale[i];
        const int32_t *arow = Acc + (size_t)i * N;
        float         *crow = C + (size_t)i * N;
        int j = j_start;
        if (b_scale_per_row) {
            __m256 asv = _mm256_set1_ps(as);
            for (; j + 8 <= j_end; j += 8) {
                __m256 a = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)(arow + j)));
                __m256 o = _mm256_mul_ps(a, _mm256_mul_ps(asv, _mm256_loadu_ps(B_scale + j)));
                if (bias != NULL) o = _mm256_add_ps(o, _mm256_loadu_ps(bias + j));
                _mm256_storeu_ps(crow + j, o);
            }
        } else {
            __m256 s = _mm256_set1_ps(as * B_scale[0]);
            for (; j + 8 <= j_end; j += 8) {
                __m256 a = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)(arow + j)));
                __m256 o = _mm256_mul_ps(a, s);
                if (bias != NULL) o = _mm256_add_ps(o, _mm256_loadu_ps(bias + j));
                _mm256_storeu_ps(crow + j, o);
            }
        }
        for (; j < j_end; j++) {
            float bs = b_scale_per_row ? B_scale[j] : B_scale[0];
            float out = (float)arow[j] * (as * bs);
            if (bias != NULL) out += bias[j];
            crow[j] = out;
        }
    }
#else
    for (int i = 0; i < M; i++) {
        float as = A_scale[i];
        const int32_t *arow = Acc + (size_t)i * N;
        float         *crow = C + (size_t)i * N;
        for (int j = j_start; j < j_end; j++) {
            float bs = b_scale_per_row ? B_scale[j] : B_scale[0];
            float out = (float)arow[j] * (as * bs);
            if (bias != NULL) out += bias[j];
            crow[j] = out;
        }
    }
#endif
}

static inline void _dequantize_f32(float *C, const int32_t *Acc,
                                   const float *A_scale, const float *B_scale,
                                   const float *bias, int M, int N,
                                   int b_scale_per_row) {
    _dequantize_f32_range(C, Acc, A_scale, B_scale, bias, M, N,
                          b_scale_per_row, 0, N);
}

/** Allocate and fill per-row signed sums of B (needed by the AVX-512 path). */
#if defined(__AVX512BW__) && defined(__AVX512VNNI__)
static int32_t *_make_rowsum(const int8_t *B, int N, int K) {
    if (!_use_avx512()) return NULL;
    int32_t *rs = (int32_t *)malloc((size_t)N * sizeof(int32_t));
    if (rs == NULL) return NULL;
    for (int j = 0; j < N; j++) {
        rs[j] = _rowsum_i8(B + (size_t)j * K, K);
    }
    return rs;
}
#endif

/** Row sums are always NULL when the AVX-512 path is not compiled in. */
#if !defined(__AVX512BW__) || !defined(__AVX512VNNI__)
#define _make_rowsum(B, N, K) NULL
#endif

void matmul_int8_f32(const float *A, const int8_t *B, const float *B_scale,
                     const float *bias, float *C,
                     int M, int N, int K, int b_scale_per_row) {
    if (M <= 0 || N <= 0 || K <= 0) return;

    int8_t  *Aq     = (int8_t *)malloc((size_t)M * K);
    int32_t *Acc    = (int32_t *)malloc((size_t)M * N * sizeof(int32_t));
    float   *Ascale = (float *)malloc((size_t)M * sizeof(float));
    if (Aq == NULL || Acc == NULL || Ascale == NULL) {
        free(Aq); free(Acc); free(Ascale);
        return;
    }

    for (int i = 0; i < M; i++) {
#if defined(__AVX2__)
        Ascale[i] = _quantize_row_f32(A + (size_t)i * K, Aq + (size_t)i * K, K);
#else
        Ascale[i] = _quantize_row_f32_scalar(A + (size_t)i * K, Aq + (size_t)i * K, K);
#endif
    }
    memset(Acc, 0, (size_t)M * N * sizeof(int32_t));

    long jblock = _JB_TARGET_BYTES / (long)K;
    if (jblock < _JB_MIN) jblock = _JB_MIN;
    if (jblock > _JB_MAX) jblock = _JB_MAX;

    int32_t *B_rowsum = _make_rowsum(B, N, K);

#if defined(__AVX2__)
    long total_bytes = (long)N * K;
    if (total_bytes >= _THREAD_MIN_BYTES) {
        int nblk = (int)(((long)N + jblock - 1) / jblock);
        _gemm_run_threads(Aq, B, B_rowsum, Acc, M, N, K, nblk, (int)jblock);
    } else {
        for (int jb = 0; jb < N; jb += (int)jblock) {
            int j_end = (int)((long)jb + jblock < N ? jb + jblock : N);
            _block_j(Aq, B, B_rowsum, Acc, M, N, K, jb, j_end);
        }
    }
#else
    for (int jb = 0; jb < N; jb += (int)jblock) {
        int j_end = (int)((long)jb + jblock < N ? jb + jblock : N);
        for (int i = 0; i < M; i++) {
            const int8_t *a_row = Aq + (size_t)i * K;
            int32_t       *c_row = Acc + (size_t)i * N;
            for (int j = jb; j < j_end; j++) {
                const int8_t *b_row = B + (size_t)j * K;
                int32_t total = 0;
                for (int k = 0; k < K; k++) {
                    total += (int32_t)a_row[k] * (int32_t)b_row[k];
                }
                c_row[j] = total;
            }
        }
    }
#endif

    _dequantize_f32(C, Acc, Ascale, B_scale, bias, M, N, b_scale_per_row);

    free(B_rowsum); free(Aq); free(Acc); free(Ascale);
}

/* ── Public API ─────────────────────────────────────────────────── */

void matmul_int8(const int8_t *A, const int8_t *B, int32_t *C,
                 int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    memset(C, 0, (size_t)M * N * sizeof(int32_t));

    /* B j-block width: keep the block (JB×K bytes) plus A (M×K) in L2. */
    long jblock = _JB_TARGET_BYTES / (long)K;
    if (jblock < _JB_MIN) jblock = _JB_MIN;
    if (jblock > _JB_MAX) jblock = _JB_MAX;

    int32_t *B_rowsum = _make_rowsum(B, N, K);

#if defined(__AVX2__)
    long total_bytes = (long)N * K;
    if (total_bytes >= _THREAD_MIN_BYTES) {
        int nblk = (int)(((long)N + jblock - 1) / jblock);
        _gemm_run_threads(A, B, B_rowsum, C, M, N, K, nblk, (int)jblock);
    } else {
        for (int jb = 0; jb < N; jb += (int)jblock) {
            int j_end = (int)((long)jb + jblock < N ? jb + jblock : N);
            _block_j(A, B, B_rowsum, C, M, N, K, jb, j_end);
        }
    }
#else
    for (int jb = 0; jb < N; jb += (int)jblock) {
        int j_end = (int)((long)jb + jblock < N ? jb + jblock : N);
        for (int i = 0; i < M; i++) {
            const int8_t *a_row = A + (size_t)i * K;
            int32_t       *c_row = C + (size_t)i * N;
            for (int j = jb; j < j_end; j++) {
                const int8_t *b_row = B + (size_t)j * K;
                int32_t total = 0;
                for (int k = 0; k < K; k++) {
                    total += (int32_t)a_row[k] * (int32_t)b_row[k];
                }
                c_row[j] = total;
            }
        }
    }
#endif
    free(B_rowsum);
}

/* Exported capability probe: 1 when the AVX-512 VNNI path is compiled in and
 * the running CPU supports it. Lets the Python layer lower the adaptive
 * quantization crossover for hosts that have the faster kernel. */
#if defined(__AVX2__)
int quant_core_has_avx512(void) { return _use_avx512(); }
#endif
