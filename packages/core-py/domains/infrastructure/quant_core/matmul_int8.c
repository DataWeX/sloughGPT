/**
 * matmul_int8.c — int8 × int8 → int32 GEMM accelerated with AVX2.
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
 * On CPUs without AVX2 a scalar fallback is used.
 *
 * Threading: for large GEMMs (B bytes ≥ _THREAD_MIN_BYTES) the j-block loop is
 * spread across threads. B is split into contiguous j-block slices, each thread
 * streams its own slice (disjoint C columns), so the result is bit-identical to
 * the single-threaded path. Thread count: MAN_GEMM_THREADS env var, else online
 * CPU count (max 64). pthread_create failure degrades to running inline.
 *
 * Build:
 *   gcc -O3 -mavx2 -shared -fPIC -pthread -o matmul_int8.so matmul_int8.c
 */

#include <immintrin.h>
#include <math.h>
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
 * Process 32 int8 values from a and b → 8 int32 partial sums.
 *
 * Strategy: widen 16× int8 to 16× int16 (two 128-bit halves),
 * multiply, then madd_epi16 pairs the int16 values into int32.
 */
static inline __m256i _dot32(const int8_t *a, const int8_t *b) {
    /* Load both 16-byte halves */
    __m128i a_lo = _mm_loadu_si128((const __m128i *)a);
    __m128i a_hi = _mm_loadu_si128((const __m128i *)(a + 16));
    __m128i b_lo = _mm_loadu_si128((const __m128i *)b);
    __m128i b_hi = _mm_loadu_si128((const __m128i *)(b + 16));

    /* Widen each half int8→int16 */
    __m256i a_w_lo = _mm256_cvtepi8_epi16(a_lo);
    __m256i a_w_hi = _mm256_cvtepi8_epi16(a_hi);
    __m256i b_w_lo = _mm256_cvtepi8_epi16(b_lo);
    __m256i b_w_hi = _mm256_cvtepi8_epi16(b_hi);

    /* Multiply int16 (low halves — safe for 127×127 = 16129 < 2¹⁵) */
    __m256i p_lo = _mm256_mullo_epi16(a_w_lo, b_w_lo);
    __m256i p_hi = _mm256_mullo_epi16(a_w_hi, b_w_hi);

    /* Pairwise int16 → int32 */
    __m256i s_lo = _mm256_madd_epi16(p_lo, _mm256_set1_epi16(1));
    __m256i s_hi = _mm256_madd_epi16(p_hi, _mm256_set1_epi16(1));

    return _mm256_add_epi32(s_lo, s_hi);
}

/** Dot product of two length-K int8 rows → int32. */
static inline int32_t _dot_row_avx2(const int8_t *a, const int8_t *b, int K) {
    __m256i acc = _mm256_setzero_si256();
    int k = 0;

    /* Process 64 elements per iteration */
    for (; k + 64 <= K; k += 64) {
        acc = _mm256_add_epi32(acc, _dot32(a + k, b + k));
        acc = _mm256_add_epi32(acc, _dot32(a + k + 32, b + k + 32));
    }
    /* Process remaining 32-element block */
    for (; k + 32 <= K; k += 32) {
        acc = _mm256_add_epi32(acc, _dot32(a + k, b + k));
    }

    int32_t total = _hsum8(acc);

    /* Scalar remainder (< 32 elements) */
    for (; k < K; k++) {
        total += (int32_t)a[k] * (int32_t)b[k];
    }
    return total;
}

/** Compute one B j-block for all M rows of A (B block reused from cache). */
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
        _block_j_avx2(a->A, a->B, a->C, a->M, a->N, a->K, jb, j_end);
    }
    return NULL;
}

static void _gemm_run_threads(const int8_t *A, const int8_t *B, int32_t *C,
                              int M, int N, int K, int nblk, int jblock) {
    int nthreads = _gemm_threads();
    if (nthreads > nblk) nthreads = nblk;
    if (nthreads < 1) nthreads = 1;
    if (nthreads <= 1) {
        for (int jb = 0; jb < N; jb += jblock) {
            int j_end = (jb + jblock < N) ? jb + jblock : N;
            _block_j_avx2(A, B, C, M, N, K, jb, j_end);
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
            _block_j_avx2(A, B, C, M, N, K, jb, j_end);
        }
        return;
    }
    int spawned = 0;
    for (int t = 0; t < nthreads; t++) {
        args[t] = (gemm_worker_arg){
            A, B, C, M, N, K,
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

#if defined(__AVX2__)
    long total_bytes = (long)N * K;
    if (total_bytes >= _THREAD_MIN_BYTES) {
        int nblk = (int)(((long)N + jblock - 1) / jblock);
        _gemm_run_threads(Aq, B, Acc, M, N, K, nblk, (int)jblock);
    } else {
        for (int jb = 0; jb < N; jb += (int)jblock) {
            int j_end = (int)((long)jb + jblock < N ? jb + jblock : N);
            _block_j_avx2(Aq, B, Acc, M, N, K, jb, j_end);
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

    free(Aq); free(Acc); free(Ascale);
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

#if defined(__AVX2__)
    long total_bytes = (long)N * K;
    if (total_bytes >= _THREAD_MIN_BYTES) {
        int nblk = (int)(((long)N + jblock - 1) / jblock);
        _gemm_run_threads(A, B, C, M, N, K, nblk, (int)jblock);
    } else {
        for (int jb = 0; jb < N; jb += (int)jblock) {
            int j_end = (int)((long)jb + jblock < N ? jb + jblock : N);
            _block_j_avx2(A, B, C, M, N, K, jb, j_end);
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
}
