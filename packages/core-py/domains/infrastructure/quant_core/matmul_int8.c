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
 * On CPUs without AVX2 a scalar fallback is used.
 *
 * Build:
 *   gcc -O3 -mavx2 -shared -fPIC -o matmul_int8.dylib matmul_int8.c
 */

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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

#endif /* __AVX2__ */

/* ── Public API ─────────────────────────────────────────────────── */

void matmul_int8(const int8_t *A, const int8_t *B, int32_t *C,
                 int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    memset(C, 0, (size_t)M * N * sizeof(int32_t));

#if defined(__AVX2__)
    for (int i = 0; i < M; i++) {
        const int8_t *a_row = A + (size_t)i * K;
        int32_t       *c_row = C + (size_t)i * N;

        for (int j = 0; j < N; j++) {
            const int8_t *b_row = B + (size_t)j * K;
            __m256i acc = _mm256_setzero_si256();
            int k = 0;

            /* Process 64 elements per iteration */
            for (; k + 64 <= K; k += 64) {
                acc = _mm256_add_epi32(acc, _dot32(a_row + k, b_row + k));
                acc = _mm256_add_epi32(acc, _dot32(a_row + k + 32, b_row + k + 32));
            }
            /* Process remaining 32-element block */
            for (; k + 32 <= K; k += 32) {
                acc = _mm256_add_epi32(acc, _dot32(a_row + k, b_row + k));
            }

            int32_t total = _hsum8(acc);

            /* Scalar remainder (< 32 elements) */
            for (; k < K; k++) {
                total += (int32_t)a_row[k] * (int32_t)b_row[k];
            }

            c_row[j] = total;
        }
    }
#else
    /* Scalar fallback */
    for (int i = 0; i < M; i++) {
        const int8_t *a_row = A + (size_t)i * K;
        int32_t       *c_row = C + (size_t)i * N;
        for (int j = 0; j < N; j++) {
            const int8_t *b_row = B + (size_t)j * K;
            int32_t total = 0;
            for (int k = 0; k < K; k++) {
                total += (int32_t)a_row[k] * (int32_t)b_row[k];
            }
            c_row[j] = total;
        }
    }
#endif
}
