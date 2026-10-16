/**
 * matmul_int4.c — int8 × packed-int4 → int32 GEMM accelerated with AVX2.
 *
 * Computes  C[i,j] = Σₖ A[i,k] · unpack_int4(B[j,k])
 *
 * Layout:
 *   A:        M×K    int8_t   (row-major, activations)
 *   B_packed: N×K/2  uint8_t  (row-major, packed int4 weights)
 *              Each byte: low nibble = val[even], high nibble = val[odd]
 *   C:        M×N    int32_t  (row-major, written)
 *
 * Unpack: sign-extend 4-bit signed value (-8..7) to int8:
 *   signed = (nibble ^ 0x08) - 0x08
 *
 * Strategy:
 *   For M == 1 the packed B row is unpacked inline inside the dot loop
 *   (register-resident, no scratch traffic). For M > 1 the packed B block is
 *   unpacked ONCE into an int8 scratch buffer per j-block, then re-used across
 *   all M rows of A — this removes M× redundant unpacking work and keeps B
 *   packed traffic at half of int8. The scratch block is sized to stay in L2.
 *
 * On CPUs without AVX2 a scalar fallback is used.
 *
 * Build:
 *   gcc -O3 -mavx2 -shared -fPIC -o matmul_int4.so matmul_int4.c
 */

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/** Target size of a packed B j-block in bytes — keep A + B-block in L2. */
#define _JB_TARGET_BYTES 262144L  /* 256KB */
#define _JB_MIN 32
#define _JB_MAX 4096

/* ── AVX2 implementation ────────────────────────────────────────── */

#if defined(__AVX2__)

/** Horizontal sum of 8 int32 lanes → single int32. */
static inline int32_t _hsum8(__m256i v) {
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

/** Process 32 int8 values from a and b → 8 int32 partial sums. */
static inline __m256i _dot32(const int8_t *a, const int8_t *b) {
    __m128i a_lo = _mm_loadu_si128((const __m128i *)a);
    __m128i a_hi = _mm_loadu_si128((const __m128i *)(a + 16));
    __m128i b_lo = _mm_loadu_si128((const __m128i *)b);
    __m128i b_hi = _mm_loadu_si128((const __m128i *)(b + 16));

    __m256i a_w_lo = _mm256_cvtepi8_epi16(a_lo);
    __m256i a_w_hi = _mm256_cvtepi8_epi16(a_hi);
    __m256i b_w_lo = _mm256_cvtepi8_epi16(b_lo);
    __m256i b_w_hi = _mm256_cvtepi8_epi16(b_hi);

    __m256i p_lo = _mm256_mullo_epi16(a_w_lo, b_w_lo);
    __m256i p_hi = _mm256_mullo_epi16(a_w_hi, b_w_hi);

    return _mm256_add_epi32(
        _mm256_madd_epi16(p_lo, _mm256_set1_epi16(1)),
        _mm256_madd_epi16(p_hi, _mm256_set1_epi16(1)));
}

/** Dot product of two length-K int8 rows → int32. */
static inline int32_t _dot_row_int8(const int8_t *a, const int8_t *b, int K) {
    __m256i acc = _mm256_setzero_si256();
    int k = 0;

    for (; k + 64 <= K; k += 64) {
        acc = _mm256_add_epi32(acc, _dot32(a + k, b + k));
        acc = _mm256_add_epi32(acc, _dot32(a + k + 32, b + k + 32));
    }
    for (; k + 32 <= K; k += 32) {
        acc = _mm256_add_epi32(acc, _dot32(a + k, b + k));
    }

    int32_t total = _hsum8(acc);

    for (; k < K; k++) {
        total += (int32_t)a[k] * (int32_t)b[k];
    }
    return total;
}

/**
 * Unpack one packed-int4 row (K nibbles) into an int8 row of length K.
 * b_packed must have K/2 bytes; K is assumed even.
 */
static inline void _unpack_row_int4(const uint8_t *b_packed, int8_t *out, int K) {
    int k = 0;
    for (; k + 32 <= K; k += 32) {
        __m128i packed = _mm_loadu_si128((const __m128i *)(b_packed + k / 2));

        __m128i lo = _mm_and_si128(packed, _mm_set1_epi8(0x0F));
        __m128i hi = _mm_and_si128(
            _mm_srli_epi16(packed, 4), _mm_set1_epi8(0x0F));

        __m128i xmask = _mm_set1_epi8(0x08);
        lo = _mm_sub_epi8(_mm_xor_si128(lo, xmask), xmask);
        hi = _mm_sub_epi8(_mm_xor_si128(hi, xmask), xmask);

        __m128i b_lo = _mm_unpacklo_epi8(lo, hi);
        __m128i b_hi = _mm_unpackhi_epi8(lo, hi);

        _mm_storeu_si128((__m128i *)(out + k), b_lo);
        _mm_storeu_si128((__m128i *)(out + k + 16), b_hi);
    }
    for (; k < K; k++) {
        int v = (k & 1) ? (b_packed[k / 2] >> 4) & 0x0F : b_packed[k / 2] & 0x0F;
        out[k] = (int8_t)((v ^ 8) - 8);
    }
}

/** Dot product of a length-K int8 row with a packed-int4 row → int32. */
static inline int32_t _dot_row_int4(const int8_t *a, const uint8_t *b_packed, int K) {
    __m256i acc = _mm256_setzero_si256();
    int k = 0;

    for (; k + 32 <= K; k += 32) {
        __m128i packed = _mm_loadu_si128((const __m128i *)(b_packed + k / 2));

        __m128i lo = _mm_and_si128(packed, _mm_set1_epi8(0x0F));
        __m128i hi = _mm_and_si128(
            _mm_srli_epi16(packed, 4), _mm_set1_epi8(0x0F));

        __m128i xmask = _mm_set1_epi8(0x08);
        lo = _mm_sub_epi8(_mm_xor_si128(lo, xmask), xmask);
        hi = _mm_sub_epi8(_mm_xor_si128(hi, xmask), xmask);

        __m128i nb_lo = _mm_unpacklo_epi8(lo, hi);  /* [val0..val15] natural order */
        __m128i nb_hi = _mm_unpackhi_epi8(lo, hi);  /* [val16..val31] natural order */

        __m128i a_lo = _mm_loadu_si128((const __m128i *)(a + k));
        __m128i a_hi = _mm_loadu_si128((const __m128i *)(a + k + 16));

        __m256i a_w_lo = _mm256_cvtepi8_epi16(a_lo);
        __m256i a_w_hi = _mm256_cvtepi8_epi16(a_hi);
        __m256i b_w_lo = _mm256_cvtepi8_epi16(nb_lo);
        __m256i b_w_hi = _mm256_cvtepi8_epi16(nb_hi);

        __m256i p_lo = _mm256_mullo_epi16(a_w_lo, b_w_lo);
        __m256i p_hi = _mm256_mullo_epi16(a_w_hi, b_w_hi);

        acc = _mm256_add_epi32(
            acc,
            _mm256_add_epi32(
                _mm256_madd_epi16(p_lo, _mm256_set1_epi16(1)),
                _mm256_madd_epi16(p_hi, _mm256_set1_epi16(1))));
    }

    int32_t total = _hsum8(acc);

    for (; k < K; k++) {
        int v = (k & 1) ? (b_packed[k / 2] >> 4) & 0x0F : b_packed[k / 2] & 0x0F;
        total += (int32_t)a[k] * (int8_t)((v ^ 8) - 8);
    }
    return total;
}

#endif /* __AVX2__ */

/* ── Public API ─────────────────────────────────────────────────── */

void matmul_int4(const int8_t *A, const uint8_t *B_packed, int32_t *C,
                 int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    memset(C, 0, (size_t)M * N * sizeof(int32_t));

    /* B j-block width: keep the packed B block (JB×K/2 bytes) plus A in L2. */
    long jblock = _JB_TARGET_BYTES / ((long)K / 2);
    if (jblock < _JB_MIN) jblock = _JB_MIN;
    if (jblock > _JB_MAX) jblock = _JB_MAX;

#if defined(__AVX2__)
    if (M > 1) {
        /* Unpack each B row once per j-block into int8 scratch, reuse across
         * all M rows of A (removes M× redundant unpack work). */
        int8_t *scratch = (int8_t *)malloc((size_t)jblock * K);
        if (scratch) {
            for (int jb = 0; jb < N; jb += (int)jblock) {
                int j_end = (int)((long)jb + jblock < N ? jb + jblock : N);
                int ncols = j_end - jb;

                for (int j = 0; j < ncols; j++) {
                    _unpack_row_int4(
                        B_packed + (size_t)(jb + j) * (K / 2),
                        scratch + (size_t)j * K, K);
                }
                for (int i = 0; i < M; i++) {
                    const int8_t *a_row = A + (size_t)i * K;
                    int32_t       *c_row = C + (size_t)i * N;
                    for (int j = 0; j < ncols; j++) {
                        c_row[jb + j] = _dot_row_int8(
                            a_row, scratch + (size_t)j * K, K);
                    }
                }
            }
            free(scratch);
            return;
        }
        /* malloc failed — fall through to inline path */
    }

    /* M == 1 (or scratch unavailable): inline unpack, register-resident. */
    for (int jb = 0; jb < N; jb += (int)jblock) {
        int j_end = (int)((long)jb + jblock < N ? jb + jblock : N);
        for (int i = 0; i < M; i++) {
            const int8_t  *a_row = A + (size_t)i * K;
            int32_t       *c_row = C + (size_t)i * N;
            for (int j = jb; j < j_end; j++) {
                c_row[j] = _dot_row_int4(a_row,
                                         B_packed + (size_t)j * (K / 2), K);
            }
        }
    }
#else
    /* Scalar fallback */
    for (int jb = 0; jb < N; jb += (int)jblock) {
        int j_end = (int)((long)jb + jblock < N ? jb + jblock : N);
        for (int i = 0; i < M; i++) {
            const int8_t  *a_row = A + (size_t)i * K;
            int32_t       *c_row = C + (size_t)i * N;
            for (int j = jb; j < j_end; j++) {
                const uint8_t *b_row = B_packed + (size_t)j * (K / 2);
                int32_t total = 0;
                for (int k = 0; k < K; k++) {
                    int v = (k & 1) ? (b_row[k / 2] >> 4) & 0x0F
                                    : b_row[k / 2] & 0x0F;
                    total += (int32_t)a_row[k] * (int8_t)((v ^ 8) - 8);
                }
                c_row[j] = total;
            }
        }
    }
#endif
}
