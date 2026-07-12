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
 * On CPUs without AVX2 a scalar fallback is used.
 *
 * Build:
 *   gcc -O3 -mavx2 -shared -fPIC -o matmul_int4.dylib matmul_int4.c
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
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

/**
 * Process 32 packed int4 values (16 packed bytes) → 8 int32 partial sums.
 *
 * Loads 16 packed bytes from b, unpacks + sign-extends to 32 int8 values,
 * then dot-products with 32 int8 values from a.
 */
static inline __m256i _dot32_int4(const int8_t *a, const uint8_t *b_packed) {
    /* ── Load 16 packed bytes (32 int4 values) ── */
    __m128i packed = _mm_loadu_si128((const __m128i *)b_packed);

    /* ── Unpack nibbles ── */
    __m128i lo = _mm_and_si128(packed, _mm_set1_epi8(0x0F));
    __m128i hi = _mm_and_si128(
        _mm_srli_epi16(packed, 4), _mm_set1_epi8(0x0F));

    /* ── Sign-extend 4-bit → int8: (val ^ 0x08) - 0x08 ── */
    __m128i xmask = _mm_set1_epi8(0x08);
    lo = _mm_sub_epi8(_mm_xor_si128(lo, xmask), xmask);
    hi = _mm_sub_epi8(_mm_xor_si128(hi, xmask), xmask);

    /* ── Interleave to natural order ──
     * lo = [val0, val2,  ..., val30]  (low nibbles)
     * hi = [val1, val3,  ..., val31]  (high nibbles)
     * unpacklo(lo, hi) → [val0, val1, ..., val15]
     * unpackhi(lo, hi) → [val16, val17, ..., val31]
     */
    __m128i b_lo = _mm_unpacklo_epi8(lo, hi);
    __m128i b_hi = _mm_unpackhi_epi8(lo, hi);

    /* ── Load 32 int8 from A ── */
    __m128i a_lo = _mm_loadu_si128((const __m128i *)a);
    __m128i a_hi = _mm_loadu_si128((const __m128i *)(a + 16));

    /* ── Dot product (same as int8 _dot32) ── */
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

#endif /* __AVX2__ */

/* ── Public API ─────────────────────────────────────────────────── */

void matmul_int4(const int8_t *A, const uint8_t *B_packed, int32_t *C,
                 int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    memset(C, 0, (size_t)M * N * sizeof(int32_t));

#if defined(__AVX2__)
    for (int i = 0; i < M; i++) {
        const int8_t  *a_row = A + (size_t)i * K;
        int32_t       *c_row = C + (size_t)i * N;

        for (int j = 0; j < N; j++) {
            const uint8_t *b_row = B_packed + (size_t)j * (K / 2);
            __m256i acc = _mm256_setzero_si256();
            int k = 0;

            /* Process 32 original-K elements per iteration.
               This consumes 16 packed bytes from B. */
            for (; k + 32 <= K; k += 32) {
                acc = _mm256_add_epi32(
                    acc, _dot32_int4(a_row + k, b_row + k / 2));
            }

            int32_t total = _hsum8(acc);

            /* Scalar remainder (< 32 elements) */
            for (; k < K; k++) {
                int b_val;
                if (k % 2 == 0) {
                    b_val = b_row[k / 2] & 0x0F;       /* low nibble */
                } else {
                    b_val = (b_row[k / 2] >> 4) & 0x0F; /* high nibble */
                }
                b_val = (b_val ^ 8) - 8;  /* sign-extend 4-bit */
                total += (int32_t)a_row[k] * b_val;
            }

            c_row[j] = total;
        }
    }
#else
    /* Scalar fallback */
    for (int i = 0; i < M; i++) {
        const int8_t  *a_row = A + (size_t)i * K;
        int32_t       *c_row = C + (size_t)i * N;
        for (int j = 0; j < N; j++) {
            const uint8_t *b_row = B_packed + (size_t)j * (K / 2);
            int32_t total = 0;
            for (int k = 0; k < K; k++) {
                int b_val;
                if (k % 2 == 0) {
                    b_val = b_row[k / 2] & 0x0F;
                } else {
                    b_val = (b_row[k / 2] >> 4) & 0x0F;
                }
                b_val = (b_val ^ 8) - 8;
                total += (int32_t)a_row[k] * b_val;
            }
            c_row[j] = total;
        }
    }
#endif
}
