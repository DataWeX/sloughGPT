/* tensor_kernels.c — fused tensor primitives, ONE source, TWO engines.
 *
 * Card d74e2cef phase 2. The NPU tensor unit's hot row/elementwise ops,
 * written once as plain C and compiled twice by
 * domain/slolib/_internal/native.py:
 *
 *   CPU: gcc -O3 -fopenmp -> ctypes.  <op>_cpu: OpenMP parallel-for over
 *        rows/elements. ctypes releases the GIL; OpenMP fans out.
 *   GPU: NVRTC -> cupy RawKernel.     <op>_gpu: one thread per
 *        row/element.
 *
 * SLO_Q is the ONLY target-specific token (empty on CPU, __device__ on
 * GPU) — same pattern as trace_core.c/scene_build.c, proven on this
 * toolchain (gcc 13 + NVRTC). No #include <math.h>: gcc and NVRTC both
 * builtin exp/tanh/sqrt with correct double signatures.
 *
 * Why these ops: scripts/benchmark_tensor_kernels.py measured the
 * numpy multi-pass + temporaries pattern (softmax/layernorm/rmsnorm/
 * gelu/relu allocate 2-8 full temporaries each) against the fused
 * single-pass loops here. matmul is deliberately ABSENT — it is BLAS
 * territory; numpy's sgemm/dgemm beats a portable C gemm, so the CPU
 * keeps it (the benchmark carries it as the control that proves the
 * harness reports losers too).
 *
 * Semantics mirror domain/shell/_internal/tensor_device.py exactly:
 *   - softmax:     row max, exp(x-max), sum, divide (axis=-1 rows;
 *                  axis=None flattens to ONE row at the call site)
 *   - layernorm:   mean/var over the row, ddof=0, w*x_hat + b; w/b may
 *                  be NULL for the weightless _layernorm_global form
 *   - rmsnorm:     x / sqrt(mean(x^2) + eps) * w; w NULL = global form
 *   - gelu:        0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) — the
 *                  tanh approximation, NOT erf (matches numpy path)
 *   - relu:        maximum(0, x) — NaN propagates (x not >0 and not <0)
 * dtype is float64 everywhere: the device's canonical dtype (CAPS
 * dtypes=["float64"]); the Python side only routes float64 inputs.
 * numpy stays the oracle AND the fallback: any input shape/dtype this
 * file doesn't take (0-d, strided views beyond the guard, f32, 3-D
 * softmax) goes down the numpy path untouched.
 */

#ifdef __CUDACC__
#define SLO_Q __device__
#else
#define SLO_Q
#endif

/* ── shared row/element helpers (identical instruction stream) ──────────── */

SLO_Q static double slo_row_max(const double *x, long cols) {
    double m = x[0];
    for (long j = 1; j < cols; ++j) {
        if (x[j] > m) m = x[j];
    }
    return m;
}

SLO_Q static void slo_softmax_row(const double *x, double *out, long cols) {
    double m = slo_row_max(x, cols);
    double s = 0.0;
    for (long j = 0; j < cols; ++j) {
        double e = exp(x[j] - m);
        out[j] = e;
        s += e;
    }
    double inv = 1.0 / s;
    for (long j = 0; j < cols; ++j) {
        out[j] *= inv;
    }
}

SLO_Q static void slo_layernorm_row(const double *x, const double *w,
                                    const double *b, double eps, double *out,
                                    long cols) {
    double mean = 0.0;
    for (long j = 0; j < cols; ++j) mean += x[j];
    mean /= (double)cols;
    double acc = 0.0;
    for (long j = 0; j < cols; ++j) {
        double d = x[j] - mean;
        acc += d * d;
    }
    double inv = 1.0 / sqrt(acc / (double)cols + eps);
    for (long j = 0; j < cols; ++j) {
        double xh = (x[j] - mean) * inv;
        out[j] = w != 0 ? w[j] * xh + b[j] : xh;
    }
}

SLO_Q static void slo_rmsnorm_row(const double *x, const double *w,
                                  double eps, double *out, long cols) {
    double acc = 0.0;
    for (long j = 0; j < cols; ++j) acc += x[j] * x[j];
    double inv = 1.0 / sqrt(acc / (double)cols + eps);
    for (long j = 0; j < cols; ++j) {
        out[j] = w != 0 ? x[j] * inv * w[j] : x[j] * inv;
    }
}

SLO_Q static double slo_gelu(double v) {
    /* sqrt(2.0/pi) as the SAME expression numpy evaluates — divide
     * then sqrt — so both engines start from the same double. */
    double c = sqrt(2.0 / 3.14159265358979323846);
    return 0.5 * v * (1.0 + tanh(c * (v + 0.044715 * v * v * v)));
}

/* ── entries: the ONLY place the two engines diverge ────────────────────── */

#ifdef __CUDACC__

extern "C" __global__ void tens_softmax_gpu(const double *x, double *out,
                                            long rows, long cols) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    slo_softmax_row(x + i * cols, out + i * cols, cols);
}

extern "C" __global__ void tens_layernorm_gpu(const double *x, const double *w,
                                              const double *b, double *out,
                                              long rows, long cols,
                                              double eps) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    slo_layernorm_row(x + i * cols, w, b, eps, out + i * cols, cols);
}

extern "C" __global__ void tens_rmsnorm_gpu(const double *x, const double *w,
                                            double *out, long rows, long cols,
                                            double eps) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    slo_rmsnorm_row(x + i * cols, w, eps, out + i * cols, cols);
}

extern "C" __global__ void tens_gelu_gpu(const double *x, double *out,
                                         long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = slo_gelu(x[i]);
}

extern "C" __global__ void tens_relu_gpu(const double *x, double *out,
                                         long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double v = x[i];
    out[i] = v > 0.0 ? v : (v < 0.0 ? 0.0 : v);
}

#else

#ifdef _OPENMP
#include <omp.h>
#endif

/* CPU entries: one OpenMP thread per row/element. ctypes releases the
 * GIL for the call, so the team runs wide across cores. */

extern void tens_softmax_cpu(const double *x, double *out, long rows,
                             long cols) {
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < rows; ++i) {
        slo_softmax_row(x + i * cols, out + i * cols, cols);
    }
}

extern void tens_layernorm_cpu(const double *x, const double *w,
                               const double *b, double *out, long rows,
                               long cols, double eps) {
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < rows; ++i) {
        slo_layernorm_row(x + i * cols, w, b, eps, out + i * cols, cols);
    }
}

extern void tens_rmsnorm_cpu(const double *x, const double *w, double *out,
                             long rows, long cols, double eps) {
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < rows; ++i) {
        slo_rmsnorm_row(x + i * cols, w, eps, out + i * cols, cols);
    }
}

extern void tens_gelu_cpu(const double *x, double *out, long n) {
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) {
        out[i] = slo_gelu(x[i]);
    }
}

extern void tens_relu_cpu(const double *x, double *out, long n) {
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) {
        double v = x[i];
        out[i] = v > 0.0 ? v : (v < 0.0 ? 0.0 : v);
    }
}

#endif
