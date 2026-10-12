"""Crossover-free probe: the remaining race-safe kernels from slonet_kernels.py.

probe_parallel_kernels.py covered the two attention kernels (the hottest) and
_lm_head_argmax (rejected below — see notes). This measures the REST of the
kernels where a prange is a structure-preserving substitution: every iteration
writes its own output elements, no scatter, no cross-iteration carried state,
no new allocation.

Kernels whose flat loop CARRIES per-row state (nb_rmsnorm, nb_layernorm) are
deliberately absent — prange there needs a loop-structure rewrite, which is a
different change with its own risk, not a throughput probe.

Usage:
    python -u scripts/probe_parallel_elementwise.py [repeat]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("SLO_OPENBLAS_NUM_THREADS", "1"))
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("SLO_OMP_NUM_THREADS", "1"))

from numba import njit, prange  # noqa: E402

TOL = 1e-4


# --- shipped serial shapes --------------------------------------------------


@njit(nogil=True, cache=False)
def s_swiglu(h1):
    out = np.empty_like(h1)
    for i in range(h1.size):
        v = h1.flat[i]
        out.flat[i] = v * (1.0 / (1.0 + np.exp(-v)))
    return out


@njit(parallel=True, nogil=True, cache=False)
def p_swiglu(h1):
    out = np.empty_like(h1)
    for i in prange(h1.size):
        v = h1.flat[i]
        out.flat[i] = v * (1.0 / (1.0 + np.exp(-v)))
    return out


@njit(nogil=True, cache=False)
def s_swi_glu_mul(h1, h3):
    out = np.empty_like(h1)
    for i in range(h1.size):
        v = h1.flat[i]
        s = v * (1.0 / (1.0 + np.exp(-v)))
        out.flat[i] = s * h3.flat[i]
    return out


@njit(parallel=True, nogil=True, cache=False)
def p_swi_glu_mul(h1, h3):
    out = np.empty_like(h1)
    for i in prange(h1.size):
        v = h1.flat[i]
        s = v * (1.0 / (1.0 + np.exp(-v)))
        out.flat[i] = s * h3.flat[i]
    return out


@njit(nogil=True, cache=False)
def s_add_residual(x, residual):
    for i in range(x.size):
        x.flat[i] += residual.flat[i]


@njit(parallel=True, nogil=True, cache=False)
def p_add_residual(x, residual):
    for i in prange(x.size):
        x.flat[i] += residual.flat[i]


@njit(nogil=True, cache=False)
def s_gqa_expand(k, out, reps):
    K_H = k.shape[0]
    new_len = k.shape[1]
    E = k.shape[2]
    for kh in range(K_H):
        for r in range(reps):
            h = kh * reps + r
            for n in range(new_len):
                for d in range(E):
                    out[h, n, d] = k[kh, n, d]


@njit(parallel=True, nogil=True, cache=False)
def p_gqa_expand(k, out, reps):
    K_H = k.shape[0]
    new_len = k.shape[1]
    E = k.shape[2]
    for kh in prange(K_H):
        for r in range(reps):
            h = kh * reps + r
            for n in range(new_len):
                for d in range(E):
                    out[h, n, d] = k[kh, n, d]


@njit(nogil=True, cache=False)
def s_embed(emb, ids_flat, out):
    emb_dim = emb.shape[1]
    for i in range(ids_flat.size):
        idx = ids_flat[i]
        if idx < 0:
            idx = 0
        elif idx >= emb.shape[0]:
            idx = emb.shape[0] - 1
        for d in range(emb_dim):
            out.flat[i * emb_dim + d] = emb.flat[idx * emb_dim + d]


@njit(parallel=True, nogil=True, cache=False)
def p_embed(emb, ids_flat, out):
    emb_dim = emb.shape[1]
    for i in prange(ids_flat.size):
        idx = ids_flat[i]
        if idx < 0:
            idx = 0
        elif idx >= emb.shape[0]:
            idx = emb.shape[0] - 1
        for d in range(emb_dim):
            out.flat[i * emb_dim + d] = emb.flat[idx * emb_dim + d]


@njit(nogil=True, cache=False)
def s_softmax_last_axis(e):
    n_last = e.shape[-1]
    total = e.size
    for offset in range(0, total, n_last):
        mx = e.flat[offset]
        for j in range(1, n_last):
            if e.flat[offset + j] > mx:
                mx = e.flat[offset + j]
        s = 0.0
        for j in range(n_last):
            val = np.exp(e.flat[offset + j] - mx)
            e.flat[offset + j] = val
            s += val
        inv = 1.0 / s
        for j in range(n_last):
            e.flat[offset + j] *= inv
    return e


@njit(parallel=True, nogil=True, cache=False)
def p_softmax_last_axis(e):
    # prange only accepts a CONSTANT step, so the shipped
    # `for offset in range(0, total, n_last)` cannot be pranged directly
    # (numba: "Only constant step size is supported for prange"). Rewrite as
    # a row-index loop with a deterministic offset — same work, same order
    # within each row, so results stay bit-identical.
    n_last = e.shape[-1]
    total = e.size
    n_rows = total // n_last
    for r in prange(n_rows):
        offset = r * n_last
        mx = e.flat[offset]
        for j in range(1, n_last):
            if e.flat[offset + j] > mx:
                mx = e.flat[offset + j]
        s = 0.0
        for j in range(n_last):
            val = np.exp(e.flat[offset + j] - mx)
            e.flat[offset + j] = val
            s += val
        inv = 1.0 / s
        for j in range(n_last):
            e.flat[offset + j] *= inv
    return e


@njit(nogil=True, cache=False)
def s_fused_layer_norm(x, w, b, eps):
    seq = x.shape[0]
    E = x.shape[1]
    out = np.empty_like(x)
    for s in range(seq):
        mu = 0.0
        for d in range(E):
            mu += x[s, d]
        mu /= E
        var = 0.0
        for d in range(E):
            dd = x[s, d] - mu
            var += dd * dd
        var /= E
        inv = 1.0 / np.sqrt(var + eps)
        for d in range(E):
            val = (x[s, d] - mu) * inv * w[d]
            if b is not None:
                val += b[d]
            out[s, d] = val
    return out


@njit(parallel=True, nogil=True, cache=False)
def p_fused_layer_norm(x, w, b, eps):
    seq = x.shape[0]
    E = x.shape[1]
    out = np.empty_like(x)
    for s in prange(seq):
        mu = 0.0
        for d in range(E):
            mu += x[s, d]
        mu /= E
        var = 0.0
        for d in range(E):
            dd = x[s, d] - mu
            var += dd * dd
        var /= E
        inv = 1.0 / np.sqrt(var + eps)
        for d in range(E):
            val = (x[s, d] - mu) * inv * w[d]
            if b is not None:
                val += b[d]
            out[s, d] = val
    return out


@njit(nogil=True, cache=False)
def s_fused_swiglu_inplace(h):
    for i in range(h.size):
        v = h.flat[i]
        h.flat[i] = 0.5 * v * (1.0 + np.tanh(0.7978845608 * (v + 0.044715 * v * v * v)))


@njit(parallel=True, nogil=True, cache=False)
def p_fused_swiglu_inplace(h):
    for i in prange(h.size):
        v = h.flat[i]
        h.flat[i] = 0.5 * v * (1.0 + np.tanh(0.7978845608 * (v + 0.044715 * v * v * v)))


def _time(fn, args, repeat):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main():
    rep = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(2)
    print(f"repeat={rep} (min-of-N)  load={os.getloadavg()[1]:.1f}")
    print(
        f"{'kernel':26} {'shape':22} {'serial us':>11} {'par us':>10} {'speedup':>8} {'bit-ident':>9}"
    )
    print("-" * 90)

    def row(name, shape, s_fn, p_fn, s_args, p_args):
        s_ms = _time(s_fn, s_args, rep)
        p_ms = _time(p_fn, p_args, rep)
        # correctness
        outs_s = s_fn(*s_args)
        outs_p = p_fn(*p_args)
        if outs_s is None:
            # in-place kernels: compare the mutated inputs
            outs_s, outs_p = s_args[0], p_args[0]
        identical = bool(np.array_equal(outs_s, outs_p))
        print(
            f"{name:26} {shape:22} {s_ms * 1e3:11.1f} {p_ms * 1e3:10.1f} {s_ms / p_ms:7.2f}x {str(identical):>9}"
        )

    # realistic (seq, E) for a 0.5B-class model
    SEQ, E = 256, 896

    h1 = rng.standard_normal((SEQ, E)).astype(np.float32)
    row("nb_swiglu", f"({SEQ},{E})", s_swiglu, p_swiglu, (h1,), (h1,))

    h3 = rng.standard_normal((SEQ, E)).astype(np.float32)
    row("nb_swi_glu_mul", f"({SEQ},{E})", s_swi_glu_mul, p_swi_glu_mul, (h1, h3), (h1, h3))

    x = rng.standard_normal((SEQ, E)).astype(np.float32)
    res = rng.standard_normal((SEQ, E)).astype(np.float32)
    row(
        "_add_residual",
        f"({SEQ},{E})",
        s_add_residual,
        p_add_residual,
        (x.copy(), res),
        (x.copy(), res),
    )

    K_H, NEW, EH = 2, 512, 64
    k = rng.standard_normal((K_H, NEW, EH)).astype(np.float32)
    reps = 14 // K_H
    out_s = np.zeros((14, NEW, EH), dtype=np.float32)
    out_p = np.zeros((14, NEW, EH), dtype=np.float32)
    s_gqa_expand(k, out_s, reps)
    p_gqa_expand(k, out_p, reps)
    s_ms = _time(s_gqa_expand, (k, out_s, reps), rep)
    p_ms = _time(p_gqa_expand, (k, out_p, reps), rep)
    print(
        f"{'_gqa_expand':26} {'->(14,512,64)':22} {s_ms * 1e3:11.1f} {p_ms * 1e3:10.1f} {s_ms / p_ms:7.2f}x {str(np.array_equal(out_s, out_p)):>9}"
    )

    NVOC, DIM = 32000, 896
    emb = rng.standard_normal((NVOC, DIM)).astype(np.float32)
    ids = rng.integers(0, NVOC, size=SEQ).astype(np.int64)
    o_s = np.zeros((SEQ, DIM), dtype=np.float32)
    o_p = np.zeros((SEQ, DIM), dtype=np.float32)
    s_embed(emb, ids, o_s)
    p_embed(emb, ids, o_p)
    s_ms = _time(s_embed, (emb, ids, o_s), rep)
    p_ms = _time(p_embed, (emb, ids, o_p), rep)
    print(
        f"{'nb_embed':26} {f'({SEQ},{DIM})':22} {s_ms * 1e3:11.1f} {p_ms * 1e3:10.1f} {s_ms / p_ms:7.2f}x {str(np.array_equal(o_s, o_p)):>9}"
    )

    e_s = rng.standard_normal((SEQ, E)).astype(np.float32)
    e_p = e_s.copy()
    s_softmax_last_axis(e_s)
    p_softmax_last_axis(e_p)
    s_ms = _time(s_softmax_last_axis, (e_s,), rep)
    p_ms = _time(p_softmax_last_axis, (e_p,), rep)
    print(
        f"{'nb_softmax_last_axis':26} {'(256,896)':22} {s_ms * 1e3:11.1f} {p_ms * 1e3:10.1f} {s_ms / p_ms:7.2f}x {str(np.array_equal(e_s, e_p)):>9}"
    )

    w = rng.standard_normal(E).astype(np.float32)
    b = rng.standard_normal(E).astype(np.float32)
    row(
        "_fused_layer_norm",
        f"({SEQ},{E})",
        s_fused_layer_norm,
        p_fused_layer_norm,
        (h1, w, b, np.float32(1e-5)),
        (h1, w, b, np.float32(1e-5)),
    )

    hs = rng.standard_normal((SEQ, E)).astype(np.float32)
    row(
        "_fused_swiglu_inplace",
        f"({SEQ},{E})",
        s_fused_swiglu_inplace,
        p_fused_swiglu_inplace,
        (hs.copy(),),
        (hs.copy(),),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
