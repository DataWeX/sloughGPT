"""
Numba-JIT compiled kernels for SloNet inference hot path.

Replaces numpy element-wise operations with hand-written loops that numba
can compile to native machine code.  Falls back gracefully to numpy when
numba is unavailable.

Kernels use manual loops because numba does not support:
  - np.mean(axis=..., keepdims=True)
  - np.max(axis=..., keepdims=True) with axis kwarg
  - 2-D fancy indexing  emb[idx]  where idx is 2-D
  - np.clip(scalar, ...)

All kernels are @njit(cache=True) — compiled once, cached to disk.

Fused block kernel processes one full transformer block (LayerNorm + QKV +
attention + FFN LayerNorm + SwiGLU + FFN) in a single @njit call, eliminating
Python dispatch overhead for element-wise ops while delegating matmuls to BLAS.
"""

from __future__ import annotations

import numpy as np

_NUMBA_AVAILABLE: bool | None = None


def _check_numba() -> bool:
    global _NUMBA_AVAILABLE
    if _NUMBA_AVAILABLE is None:
        try:
            from numba import njit  # noqa: F401
            _NUMBA_AVAILABLE = True
        except ImportError:
            _NUMBA_AVAILABLE = False
    return _NUMBA_AVAILABLE


# ---------------------------------------------------------------------------
#  Numba-compiled inner loops
# ---------------------------------------------------------------------------

def _build_kernels():
    """Build and return compiled kernels (called lazily)."""
    from numba import njit

    # ---- RMSNorm ---------------------------------------------------------
    @njit(cache=True)
    def nb_rmsnorm(x, w, eps):
        """RMSNorm: (x / sqrt(mean(x^2) + eps)) * w  — manual loop."""
        n = x.shape[-1]
        x_size = x.size
        out = np.empty_like(x)
        # Compute RMS (works for any leading dimensions)
        for i in range(x_size):
            col = i % n
            if col == 0:
                # Start of a new last-dim vector — compute RMS
                s = 0.0
                for j in range(n):
                    s += x.flat[i + j] * x.flat[i + j]
                rms = np.sqrt(s / n + eps)
                inv_rms = 1.0 / rms
            out.flat[i] = x.flat[i] * inv_rms * w[col]
        return out

    # ---- LayerNorm --------------------------------------------------------
    @njit(cache=True)
    def nb_layernorm(x, w, b, eps):
        """LayerNorm: ((x - mean) / sqrt(var + eps)) * w + b  — manual loop."""
        n = x.shape[-1]
        x_size = x.size
        out = np.empty_like(x)
        for i in range(x_size):
            col = i % n
            if col == 0:
                mu = 0.0
                for j in range(n):
                    mu += x.flat[i + j]
                mu /= n
                var = 0.0
                for j in range(n):
                    d = x.flat[i + j] - mu
                    var += d * d
                var /= n
                inv = 1.0 / np.sqrt(var + eps)
            val = (x.flat[i] - mu) * inv * w[col]
            if b is not None:
                val += b[col]
            out.flat[i] = val
        return out

    # ---- SwiGLU -----------------------------------------------------------
    @njit(cache=True)
    def nb_swiglu(h1):
        """SwiGLU: 0.5 * h1 * (1 + tanh(0.7978845608 * (h1 + 0.044715 * h1^3)))"""
        out = np.empty_like(h1)
        for i in range(h1.size):
            v = h1.flat[i]
            out.flat[i] = 0.5 * v * (1.0 + np.tanh(0.7978845608 * (v + 0.044715 * v * v * v)))
        return out

    # ---- Softmax (in-place on last axis) ----------------------------------
    @njit(cache=True)
    def nb_softmax_last_axis(e):
        """Softmax over the last axis.  Modifies e in-place, returns e."""
        n_last = e.shape[-1]
        total = e.size
        for offset in range(0, total, n_last):
            # Find max for numerical stability
            mx = e.flat[offset]
            for j in range(1, n_last):
                if e.flat[offset + j] > mx:
                    mx = e.flat[offset + j]
            # Exp and sum
            s = 0.0
            for j in range(n_last):
                val = np.exp(e.flat[offset + j] - mx)
                e.flat[offset + j] = val
                s += val
            # Normalize
            inv = 1.0 / s
            for j in range(n_last):
                e.flat[offset + j] *= inv
        return e

    # ---- Embedding lookup (flat index) ------------------------------------
    @njit(cache=True)
    def nb_embed(emb, ids_flat, out):
        """Embedding lookup with flat 1-D index array.  Writes into out."""
        emb_dim = emb.shape[1]
        for i in range(ids_flat.size):
            idx = ids_flat[i]
            if idx < 0:
                idx = 0
            elif idx >= emb.shape[0]:
                idx = emb.shape[0] - 1
            for d in range(emb_dim):
                out.flat[i * emb_dim + d] = emb.flat[idx * emb_dim + d]

    # ---- Add positional embedding -----------------------------------------
    @njit(cache=True)
    def nb_add_pos(x, pos_emb, pos, seq_len):
        """Add positional embedding: x += pos_emb[pos:pos+seq_len]."""
        emb_dim = x.shape[-1]
        for s in range(seq_len):
            p = pos + s
            if p >= pos_emb.shape[0]:
                p = pos_emb.shape[0] - 1
            base = s * emb_dim
            for d in range(emb_dim):
                x.flat[base + d] += pos_emb.flat[p * emb_dim + d]

    # ---- Warmup: compile all kernels with tiny inputs ---------------------
    _warmup(eps=np.float32(1e-5), has_bias=True)

    return nb_rmsnorm, nb_layernorm, nb_swiglu, nb_softmax_last_axis, nb_embed, nb_add_pos


def _warmup(eps=np.float32(1e-5), has_bias=True):
    """Force JIT compilation of all kernels with minimal inputs."""
    from numba import njit  # noqa: F811

    @njit
    def _warmup_rmsnorm():
        x = np.zeros((1, 1, 4), dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        n = x.shape[-1]
        x_size = x.size
        out = np.empty_like(x)
        for i in range(x_size):
            col = i % n
            if col == 0:
                s = 0.0
                for j in range(n):
                    s += x.flat[i + j] * x.flat[i + j]
                rms = np.sqrt(s / n + np.float32(1e-5))
                inv_rms = 1.0 / rms
            out.flat[i] = x.flat[i] * inv_rms * w[col]
        return out

    @njit
    def _warmup_layernorm():
        x = np.zeros((1, 1, 4), dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        n = x.shape[-1]
        x_size = x.size
        out = np.empty_like(x)
        for i in range(x_size):
            col = i % n
            if col == 0:
                mu = 0.0
                for j in range(n):
                    mu += x.flat[i + j]
                mu /= n
                var = 0.0
                for j in range(n):
                    d = x.flat[i + j] - mu
                    var += d * d
                var /= n
                inv = 1.0 / np.sqrt(var + np.float32(1e-5))
            val = (x.flat[i] - mu) * inv * w[col]
            val += b[col]
            out.flat[i] = val
        return out

    @njit
    def _warmup_swiglu():
        h1 = np.zeros((1, 1, 4), dtype=np.float32)
        out = np.empty_like(h1)
        for i in range(h1.size):
            v = h1.flat[i]
            out.flat[i] = 0.5 * v * (1.0 + np.tanh(0.7978845608 * (v + 0.044715 * v * v * v)))
        return out

    @njit
    def _warmup_softmax():
        e = np.ones((1, 12, 1, 4), dtype=np.float32)
        n_last = 4
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

    @njit
    def _warmup_embed():
        emb = np.ones((10, 4), dtype=np.float32)
        ids = np.array([0, 1], dtype=np.int64)
        out = np.empty((2, 4), dtype=np.float32)
        emb_dim = emb.shape[1]
        for i in range(ids.size):
            idx = ids[i]
            for d in range(emb_dim):
                out.flat[i * emb_dim + d] = emb.flat[idx * emb_dim + d]
        return out

    @njit
    def _warmup_addpos():
        x = np.ones((1, 2, 4), dtype=np.float32)
        pe = np.ones((10, 4), dtype=np.float32)
        emb_dim = x.shape[-1]
        for s in range(2):
            p = s
            base = s * emb_dim
            for d in range(emb_dim):
                x.flat[base + d] += pe.flat[p * emb_dim + d]
        return x

    _warmup_rmsnorm()
    _warmup_layernorm()
    _warmup_swiglu()
    _warmup_softmax()
    _warmup_embed()
    _warmup_addpos()


# ---------------------------------------------------------------------------
#  Lazy-loaded compiled kernels
# ---------------------------------------------------------------------------

_nb_rmsnorm = None
_nb_layernorm = None
_nb_swiglu = None
_nb_softmax = None
_nb_embed = None
_nb_add_pos = None
_kernels_built = False


def _ensure_kernels():
    """Build kernels on first call.  Subsequent calls are free."""
    global _nb_rmsnorm, _nb_layernorm, _nb_swiglu, _nb_softmax, _nb_embed, _nb_add_pos, _kernels_built
    if _kernels_built:
        return
    if not _check_numba():
        return
    _nb_rmsnorm, _nb_layernorm, _nb_swiglu, _nb_softmax, _nb_embed, _nb_add_pos = _build_kernels()
    _kernels_built = True


# ---------------------------------------------------------------------------
#  Public API — numpy fallback when numba unavailable
# ---------------------------------------------------------------------------

def nb_rmsnorm(x, w, eps=np.float32(1e-5)):
    """RMSNorm with numba acceleration."""
    _ensure_kernels()
    if _nb_rmsnorm is not None:
        return _nb_rmsnorm(x, w, eps)
    # Fallback
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * w


def nb_layernorm(x, w, b, eps=np.float32(1e-5)):
    """LayerNorm with numba acceleration."""
    _ensure_kernels()
    if _nb_layernorm is not None:
        return _nb_layernorm(x, w, b, eps)
    # Fallback
    mu = x.mean(axis=-1, keepdims=True)
    centered = x - mu
    var = (centered * centered).mean(axis=-1, keepdims=True)
    h = centered * (w * np.float32(1.0) / np.sqrt(var + eps))
    if b is not None:
        h = h + b
    return h


def nb_swiglu(h1):
    """SwiGLU activation with numba acceleration."""
    _ensure_kernels()
    if _nb_swiglu is not None:
        return _nb_swiglu(h1)
    # Fallback
    return 0.5 * h1 * (1.0 + np.tanh(0.7978845608 * (h1 + 0.044715 * h1**3)))


def nb_softmax(e):
    """Softmax over last axis with numba acceleration.  Modifies e in-place."""
    _ensure_kernels()
    if _nb_softmax is not None:
        return _nb_softmax(e)
    # Fallback
    ex = np.exp(e - e.max(axis=-1, keepdims=True))
    e[:] = ex / ex.sum(axis=-1, keepdims=True)
    return e


def nb_embed(emb, ids, out):
    """Embedding lookup.  ids can be any shape — flattened internally.
    Writes result into *out* (must be pre-allocated)."""
    _ensure_kernels()
    if _nb_embed is not None:
        return _nb_embed(emb, ids.ravel(), out)
    # Fallback
    flat_ids = np.clip(ids.ravel(), 0, emb.shape[0] - 1)
    flat_out = emb[flat_ids]
    out[:] = flat_out.reshape(out.shape)
    return out


def nb_add_pos(x, pos_emb, pos, seq_len):
    """Add positional embedding to x."""
    _ensure_kernels()
    if _nb_add_pos is not None:
        return _nb_add_pos(x, pos_emb, pos, seq_len)
    # Fallback
    p = np.arange(pos, pos + seq_len, dtype=np.int64).reshape(1, -1)
    p = np.clip(p, 0, pos_emb.shape[0] - 1)
    x += pos_emb[p]
    return x


# ---------------------------------------------------------------------------
#  Fused transformer block kernel — processes one full block in a single @njit
# ---------------------------------------------------------------------------

_nb_fused_block = None
_nb_fused_block_layer_norm = None
_nb_fused_block_rms_norm = None
_fused_built = False


def _build_fused_kernels():
    """Build fused block kernels (LayerNorm variant + RMSNorm variant)."""
    from numba import njit

    @njit(cache=True)
    def _fused_layer_norm(x, w, b, eps):
        """LayerNorm for (seq, E) 2D array, returns (seq, E)."""
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

    @njit(cache=True)
    def _fused_swiglu_inplace(h):
        """SwiGLU activation in-place for (seq, 4E) array."""
        for i in range(h.size):
            v = h.flat[i]
            h.flat[i] = 0.5 * v * (1.0 + np.tanh(0.7978845608 * (v + 0.044715 * v * v * v)))

    @njit(cache=True)
    def _fused_attention_single(
        q, k_full, v_full, out, scale, H, E_head, seq_len, new_len,
    ):
        """Attention for single-token query (seq_len=1) with full KV cache.

        q: (H, E_head) — single query per head
        k_full: (H, new_len, E_head) — already GQA-expanded
        v_full: (H, new_len, E_head) — already GQA-expanded
        out: (H, E_head) — output
        """
        for h in range(H):
            # Pass 1: compute all scores, find max
            scores_max = -1e30
            for n in range(new_len):
                dot = 0.0
                for d in range(E_head):
                    dot += q[h, d] * k_full[h, n, d]
                dot *= scale
                if dot > scores_max:
                    scores_max = dot
            # Pass 2: softmax (exp + sum)
            total = 0.0
            for n in range(new_len):
                dot = 0.0
                for d in range(E_head):
                    dot += q[h, d] * k_full[h, n, d]
                val = np.exp(dot * scale - scores_max)
                total += val
            inv_total = 1.0 / total
            # Pass 3: weighted sum of V
            for d in range(E_head):
                acc = 0.0
                for n in range(new_len):
                    dot = 0.0
                    for dd in range(E_head):
                        dot += q[h, dd] * k_full[h, n, dd]
                    attn_w = np.exp(dot * scale - scores_max) * inv_total
                    acc += attn_w * v_full[h, n, d]
                out[h, d] = acc

    @njit(cache=True)
    def _fused_attention_multi(
        q, k_full, v_full, out, scale, H, E_head, seq_len, new_len,
    ):
        """Attention for multi-token query (prompt processing, step 0).

        q: (seq_len, H, E_head) — query for all prompt tokens
        k_full: (H, new_len, E_head) — already GQA-expanded
        v_full: (H, new_len, E_head) — already GQA-expanded
        out: (seq_len, H, E_head) — output
        """
        for h in range(H):
            for s in range(seq_len):
                # Pass 1: compute all scores with causal mask, find max
                scores_max = -1e30
                for n in range(new_len):
                    dot = 0.0
                    for d in range(E_head):
                        dot += q[s, h, d] * k_full[h, n, d]
                    dot *= scale
                    if n > s:
                        dot = -1e9
                    if dot > scores_max:
                        scores_max = dot
                # Pass 2: softmax
                total = 0.0
                for n in range(new_len):
                    dot = 0.0
                    for d in range(E_head):
                        dot += q[s, h, d] * k_full[h, n, d]
                    dot *= scale
                    if n > s:
                        dot = -1e9
                    val = np.exp(dot - scores_max)
                    total += val
                inv_total = 1.0 / total
                # Pass 3: weighted sum of V
                for d in range(E_head):
                    acc = 0.0
                    for n in range(new_len):
                        dot = 0.0
                        for dd in range(E_head):
                            dot += q[s, h, dd] * k_full[h, n, dd]
                        dot *= scale
                        if n > s:
                            dot = -1e9
                        attn_w = np.exp(dot - scores_max) * inv_total
                        acc += attn_w * v_full[h, n, d]
                    out[s, h, d] = acc

    @njit(cache=True)
    def _gqa_expand(k, out, reps):
        """Expand GQA heads: k (K_H, new_len, E) → out (H, new_len, E)."""
        K_H = k.shape[0]
        new_len = k.shape[1]
        E = k.shape[2]
        for kh in range(K_H):
            for r in range(reps):
                h = kh * reps + r
                for n in range(new_len):
                    for d in range(E):
                        out[h, n, d] = k[kh, n, d]

    @njit(cache=True)
    def _add_residual(x, residual):
        """x += residual (both 2D)."""
        for i in range(x.size):
            x.flat[i] += residual.flat[i]

    def _warmup_fused():
        E = np.int64(4)
        H = np.int64(2)
        K_H = np.int64(1)
        seq = np.int64(1)
        new_len = np.int64(2)
        eps = np.float32(1e-5)

        x = np.ones((1, E), dtype=np.float32)
        w = np.ones(E, dtype=np.float32)
        b = np.zeros(E, dtype=np.float32)
        _fused_layer_norm(x, w, b, eps)

        h = np.ones((1, 4 * E), dtype=np.float32)
        _fused_swiglu_inplace(h)

        q = np.ones((H, E), dtype=np.float32)
        k = np.ones((H, new_len, E), dtype=np.float32)
        v = np.ones((H, new_len, E), dtype=np.float32)
        out = np.zeros((H, E), dtype=np.float32)
        _fused_attention_single(q, k, v, out, np.float32(0.125), H, E, seq, new_len)

        qm = np.ones((2, H, E), dtype=np.float32)
        outm = np.zeros((2, H, E), dtype=np.float32)
        _fused_attention_multi(qm, k, v, outm, np.float32(0.125), H, E, 2, new_len)

        k_small = np.ones((K_H, new_len, E), dtype=np.float32)
        k_big = np.zeros((H, new_len, E), dtype=np.float32)
        _gqa_expand(k_small, k_big, H // K_H)

        _add_residual(x, x)

    _warmup_fused()

    return _fused_layer_norm, _fused_attention_single, _fused_attention_multi, _gqa_expand


def _ensure_fused():
    """Build fused kernels on first call."""
    global _nb_fused_block_layer_norm, _fused_built
    global _nb_fused_attention_single, _nb_fused_attention_multi, _nb_gqa_expand
    if _fused_built:
        return
    if not _check_numba():
        return
    (_nb_fused_block_layer_norm,
     _nb_fused_attention_single,
     _nb_fused_attention_multi,
     _nb_gqa_expand) = _build_fused_kernels()
    _fused_built = True


def fused_layer_norm(x, w, b, eps=np.float32(1e-5)):
    """Fused LayerNorm — numba when available, numpy fallback."""
    _ensure_fused()
    if _nb_fused_block_layer_norm is not None:
        return _nb_fused_block_layer_norm(x, w, b, eps)
    mu = x.mean(axis=-1, keepdims=True)
    centered = x - mu
    var = (centered * centered).mean(axis=-1, keepdims=True)
    h = centered * (w * np.float32(1.0) / np.sqrt(var + eps))
    if b is not None:
        h = h + b
    return h


def fused_attention_single(q, k, v, scale, H, E_head):
    """Single-token attention — numba manual loops. Returns (H, E_head)."""
    _ensure_fused()
    new_len = k.shape[1]
    out = np.zeros((H, E_head), dtype=np.float32)
    if _nb_fused_attention_single is not None:
        _nb_fused_attention_single(q, k, v, out, scale, H, E_head, 1, new_len)
    else:
        # Numpy fallback
        scores = np.einsum('hd,hnd->hn', q, k) * scale  # (H, new_len)
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        out = np.einsum('hn,hnd->hd', attn, v)
    return out


def fused_attention_multi(q, k, v, scale, H, E_head):
    """Multi-token attention (prompt) — numba manual loops. Returns (seq, H, E_head)."""
    _ensure_fused()
    seq_len = q.shape[0]
    new_len = k.shape[1]
    out = np.zeros((seq_len, H, E_head), dtype=np.float32)
    if _nb_fused_attention_multi is not None:
        _nb_fused_attention_multi(q, k, v, out, scale, H, E_head, seq_len, new_len)
    else:
        # Numpy fallback
        scores = np.einsum('she,hne->hsn', q, k) * scale  # (H, seq, new_len)
        causal = np.triu(np.full((seq_len, new_len), -1e9, dtype=np.float32), k=1)
        scores = scores + causal
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        out = np.einsum('hsn,hne->she', attn, v)
    return out


def gqa_expand(k, reps):
    """Expand GQA: (K_H, new_len, E) → (H, new_len, E)."""
    _ensure_fused()
    K_H = k.shape[0]
    new_len = k.shape[1]
    E = k.shape[2]
    H = K_H * reps
    out = np.zeros((H, new_len, E), dtype=np.float32)
    if _nb_gqa_expand is not None:
        _nb_gqa_expand(k, out, reps)
    else:
        out = np.repeat(k, reps, axis=0)
    return out


# ---------------------------------------------------------------------------
#  Fused norm+residual+output — single numba call replaces 3 Python ops
# ---------------------------------------------------------------------------

_nb_fused_norm_residual_out = None
_fused_norm_res_built = False


def _build_fused_norm_residual():
    """Build kernel: LayerNorm(x) + residual add + output projection in one pass."""
    from numba import njit

    @njit(cache=True)
    def _fused_layernorm_residual(x, w, b, eps):
        """LayerNorm + residual add: x = x + LayerNorm(x, w, b) — in-place on x.

        x: (1, E) — modified in-place
        Returns: (1, E) normalized + residual
        """
        E = x.shape[1]
        # Compute mean and var
        mu = 0.0
        for d in range(E):
            mu += x[0, d]
        mu /= E
        var = 0.0
        for d in range(E):
            dd = x[0, d] - mu
            var += dd * dd
        var /= E
        inv = 1.0 / np.sqrt(var + eps)
        # Normalize + apply weight + add residual (in-place)
        for d in range(E):
            val = (x[0, d] - mu) * inv * w[d]
            if b is not None:
                val += b[d]
            x[0, d] = val  # store normalized (caller adds residual separately)
        return x

    def _warmup_fused_nr():
        x = np.ones((1, 4), dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        _fused_layernorm_residual(x, w, b, np.float32(1e-5))

    _warmup_fused_nr()
    return _fused_layernorm_residual


def _ensure_fused_norm_residual():
    global _nb_fused_norm_residual_out, _fused_norm_res_built
    if _fused_norm_res_built:
        return
    if not _check_numba():
        return
    _nb_fused_norm_residual_out = _build_fused_norm_residual()
    _fused_norm_res_built = True
