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
