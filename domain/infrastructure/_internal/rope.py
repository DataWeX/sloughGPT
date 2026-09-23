"""RoPE from first principles — derived, not copied.

This module builds Rotary Position Embeddings from the mathematical
constraint: q_m · k_n must depend only on content and relative position m-n.

No reference code was used. Every line follows from the derivation.
"""

from __future__ import annotations

import numpy as np


def rope_frequencies(dim: int, base: float = 10000.0) -> np.ndarray:
    """Compute rotation frequencies for each 2D subspace.

    Derivation:
        We need D/2 independent rotations. Subspace i rotates at frequency θ_i.
        Low i → fast rotation (sensitive to small position changes).
        High i → slow rotation (sensitive to large position changes).

        θ_i = 1 / base^(2i/D)

        Why this formula? It's a geometric progression in frequency space.
        - At i=0: θ_0 = 1 (fastest — period 2π)
        - At i=D/2-1: θ_{D/2-1} = 1/base^((D-2)/D) ≈ 1/base (slowest — period ~2π·base)

        The base controls the longest wavelength. For base=10000, the slowest
        frequency has period ~62831 tokens — enough for any practical sequence.

    Args:
        dim: Vector dimension (must be even).
        base: Base for frequency decay. Larger = longer wavelengths.

    Returns:
        Array of shape (D/2,) with rotation frequencies.
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")
    # θ_i = 1 / base^(2i/D) = base^(-2i/D)
    # Using exp for numerical stability: θ_i = exp(-2i/D · ln(base))
    i = np.arange(0, dim, 2, dtype=np.float32)  # 0, 2, 4, ..., D-2
    return np.exp(-i / dim * np.log(base))


def rope_rotation(
    pos: int, freqs: np.ndarray, seq_len: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cos and sin for a range of positions.

    Derivation:
        For position m, subspace i rotates by angle m·θ_i.
        We need cos(m·θ_i) and sin(m·θ_i) for each subspace.

        For a sequence of positions [pos, pos+1, ..., pos+seq_len-1]:
            angles[m, i] = (pos + m) · θ_i
            This is an outer product: t × freqs

    Args:
        pos: Starting position index (m).
        freqs: Rotation frequencies from rope_frequencies().
        seq_len: Number of positions to compute (default 1 for single token).

    Returns:
        (cos, sin) each of shape (seq_len, len(freqs)).
    """
    t = np.arange(pos, pos + seq_len, dtype=np.float32)
    angles = np.outer(t, freqs)  # (seq_len, D/2)
    return np.cos(angles), np.sin(angles)


def rope_apply(
    x: np.ndarray, pos: int, dim: int, base: float = 10000.0
) -> np.ndarray:
    """Apply rotary position embeddings to a vector.

    Derivation:
        Given x = [x_0, x_1, ..., x_{D-1}], split into:
            x1 = [x_0, x_1, ..., x_{D/2-1}]      (first half)
            x2 = [x_{D/2}, x_{D/2+1}, ..., x_{D-1}]  (second half)

        For each subspace i, apply 2D rotation:
            [x'_i      ]   [cos(m·θ_i)  -sin(m·θ_i)] [x_i      ]
            [x'_{i+D/2}] = [sin(m·θ_i)   cos(m·θ_i)] [x_{i+D/2}]

        Which gives:
            x'_i      =  x_i · cos(m·θ_i) - x_{i+D/2} · sin(m·θ_i)
            x'_{i+D/2} =  x_{i+D/2} · cos(m·θ_i) + x_i · sin(m·θ_i)

        This is equivalent to:
            x' = [x1·cos - x2·sin, x2·cos + x1·sin]

    Args:
        x: Input vector. Shape (..., D) where D is head_dim.
        pos: Starting position index.
        dim: Dimension of the last axis (head_dim).
        base: Base for frequency decay.

    Returns:
        Rotated vector, same shape as x.
    """
    seq_len = x.shape[0] if x.ndim >= 1 else 1
    freqs = rope_frequencies(dim, base)
    cos, sin = rope_rotation(pos, freqs, seq_len=seq_len)

    # Broadcast cos/sin to match x's shape
    # x can be (D,), (seq, D), or (seq, heads, D)
    if x.ndim == 1:
        # Single vector — squeeze seq_len dim from cos/sin
        cos = cos[0]  # (D/2,)
        sin = sin[0]
    elif x.ndim == 2:
        pass  # (seq, D/2) matches (seq, D)
    elif x.ndim == 3:
        cos = cos[:, np.newaxis, :]  # (seq, 1, D/2)
        sin = sin[:, np.newaxis, :]

    half = dim // 2
    x1 = x[..., :half]
    x2 = x[..., half:]

    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
