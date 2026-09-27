"""Canonical model dimension helpers — single source of truth.

``align_dim_ff`` is THE storage contract between model construction and
weight loading: SloTransformer rounds the feed-forward dim up to a multiple
of 64 (SIMD alignment) when allocating FFN buffers, and the weight loader
must pad checkpoint tensors to exactly that rounded size. Both sides import
this one function — never re-implement the rounding inline.
"""

from __future__ import annotations

FF_ALIGN = 64


def align_dim_ff(dim: int) -> int:
    """Round a feed-forward dim up to the next multiple of ``FF_ALIGN``.

    Args:
        dim: Raw ``intermediate_size`` (or ``n_inner``) from model config.

    Returns:
        Aligned dim (>= input); multiples of 64 pass through unchanged.
    """
    return ((dim + FF_ALIGN - 1) // FF_ALIGN) * FF_ALIGN
