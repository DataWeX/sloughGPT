"""Backward-compatibility shim — imports from the new ``domain.ops`` package."""

from domain.ops import (
    ChunkedOperation,
    FusedAttentionBias,
    FusedCrossEntropyLoss,
    FusedLayerNorm,
    FusedRMSNorm,
    FusedScaleBias,
    MemoryEfficientSoftmax,
    OptimizedEmbedding,
    chunked_matmul,
    efficient_cross_entropy,
    estimate_attention_memory,
    fused_swiglu,
    gelu,
    ragged_to_padded,
    silu,
)

__all__ = [
    "FusedLayerNorm",
    "FusedRMSNorm",
    "FusedCrossEntropyLoss",
    "FusedAttentionBias",
    "ChunkedOperation",
    "MemoryEfficientSoftmax",
    "FusedScaleBias",
    "OptimizedEmbedding",
    "fused_swiglu",
    "efficient_cross_entropy",
    "chunked_matmul",
    "ragged_to_padded",
    "estimate_attention_memory",
    "silu",
    "gelu",
]
