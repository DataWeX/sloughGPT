"""ops — Optimized NumPy operations (fused ops, attention, softmax).

Public API:
    FusedLayerNorm, FusedRMSNorm, FusedCrossEntropyLoss, FusedAttentionBias
    ChunkedOperation, MemoryEfficientSoftmax, FusedScaleBias, OptimizedEmbedding
    fused_swiglu, efficient_cross_entropy, chunked_matmul, silu, gelu
"""

from domain.ops._internal.ops import (
    FusedLayerNorm,
    FusedRMSNorm,
    FusedCrossEntropyLoss,
    FusedAttentionBias,
    ChunkedOperation,
    MemoryEfficientSoftmax,
    FusedScaleBias,
    OptimizedEmbedding,
    fused_swiglu,
    efficient_cross_entropy,
    chunked_matmul,
    ragged_to_padded,
    estimate_attention_memory,
    silu,
    gelu,
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
