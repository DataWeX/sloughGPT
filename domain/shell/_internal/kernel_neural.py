from __future__ import annotations

"""
Neural Kernel — re-export shim.

Canonical source is addons.neural. This module exists for backward compatibility.

Migration:
    from domain.shell._internal.addons.neural import NeuralProcess, NeuralKVCache, ...
"""

from .addons.neural import (  # noqa: F401
    BatchProcessor,
    BatchRequest,
    BatchResult,
    CacheStrategy,
    EmbeddingEntry,
    EmbeddingStoreDevice,
    GradientAccumulator,
    KVCacheEntry,
    MultiHeadAttentionDevice,
    NeuralEmbeddingStore,
    NeuralEngineDevice,
    NeuralInterrupt,
    NeuralKernel,
    NeuralKVCache,
    NeuralMemoryType,
    NeuralOp,
    NeuralProcess,
    NeuralProcessType,
    NeuralState,
    NeuralSyscall,
    TokenizerDevice,
)
