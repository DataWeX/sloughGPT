from __future__ import annotations

"""
Neural Kernel — re-export shim.

Canonical source is addons.neural. This module exists for backward compatibility.

Migration:
    from domains.shell.addons.neural import NeuralProcess, NeuralKVCache, ...
"""

from .addons.neural import (  # noqa: F401
    NeuralOp, NeuralState, NeuralProcessType, NeuralMemoryType, CacheStrategy,
    NeuralProcess, KVCacheEntry, NeuralKVCache,
    EmbeddingEntry, NeuralEmbeddingStore,
    NeuralEngineDevice, TokenizerDevice, EmbeddingStoreDevice,
    MultiHeadAttentionDevice,
    NeuralInterrupt, NeuralSyscall,
    GradientAccumulator, BatchRequest, BatchResult, BatchProcessor,
    NeuralKernel,
)
