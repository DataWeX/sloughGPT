"""
AI-Native Neural Kernel — DEPRECATED.

This module is now a re-export shim. The unified Kernel class lives in kernel.py.

Migration:
    from domains.shell.kernel import Kernel
"""
from .kernel import (  # noqa: F401
    Kernel as NeuralKernel,  # backward compat alias
    NeuralProcess, NeuralProcessType, NeuralState, NeuralOp,
    NeuralMemoryType, CacheStrategy, NeuralKVCache, NeuralEmbeddingStore,
    NeuralEngineDevice, TokenizerDevice, EmbeddingStoreDevice,
    NeuralInterrupt, NeuralSyscall, GradientAccumulator, BatchProcessor,
    BatchRequest, BatchResult, MultiHeadAttentionDevice,
)
