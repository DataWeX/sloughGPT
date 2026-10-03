from __future__ import annotations

"""
Neural Kernel — re-export shim.

Canonical source is addons.neural. This module exists for backward compatibility.

Migration:
    from domain.shell._internal.addons.neural import NeuralProcess, NeuralKVCache, ...
"""

import importlib as _importlib

_REEXPORTS = (
    "BatchProcessor",
    "BatchRequest",
    "BatchResult",
    "CacheStrategy",
    "EmbeddingEntry",
    "EmbeddingStoreDevice",
    "GradientAccumulator",
    "KVCacheEntry",
    "MultiHeadAttentionDevice",
    "NeuralEmbeddingStore",
    "NeuralEngineDevice",
    "NeuralInterrupt",
    "NeuralKernel",
    "NeuralKVCache",
    "NeuralMemoryType",
    "NeuralOp",
    "NeuralProcess",
    "NeuralProcessType",
    "NeuralState",
    "NeuralSyscall",
    "TokenizerDevice",
)


def __getattr__(name: str):
    """Lazily resolve re-exported names from the optional addons package.

    ``addons`` is a separately-vendored extension (historically shipped as a
    symlink).  Resolving lazily keeps this shim importable when the addons
    package is missing or broken; the ImportError surfaces only when a
    re-exported name is actually used.
    """
    if name not in _REEXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        _neural = _importlib.import_module(".addons.neural", __package__)
    except Exception as exc:
        raise ImportError(
            "neural kernel addon unavailable: addons.neural is missing or broken"
        ) from exc
    value = getattr(_neural, name, None)
    if value is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals()) + list(_REEXPORTS))
