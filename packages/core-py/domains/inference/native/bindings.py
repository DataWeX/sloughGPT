"""
bindings.py - Python ctypes bridge to libtransformer_forward.dylib.

Loads the compiled C library and exposes:
  - transformer_load_weights()
  - transformer_forward_step()
  - transformer_kv_cache_init/reset/free()
"""

import ctypes
import os
from pathlib import Path
import numpy as np

_LIB = None

def _find_lib():
    global _LIB
    if _LIB is not None:
        return _LIB

    candidates = [
        Path(__file__).parent / "libtransformer_forward.dylib",
        Path(__file__).parent / "libtransformer_forward.so",
    ]
    env_path = os.environ.get("MAN_TRANSFORMER_LIB", "")
    if env_path:
        candidates.append(Path(env_path))

    for p in candidates:
        if p.exists():
            try:
                _LIB = ctypes.CDLL(str(p))
                _setup_signatures(_LIB)
                return _LIB
            except OSError:
                continue

    raise RuntimeError(  # pragma: no cover — lib ships alongside this module
        "libtransformer_forward.dylib not found. Compile with:\n"
        "  cc -O3 -march=native -dynamiclib -framework Accelerate "
        "-o libtransformer_forward.dylib transformer_forward.c"
    )


def _setup_signatures(lib):
    """Set argtypes/restype for all exported functions."""
    class TransformerConfig(ctypes.Structure):
        _fields_ = [
            ("n_layers", ctypes.c_int),
            ("hidden_dim", ctypes.c_int),
            ("n_heads", ctypes.c_int),
            ("n_kv_heads", ctypes.c_int),
            ("head_dim", ctypes.c_int),
            ("ff_dim", ctypes.c_int),
            ("vocab_size", ctypes.c_int),
            ("block_size", ctypes.c_int),
            ("rope_base", ctypes.c_float),
            ("rope_theta", ctypes.c_float),
        ]

    class TransformerWeights(ctypes.Structure):
        _fields_ = [
            ("data", ctypes.POINTER(ctypes.c_float)),
            ("total_floats", ctypes.c_int),
            ("config", TransformerConfig),
            ("tok_emb_offset", ctypes.c_int),
            ("layer_offsets", ctypes.c_int * 64),
            ("norm_offset", ctypes.c_int),
            ("lm_head_offset", ctypes.c_int),
        ]

    class TransformerKVCache(ctypes.Structure):
        _fields_ = [
            ("k", ctypes.POINTER(ctypes.c_float)),
            ("v", ctypes.POINTER(ctypes.c_float)),
            ("n_layers", ctypes.c_int),
            ("n_kv_heads", ctypes.c_int),
            ("head_dim", ctypes.c_int),
            ("seq_capacity", ctypes.c_int),
            ("seq_len", ctypes.c_int),
        ]

    lib._Config = TransformerConfig
    lib._Weights = TransformerWeights
    lib._KVCache = TransformerKVCache

    lib.transformer_load_weights.argtypes = [
        ctypes.POINTER(TransformerWeights),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(TransformerConfig),
    ]
    lib.transformer_load_weights.restype = ctypes.c_int

    lib.transformer_free_weights.argtypes = [ctypes.POINTER(TransformerWeights)]
    lib.transformer_free_weights.restype = None

    lib.transformer_kv_cache_init.argtypes = [
        ctypes.POINTER(TransformerKVCache),
        ctypes.POINTER(TransformerConfig),
        ctypes.c_int,
    ]
    lib.transformer_kv_cache_init.restype = ctypes.c_int

    lib.transformer_kv_cache_free.argtypes = [ctypes.POINTER(TransformerKVCache)]
    lib.transformer_kv_cache_free.restype = None

    lib.transformer_kv_cache_reset.argtypes = [ctypes.POINTER(TransformerKVCache)]
    lib.transformer_kv_cache_reset.restype = None

    lib.transformer_forward_step.argtypes = [
        ctypes.POINTER(TransformerWeights),
        ctypes.POINTER(TransformerKVCache),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.transformer_forward_step.restype = ctypes.c_int


def load_lib():
    """Return the loaded C library singleton."""
    return _find_lib()
