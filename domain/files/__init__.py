"""files — compressed blob store + pull-through hub (sync cores, standalone).

Public API:
    FileStore — put/get/list/delete with per-class compress-at-rest:
        text/code → zstd (stored frame doubles as wire format),
        model weights → raw (proven incompressible), unknown → zstd-if-smaller.
    FileHub — pull-through cache over FileStore: downcraft-backed pull
        (resume + checksum), sha256 dedup, constant-time token auth.
"""
from __future__ import annotations

from domain.files._internal.hub import FileHub
from domain.files._internal.quant_ingest import (
    QuantResult,
    UnsupportedForQuantization,
    load_quantized,
    quantize_safetensors,
)
from domain.files._internal.store import (
    DEFAULT_ROOT,
    MODEL_EXTENSIONS,
    TEXT_EXTENSIONS,
    FileStore,
)

__all__ = [
    "FileStore",
    "FileHub",
    "DEFAULT_ROOT",
    "MODEL_EXTENSIONS",
    "TEXT_EXTENSIONS",
    "QuantResult",
    "UnsupportedForQuantization",
    "load_quantized",
    "quantize_safetensors",
]
