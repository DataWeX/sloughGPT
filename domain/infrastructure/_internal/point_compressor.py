"""
Point compressor — stores weights as functions, not raw values.

Backward-compatible shim — all code lives in pugqeep/ package.
"""

from __future__ import annotations

from .pugqeep import (
    PGQ,
    ModelTree,
    Point,
    PointCompressor,
    PointDeduplicator,
    PointLibrary,
    PointLibrarySync,
    load_model_to_points,
)

# Backward compat aliases
PointLib = PGQ
def save_library(lib, path):
    return lib.save(path)
load_library = PGQ.load

__all__ = [
    "Point",
    "PointCompressor",
    "PointLibrary",
    "ModelTree",
    "PointDeduplicator",
    "PointLibrarySync",
    "load_model_to_points",
    "save_library",
    "load_library",
    "PGQ",
    "PointLib",
]
