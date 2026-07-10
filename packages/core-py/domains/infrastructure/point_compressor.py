"""
Point compressor — stores weights as functions, not raw values.

Backward-compatible shim — all code lives in pugqeep/ package.
"""

from domains.infrastructure.pugqeep import (
    Point,
    PointCompressor,
    PointLibrary,
    ModelTree,
    PointDeduplicator,
    PointLibrarySync,
    load_model_to_points,
    PGQ,
)

# Backward compat aliases
PointLib = PGQ
save_library = lambda lib, path: lib.save(path)
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
