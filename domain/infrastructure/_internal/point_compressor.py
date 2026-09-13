"""Backward-compatibility shim."""
import domains.infrastructure.point_compressor as _mod
globals().update(vars(_mod))
