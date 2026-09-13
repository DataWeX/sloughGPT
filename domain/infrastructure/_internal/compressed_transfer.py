"""Backward-compatibility shim."""
import domains.infrastructure.compressed_transfer as _mod
globals().update(vars(_mod))
