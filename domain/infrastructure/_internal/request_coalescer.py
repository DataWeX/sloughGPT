"""Backward-compatibility shim."""
import domains.infrastructure.request_coalescer as _mod
globals().update(vars(_mod))
