"""Backward-compatibility shim."""
import domains.infrastructure.compute_backend as _mod
globals().update(vars(_mod))
