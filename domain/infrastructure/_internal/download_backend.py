"""Backward-compatibility shim."""
import domains.infrastructure.download_backend as _mod
globals().update(vars(_mod))
