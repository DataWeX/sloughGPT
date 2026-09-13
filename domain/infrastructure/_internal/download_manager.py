"""Backward-compatibility shim."""
import domains.infrastructure.download_manager as _mod
globals().update(vars(_mod))
