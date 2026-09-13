"""Backward-compatibility shim."""
import domains.infrastructure.local_download as _mod
globals().update(vars(_mod))
