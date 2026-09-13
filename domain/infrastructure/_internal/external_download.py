"""Backward-compatibility shim."""
import domains.infrastructure.external_download as _mod
globals().update(vars(_mod))
