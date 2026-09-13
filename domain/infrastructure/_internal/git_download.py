"""Backward-compatibility shim."""
import domains.infrastructure.git_download as _mod
globals().update(vars(_mod))
