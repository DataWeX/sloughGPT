"""Backward-compatibility shim."""
import domains.infrastructure.entity_repositories as _mod
globals().update(vars(_mod))
