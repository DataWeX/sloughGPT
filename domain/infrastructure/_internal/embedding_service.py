"""Backward-compatibility shim."""
import domains.infrastructure.embedding_service as _mod
globals().update(vars(_mod))
