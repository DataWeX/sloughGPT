"""Backward-compatibility shim."""
import domains.infrastructure.auto_ingest as _mod
globals().update(vars(_mod))
