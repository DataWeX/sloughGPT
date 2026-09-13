"""Backward-compatibility shim."""
import domains.infrastructure.producer_consumer as _mod
globals().update(vars(_mod))
