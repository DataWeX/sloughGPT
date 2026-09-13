"""Backward-compatibility shim."""
import domains.infrastructure.pugqeep.queue as _mod
globals().update(vars(_mod))
