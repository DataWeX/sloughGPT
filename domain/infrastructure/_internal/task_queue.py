"""Backward-compatibility shim."""
import domains.infrastructure.task_queue as _mod
globals().update(vars(_mod))
