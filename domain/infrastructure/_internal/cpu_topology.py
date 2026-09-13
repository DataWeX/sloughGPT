"""Backward-compatibility shim."""
import domains.infrastructure.cpu_topology as _mod
globals().update(vars(_mod))
