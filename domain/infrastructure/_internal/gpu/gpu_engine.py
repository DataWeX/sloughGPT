"""Backward-compatibility shim."""
import domains.infrastructure.gpu.gpu_engine as _mod
globals().update(vars(_mod))
