"""Backward-compatibility shim."""
import domains.infrastructure.gpu.compile_shaders as _mod
globals().update(vars(_mod))
