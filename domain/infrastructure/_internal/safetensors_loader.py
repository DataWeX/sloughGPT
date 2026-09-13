"""Backward-compatibility shim."""
import domains.infrastructure.safetensors_loader as _mod
globals().update(vars(_mod))
