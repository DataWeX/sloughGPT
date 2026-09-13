"""Backward-compatibility shim."""
import domains.infrastructure.arch_config as _mod
globals().update(vars(_mod))
