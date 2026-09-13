"""Backward-compatibility shim."""
from domain.shell._internal import device_system as _mod
globals().update({k: v for k, v in vars(_mod).items()})
