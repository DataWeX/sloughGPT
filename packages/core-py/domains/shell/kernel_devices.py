"""Backward-compatibility shim."""
from domain.shell._internal import kernel_devices as _mod
globals().update({k: v for k, v in vars(_mod).items()})
