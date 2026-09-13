"""Backward-compatibility shim."""
from domain.shell._internal import commands as _mod
globals().update({k: v for k, v in vars(_mod).items()})
