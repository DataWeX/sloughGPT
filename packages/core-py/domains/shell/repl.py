"""Backward-compatibility shim."""
from domain.shell._internal import repl as _mod
globals().update({k: v for k, v in vars(_mod).items()})
