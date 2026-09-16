"""Backward-compatibility shim."""

from domain.shell._internal import cycles_device as _mod

globals().update(dict(vars(_mod).items()))
