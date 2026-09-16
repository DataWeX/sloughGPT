"""Backward-compatibility shim."""

from domain.shell._internal import device_decorators as _mod

globals().update(dict(vars(_mod).items()))
