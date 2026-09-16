"""Backward-compatibility shim."""

from domain.shell._internal import storage_device as _mod

globals().update(dict(vars(_mod).items()))
