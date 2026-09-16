"""Backward-compatibility shim."""

from domain.shell._internal import display_device as _mod

globals().update(dict(vars(_mod).items()))
