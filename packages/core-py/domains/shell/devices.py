"""Backward-compatibility shim."""

from domain.shell._internal import devices as _mod

globals().update(dict(vars(_mod).items()))
