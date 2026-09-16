"""Backward-compatibility shim."""

from domain.shell._internal import interactive as _mod

globals().update(dict(vars(_mod).items()))
