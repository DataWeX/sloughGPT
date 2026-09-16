"""Backward-compatibility shim."""

from domain.shell._internal import audit as _mod

globals().update(dict(vars(_mod).items()))
