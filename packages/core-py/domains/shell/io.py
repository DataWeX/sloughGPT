"""Backward-compatibility shim."""

from domain.shell._internal import io as _mod

globals().update(dict(vars(_mod).items()))
