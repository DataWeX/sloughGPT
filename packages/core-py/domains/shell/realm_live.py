"""Backward-compatibility shim."""

from domain.shell._internal import realm_live as _mod

globals().update(dict(vars(_mod).items()))
