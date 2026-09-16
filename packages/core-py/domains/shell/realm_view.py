"""Backward-compatibility shim."""

from domain.shell._internal import realm_view as _mod

globals().update(dict(vars(_mod).items()))
