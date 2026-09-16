"""Backward-compatibility shim."""

from domain.shell._internal import render_neural as _mod

globals().update(dict(vars(_mod).items()))
