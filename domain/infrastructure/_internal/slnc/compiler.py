"""Backward-compatibility shim."""

import domains.infrastructure.slnc.compiler as _mod

globals().update(vars(_mod))
