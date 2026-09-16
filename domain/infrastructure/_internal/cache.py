"""Backward-compatibility shim."""

import domains.infrastructure.cache as _mod

globals().update(vars(_mod))
