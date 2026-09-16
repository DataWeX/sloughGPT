"""Backward-compatibility shim."""

import domains.infrastructure.pugqeep.cache as _mod

globals().update(vars(_mod))
