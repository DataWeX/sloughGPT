"""Backward-compatibility shim."""

import domains.infrastructure.pugqeep.store as _mod

globals().update(vars(_mod))
