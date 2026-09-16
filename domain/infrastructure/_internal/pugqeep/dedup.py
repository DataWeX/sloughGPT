"""Backward-compatibility shim."""

import domains.infrastructure.pugqeep.dedup as _mod

globals().update(vars(_mod))
