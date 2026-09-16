"""Backward-compatibility shim."""

import domains.infrastructure.pugqeep.executor as _mod

globals().update(vars(_mod))
