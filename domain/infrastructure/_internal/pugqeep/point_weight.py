"""Backward-compatibility shim."""

import domains.infrastructure.pugqeep.point_weight as _mod

globals().update(vars(_mod))
