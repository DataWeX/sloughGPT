"""Backward-compatibility shim."""

import domains.infrastructure.pugqeep.task_queue as _mod

globals().update(vars(_mod))
