"""Backward-compatibility shim."""
import domains.infrastructure.slonet_server as _mod
globals().update(vars(_mod))
