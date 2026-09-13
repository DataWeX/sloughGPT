"""Backward-compatibility shim."""
import domains.infrastructure.conversation_log as _mod
globals().update(vars(_mod))
