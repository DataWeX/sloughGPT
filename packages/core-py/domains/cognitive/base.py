"""Backward-compatibility shim — re-exports from domain.cognitive._internal.base."""

from domain.cognitive._internal.base import CognitiveDomain, CognitiveException

__all__ = ["CognitiveDomain", "CognitiveException"]
