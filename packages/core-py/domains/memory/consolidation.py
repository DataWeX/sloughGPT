"""Backward-compatibility — allows ``from domain.memory._internal.consolidation import ...``."""
from domain.memory._internal.consolidation import plan_consolidation

__all__ = ["plan_consolidation"]