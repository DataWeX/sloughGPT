"""Backward-compatibility — allows ``from domain.memory._internal.config import ...``."""
from domain.memory._internal.config import MemoryConfig

__all__ = ["MemoryConfig"]