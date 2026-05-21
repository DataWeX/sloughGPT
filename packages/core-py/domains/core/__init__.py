"""
domains/core/ - Slo Core Architecture

SloEngine is THE core model wrapper. All inference flows through here.
Cognitive and reasoning engines are first-class citizens, built INTO the soul.
"""

from .soul import SloEngine, GenerationContext

__all__ = ["SloEngine", "GenerationContext"]
