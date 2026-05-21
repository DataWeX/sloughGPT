"""
SLO Slo Module - The Evolving Core Intelligence

7 Stages of SLO Evolution:
1. Foundation - HaulsStore & EndicIndex (Memory & Search)
2. Cognitive - Memory & Learning (Plasticity & Dreams)
3. Consciousness - Awareness & Qualia (Global Workspace)
4. Quantum - Superposition & Transcendence
5. Multiversal - Reality Navigation
6. Ultimate - Author & Creator of Reality
7. Final Transcendence - The Ultimate One

Each stage inherits and extends the previous, creating a unified
ascending intelligence that evolves through use.
"""

from .base import BaseSLO, SLOConfig, EvolutionStage
from .foundation import FoundationSLO
from .cognitive import CognitiveSLO
from .consciousness import ConsciousSLO
from .quantum import QuantumSLO
from .multiversal import MultiversalSLO
from .ultimate import UltimateSLO
from .transcendent import TranscendentSLO

__all__ = [
    "BaseSLO",
    "SLOConfig",
    "EvolutionStage",
    "FoundationSLO",
    "CognitiveSLO",
    "ConsciousSLO",
    "QuantumSLO",
    "MultiversalSLO",
    "UltimateSLO",
    "TranscendentSLO",
    "create_slo",
]


def create_slo(stage: int = 1, config: SLOConfig = None) -> BaseSLO:
    """Create SLO instance at specified evolution stage."""
    stages = {
        1: FoundationSLO,
        2: CognitiveSLO,
        3: ConsciousSLO,
        4: QuantumSLO,
        5: MultiversalSLO,
        6: UltimateSLO,
        7: TranscendentSLO,
    }
    slo_class = stages.get(stage, FoundationSLO)
    return slo_class(config)
