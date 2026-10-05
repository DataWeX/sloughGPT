"""models — Pluggable model backends.

Public API:
    ModelInterface, ModelLoader, SloughGPTModel
    RMSNorm, SloughGPTAttention, SloughGPTBlock, SwiGLU
    rotate_half, apply_rotary_pos_emb
    KnowledgeProcessor, ConsciousnessProcessor, apply_processors, get_provider, list_providers
"""

from domain.models._internal.models import (
    ModelInterface,
    ModelLoader,
    RMSNorm,
    SloughGPTAttention,
    SloughGPTBlock,
    SloughGPTModel,
    SwiGLU,
    apply_rotary_pos_emb,
    rotate_half,
)
from domain.models._internal.provider import (
    ConsciousnessProcessor,
    KnowledgeProcessor,
    apply_processors,
    get_provider,
    list_providers,
)

__all__ = [
    "ModelInterface",
    "ModelLoader",
    "SloughGPTModel",
    "RMSNorm",
    "SloughGPTAttention",
    "SloughGPTBlock",
    "SwiGLU",
    "rotate_half",
    "apply_rotary_pos_emb",
    "KnowledgeProcessor",
    "ConsciousnessProcessor",
    "apply_processors",
    "get_provider",
    "list_providers",
]
