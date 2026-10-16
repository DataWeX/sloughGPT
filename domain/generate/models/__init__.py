"""models — Pluggable model backends.

Public API:
    ModelInterface, ModelLoader, SloughGPTModel
    RMSNorm, SloughGPTAttention, SloughGPTBlock, SwiGLU
    rotate_half, apply_rotary_pos_emb
    KnowledgeProcessor, ConsciousnessProcessor, apply_processors, get_provider, list_providers
"""

from __future__ import annotations

from domain.generate.models._internal.models import (
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
from domain.generate.models._internal.provider import (
    ConsciousnessProcessor,
    KnowledgeProcessor,
    apply_processors,
    attach_process_guard_to_provider,
    clear_providers,
    get_provider,
    list_providers,
    setup_providers,
    update_personality_traits,
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
    "attach_process_guard_to_provider",
    "clear_providers",
    "get_provider",
    "list_providers",
    "setup_providers",
    "update_personality_traits",
]
