"""models — Pluggable model backends.

Public API:
    ModelInterface, ModelLoader, SloughGPTModel
    rotate_half, apply_rotary_pos_emb
    KnowledgeProcessor, apply_processors, get_provider, list_providers
"""

from domain.models._internal.models import (
    ModelInterface,
    ModelLoader,
    SloughGPTModel,
    apply_rotary_pos_emb,
    rotate_half,
)
from domain.models._internal.provider import (
    KnowledgeProcessor,
    apply_processors,
    get_provider,
    list_providers,
    update_personality_traits,
)

__all__ = [
    "ModelInterface",
    "ModelLoader",
    "SloughGPTModel",
    "rotate_half",
    "apply_rotary_pos_emb",
    "KnowledgeProcessor",
    "apply_processors",
    "get_provider",
    "list_providers",
    "update_personality_traits",
]
