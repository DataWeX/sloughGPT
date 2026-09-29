"""models — Pluggable model backends.

Public API:
    ModelInterface, ModelLoader, SloughGPTModel
    rotate_half, apply_rotary_pos_emb
    KnowledgeProcessor, ConsciousnessProcessor, apply_processors, get_provider, list_providers
    setup_providers, attach_process_guard_to_provider, clear_providers
"""

from domain.models._internal.models import (
    ModelInterface,
    ModelLoader,
    SloughGPTModel,
    apply_rotary_pos_emb,
    rotate_half,
)
from domain.models._internal.provider import (
    ConsciousnessProcessor,
    KnowledgeProcessor,
    apply_processors,
    attach_process_guard_to_provider,
    clear_providers,
    get_provider,
    list_providers,
    setup_providers,
)

__all__ = [
    "ModelInterface",
    "ModelLoader",
    "SloughGPTModel",
    "rotate_half",
    "apply_rotary_pos_emb",
    "KnowledgeProcessor",
    "ConsciousnessProcessor",
    "apply_processors",
    "get_provider",
    "list_providers",
    "setup_providers",
    "attach_process_guard_to_provider",
    "clear_providers",
]
