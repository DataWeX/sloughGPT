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
    list_providers,
    update_personality_traits,
)

# Names the API layer reads are resolved lazily so a test patch on
# domain.models._internal.provider.<name> stays visible at call time
# (eager binding would freeze the pre-patch object — AGENT_SYNC gotcha).
_LAZY_IMPORTS = {
    "get_provider": ("._internal.provider", "get_provider"),
    "setup_providers": ("._internal.provider", "setup_providers"),
    "attach_process_guard_to_provider": (
        "._internal.provider",
        "attach_process_guard_to_provider",
    ),
    "clear_providers": ("._internal.provider", "clear_providers"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "setup_providers",
    "attach_process_guard_to_provider",
    "clear_providers",
]
