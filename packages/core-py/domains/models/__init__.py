"""Backward-compatibility shim — imports from the new ``domain.models`` package."""

from domain.models import (
    ModelInterface,
    ModelLoader,
    SloughGPTModel,
    apply_rotary_pos_emb,
    rotate_half,
)

__all__ = [
    "ModelInterface",
    "ModelLoader",
    "SloughGPTModel",
    "rotate_half",
    "apply_rotary_pos_emb",
]
