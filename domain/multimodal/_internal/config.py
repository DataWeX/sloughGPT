from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MultiModalConfig:
    """Configuration for multi-modal models."""

    image_size: int = 224
    patch_size: int = 16
    vision_hidden_size: int = 768
    vision_num_layers: int = 12
    vision_num_heads: int = 12
    vocab_size: int = 50257
    text_hidden_size: int = 768
    text_num_layers: int = 12
    text_num_heads: int = 12
    max_seq_length: int = 512
    fusion_type: str = "cross_attention"
    projection_dim: int = 768
