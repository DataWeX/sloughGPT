"""VLMConfig — dataclass for vision-language model training configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VLMConfig:
    """Configuration for two-stage VLM training.

    Stage 1 trains only the vision→LLM connector projection while
    both the vision encoder and LLM are frozen. Stage 2 trains the
    connector + LoRA adapters on the LLM while the vision encoder
    stays frozen.
    """

    stage1_epochs: int = 1
    stage2_epochs: int = 2

    batch_size: int = 4
    stage1_lr: float = 1e-3
    stage2_lr: float = 2e-5
    lora_rank: int = 8
    lora_alpha: int = 16

    vision_encoder: str = "google/siglip-base-patch16-224"
    llm: str = "Qwen/Qwen2.5-0.5B-Instruct"

    max_seq_length: int = 256
    output_dir: str = "models/vlm-finetuned"

    freeze_vision: bool = True
    device: str = "cpu"
    seed: int = 42
