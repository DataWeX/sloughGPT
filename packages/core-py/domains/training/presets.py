"""Training Presets — ready-made training configurations for common scenarios.

Provides curated presets for quick-start training with sensible defaults.
Users can select a preset and customize from there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrainingPreset:
    """A named training configuration preset."""
    name: str
    description: str
    model: str
    method: str
    epochs: int
    batch_size: int
    learning_rate: float
    max_seq_length: int
    warmup_steps: int
    weight_decay: float
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "model": self.model,
            "method": self.method,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_seq_length": self.max_seq_length,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "use_lora": self.use_lora,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "tags": self.tags,
        }


# ── Built-in Presets ──────────────────────────────────────────────

PRESETS: Dict[str, TrainingPreset] = {
    "quick-finetune": TrainingPreset(
        name="Quick Fine-Tune",
        description="Fast fine-tuning for small datasets. Good for prototyping.",
        model="gpt2",
        method="finetune",
        epochs=3,
        batch_size=8,
        learning_rate=5e-5,
        max_seq_length=512,
        warmup_steps=50,
        weight_decay=0.01,
        tags=["quick", "beginner"],
    ),
    "lora-adapter": TrainingPreset(
        name="LoRA Adapter",
        description="Parameter-efficient fine-tuning using LoRA. Ideal for limited GPU memory.",
        model="gpt2",
        method="lora",
        epochs=5,
        batch_size=4,
        learning_rate=3e-4,
        max_seq_length=1024,
        warmup_steps=100,
        weight_decay=0.01,
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        tags=["lora", "efficient", "low-memory"],
    ),
    "full-training": TrainingPreset(
        name="Full Training",
        description="Full parameter training for large datasets. Best quality but needs GPU.",
        model="gpt2",
        method="finetune",
        epochs=10,
        batch_size=16,
        learning_rate=2e-5,
        max_seq_length=1024,
        warmup_steps=200,
        weight_decay=0.05,
        tags=["full", "high-quality", "gpu"],
    ),
    "chat-finetune": TrainingPreset(
        name="Chat Fine-Tune",
        description="Fine-tune for conversational AI. Uses chat-specific training data.",
        model="gpt2",
        method="chat",
        epochs=5,
        batch_size=4,
        learning_rate=2e-5,
        max_seq_length=2048,
        warmup_steps=100,
        weight_decay=0.01,
        tags=["chat", "conversational"],
    ),
    "distillation": TrainingPreset(
        name="Knowledge Distillation",
        description="Distill knowledge from a larger model to a smaller one.",
        model="gpt2",
        method="distill",
        epochs=5,
        batch_size=8,
        learning_rate=5e-5,
        max_seq_length=512,
        warmup_steps=50,
        weight_decay=0.01,
        tags=["distillation", "compression"],
    ),
    "rlhf-alignment": TrainingPreset(
        name="RLHF Alignment",
        description="Reinforcement Learning from Human Feedback for alignment.",
        model="gpt2",
        method="rlhf",
        epochs=3,
        batch_size=4,
        learning_rate=1e-5,
        max_seq_length=1024,
        warmup_steps=100,
        weight_decay=0.01,
        tags=["rlhf", "alignment", "advanced"],
    ),
    "code-generation": TrainingPreset(
        name="Code Generation",
        description="Fine-tune for code generation tasks. Optimized for programming data.",
        model="gpt2",
        method="finetune",
        epochs=8,
        batch_size=8,
        learning_rate=3e-5,
        max_seq_length=2048,
        warmup_steps=150,
        weight_decay=0.02,
        tags=["code", "programming"],
    ),
    "low-resource": TrainingPreset(
        name="Low Resource",
        description="Optimized for machines with limited RAM/GPU. Smaller batch sizes.",
        model="gpt2",
        method="finetune",
        epochs=3,
        batch_size=2,
        learning_rate=5e-5,
        max_seq_length=256,
        warmup_steps=20,
        weight_decay=0.01,
        tags=["low-resource", "cpu", "small"],
    ),
}


def list_presets() -> List[Dict[str, Any]]:
    """List all available presets."""
    return [p.to_dict() for p in PRESETS.values()]


def get_preset(name: str) -> Optional[Dict[str, Any]]:
    """Get a preset by name."""
    preset = PRESETS.get(name)
    return preset.to_dict() if preset else None


def apply_preset(name: str) -> Optional[Dict[str, Any]]:
    """Get a preset config ready to be applied to training settings."""
    preset = PRESETS.get(name)
    if not preset:
        return None
    return {
        "preferred_model": preset.model,
        "preferred_method": preset.method,
        "default_epochs": preset.epochs,
        "default_batch_size": preset.batch_size,
        "default_learning_rate": preset.learning_rate,
        "default_max_seq_length": preset.max_seq_length,
        "default_warmup_steps": preset.warmup_steps,
        "default_weight_decay": preset.weight_decay,
        "use_lora": preset.use_lora,
        "default_lora_rank": preset.lora_rank,
        "default_lora_alpha": preset.lora_alpha,
    }


__all__ = [
    "TrainingPreset",
    "PRESETS",
    "list_presets",
    "get_preset",
    "apply_preset",
]
