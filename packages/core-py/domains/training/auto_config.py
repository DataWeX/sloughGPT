"""Auto-configuration engine for training.

Reads a dataset, analyses it, and produces a complete training config
that works without the user needing to understand any ML parameters.

Design principle: the user says "train on this data", the backend
figures out the rest.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("man.training.auto_config")


@dataclass
class DatasetAnalysis:
    """What we know about a dataset after inspection."""

    path: str
    format: str  # "text", "messages", "jsonl", "unknown"
    sample_count: int = 0
    char_count: int = 0
    word_count: int = 0
    avg_line_length: float = 0.0
    has_dialogue_markers: bool = False
    has_role_fields: bool = False
    preview_lines: List[str] = field(default_factory=list)

    @property
    def is_dialogue(self) -> bool:
        return self.has_dialogue_markers or self.has_role_fields

    @property
    def is_messages_format(self) -> bool:
        return self.format == "messages"

    @property
    def size_category(self) -> str:
        """tiny | small | medium | large"""
        if self.word_count < 1000:
            return "tiny"
        if self.word_count < 10_000:
            return "small"
        if self.word_count < 100_000:
            return "medium"
        return "large"


@dataclass
class TrainingConfig:
    """Complete training configuration — everything the trainer needs."""

    # What to train
    dataset: str = ""
    data_path: str = ""
    model: str = "gpt2"

    # How to train
    method: str = "finetune"  # "distill" | "finetune" | "vlm"
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    warmup_steps: int = 50
    weight_decay: float = 0.01

    # LoRA (auto-enabled for fine-tune)
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16

    # RL post-training (auto-enabled for chat models)
    rl_post_train: bool = False
    rl_num_generations: int = 4
    rl_learning_rate: float = 1e-6
    rl_kl_coef: float = 0.1
    rl_reward_mode: str = "length"

    # What we discovered
    analysis: Optional[DatasetAnalysis] = None

    # Plain-language explanation of what we chose and why
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable dict for the training router."""
        return {
            "model": self.model,
            "dataset": self.dataset,
            "data_path": self.data_path,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_seq_length": self.max_seq_length,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "use_lora": self.use_lora,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "rl_post_train": self.rl_post_train,
            "rl_num_generations": self.rl_num_generations,
            "rl_learning_rate": self.rl_learning_rate,
            "rl_kl_coef": self.rl_kl_coef,
            "rl_reward_mode": self.rl_reward_mode,
        }


def analyse_dataset(dataset_path: str) -> DatasetAnalysis:
    """Read a dataset file and extract statistics.

    Supports:
    - Plain text files (input.txt)
    - JSONL with ``{"messages": [...]}`` or ``{"text": "..."}`` lines
    - Detects dialogue markers (``User:``, ``Assistant:``, ``Human:``)
    """
    path = Path(dataset_path)
    if not path.exists():
        return DatasetAnalysis(path=dataset_path, format="unknown")

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    non_empty = [l.strip() for l in lines if l.strip()]

    # Detect format
    fmt = "text"
    has_role_fields = False
    sample_count = 0

    if path.suffix == ".jsonl" or (non_empty and _is_json(non_empty[0])):
        fmt = "jsonl"
        messages_count = 0
        text_count = 0
        for line in non_empty[:100]:  # sample first 100 lines
            try:
                obj = json.loads(line)
                if "messages" in obj or "conversations" in obj:
                    messages_count += 1
                    has_role_fields = True
                elif "text" in obj or "content" in obj:
                    text_count += 1
            except (json.JSONDecodeError, TypeError):
                pass
        sample_count = len(non_empty)
        if messages_count > text_count:
            fmt = "messages"
    else:
        sample_count = len(non_empty)

    # Detect dialogue markers
    dialogue_markers = ["user:", "assistant:", "human:", "ai:", "system:", "<|user|>", "<|assistant|>"]
    marker_hits = 0
    for line in non_empty[:200]:
        lower = line.lower().strip()
        if any(lower.startswith(m) or f" {m}" in lower for m in dialogue_markers):
            marker_hits += 1
    has_dialogue_markers = marker_hits > len(non_empty[:200]) * 0.1  # >10% of lines

    word_count = len(text.split())
    char_count = len(text)
    avg_line_length = char_count / max(len(non_empty), 1)

    return DatasetAnalysis(
        path=dataset_path,
        format=fmt,
        sample_count=sample_count,
        char_count=char_count,
        word_count=word_count,
        avg_line_length=avg_line_length,
        has_dialogue_markers=has_dialogue_markers,
        has_role_fields=has_role_fields,
        preview_lines=non_empty[:5],
    )


def auto_configure(
    dataset: str,
    dataset_path: str,
    available_models: Optional[List[str]] = None,
    preferred_model: Optional[str] = None,
) -> TrainingConfig:
    """Analyse a dataset and produce a complete training config.

    This is the core of the persona-aligned backend:
    the user provides a dataset name, we figure out everything else.

    Args:
        dataset: Dataset folder name (under datasets/)
        dataset_path: Full path to the dataset file
        available_models: List of available model IDs
        preferred_model: User's preferred model (overrides auto-select)

    Returns:
        TrainingConfig with all parameters set and a plain-language explanation.
    """
    analysis = analyse_dataset(dataset_path)
    models = available_models or ["gpt2"]

    # Pick model
    model = preferred_model or _pick_model(analysis, models)

    # Pick method
    method = _pick_method(analysis)

    # Pick hyperparameters
    epochs = _pick_epochs(analysis)
    batch_size = _pick_batch_size(analysis)
    lr = _pick_lr(method, analysis)
    max_seq_length = _pick_seq_length(analysis)
    warmup_steps = _pick_warmup(analysis)

    # LoRA: always on for fine-tune (saves memory, nearly free)
    use_lora = method == "finetune"

    # RL: auto-enable for chat models (instruction-tuned models benefit from GRPO)
    is_chat_model = any(
        kw in model.lower()
        for kw in ["instruct", "chat", "qwen", "smollm", "tinyllama", "phi"]
    )
    rl_post_train = method == "finetune" and is_chat_model

    # Build explanation
    explanation = _build_explanation(analysis, method, model, epochs, rl_post_train)

    config = TrainingConfig(
        dataset=dataset,
        data_path=dataset_path,
        model=model,
        method=method,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        max_seq_length=max_seq_length,
        warmup_steps=warmup_steps,
        use_lora=use_lora,
        lora_rank=8,
        lora_alpha=16,
        rl_post_train=rl_post_train,
        rl_num_generations=4,
        rl_learning_rate=1e-6,
        rl_kl_coef=0.1,
        rl_reward_mode="length",
        analysis=analysis,
        explanation=explanation,
    )

    logger.info(
        "Auto-config: dataset=%s, method=%s, model=%s, epochs=%d, rl=%s",
        dataset, method, model, epochs, rl_post_train,
    )

    return config


# ---------------------------------------------------------------------------
# Internal helpers — the "brain" of auto-config
# ---------------------------------------------------------------------------

def _is_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def _pick_model(analysis: DatasetAnalysis, models: List[str]) -> str:
    """Pick the best model for this dataset.

    Rules:
    - Dialogue datasets → prefer chat-tuned models (Qwen, SmolLM)
    - Tiny datasets → use small models (avoid overfitting)
    - Large datasets → any model works, prefer the first available
    """
    chat_models = [m for m in models if any(
        kw in m.lower() for kw in ["instruct", "chat", "qwen", "smollm"]
    )]
    small_models = [m for m in models if any(
        kw in m.lower() for kw in ["0.5b", "135m", "124m", "gpt2"]
    )]

    if analysis.is_dialogue and chat_models:
        return chat_models[0]
    if analysis.size_category in ("tiny", "small") and small_models:
        return small_models[0]
    # Prefer Qwen if available (best chat quality on CPU)
    qwen = [m for m in models if "qwen" in m.lower()]
    if qwen:
        return qwen[0]
    return models[0]


def _pick_method(analysis: DatasetAnalysis) -> str:
    """Pick the training method.

    Rules:
    - Messages format (JSONL with role fields) → finetune
    - Dialogue markers in text → finetune
    - Plain text, small → distill
    - Plain text, large → finetune
    """
    if analysis.is_messages_format:
        return "finetune"
    if analysis.is_dialogue:
        return "finetune"
    if analysis.size_category in ("tiny", "small"):
        return "distill"
    return "finetune"


def _pick_epochs(analysis: DatasetAnalysis) -> int:
    """More data → fewer epochs needed.

    - tiny (<1K words): 10 epochs (needs repetition to learn)
    - small (<10K words): 5 epochs
    - medium (<100K words): 3 epochs
    - large (100K+ words): 2 epochs
    """
    return {
        "tiny": 10,
        "small": 5,
        "medium": 3,
        "large": 2,
    }.get(analysis.size_category, 3)


def _pick_batch_size(analysis: DatasetAnalysis) -> int:
    """Smaller datasets → smaller batches.

    - tiny: 2
    - small: 4
    - medium: 4
    - large: 8
    """
    return {
        "tiny": 2,
        "small": 4,
        "medium": 4,
        "large": 8,
    }.get(analysis.size_category, 4)


def _pick_lr(method: str, analysis: DatasetAnalysis) -> float:
    """Learning rate depends on method and dataset size.

    - distill: higher LR (1e-3) — training a small model from scratch
    - finetune on tiny data: lower LR (1e-5) — avoid overfitting
    - finetune on medium+ data: standard LR (2e-4)
    """
    if method == "distill":
        return 1e-3
    if analysis.size_category == "tiny":
        return 1e-5
    if analysis.size_category == "small":
        return 5e-5
    return 2e-4


def _pick_seq_length(analysis: DatasetAnalysis) -> int:
    """Shorter texts → shorter sequence length (saves memory).

    - avg line < 50 chars: 128
    - avg line < 200 chars: 256
    - else: 512
    """
    if analysis.avg_line_length < 50:
        return 128
    if analysis.avg_line_length < 200:
        return 256
    return 512


def _pick_warmup(analysis: DatasetAnalysis) -> int:
    """Fewer samples → fewer warmup steps."""
    if analysis.sample_count < 50:
        return 10
    if analysis.sample_count < 500:
        return 30
    return 50


def _build_explanation(
    analysis: DatasetAnalysis,
    method: str,
    model: str,
    epochs: int,
    rl_enabled: bool,
) -> str:
    """Build a plain-language explanation of the chosen config.

    This is what Alex sees — no jargon, just what we're doing and why.
    """
    parts = []

    # Dataset description
    if analysis.is_messages_format:
        parts.append(f"Found {analysis.sample_count} conversation turns")
    elif analysis.is_dialogue:
        parts.append(f"Found dialogue text ({analysis.word_count:,} words)")
    else:
        parts.append(f"Found {analysis.word_count:,} words of text")

    # Method
    if method == "distill":
        parts.append(f"Training a small model from scratch using {model} as teacher")
    else:
        parts.append(f"Fine-tuning {model} on your data")

    # Epochs explanation
    if analysis.size_category == "tiny":
        parts.append(f"Small dataset — training for {epochs} epochs to learn patterns")
    elif analysis.size_category == "large":
        parts.append(f"Large dataset — {epochs} epochs is enough")
    else:
        parts.append(f"Training for {epochs} epochs")

    # RL explanation
    if rl_enabled:
        parts.append("Auto-enabling personality reinforcement (your AI will learn to give better answers)")

    return ". ".join(parts) + "."


def plain_language_verdict(eval_delta: Dict[str, Any]) -> str:
    """Translate eval delta metrics into a plain-language verdict.

    Takes the dict from ``LoRAEvaluator.compare()`` and returns
    a sentence Alex can understand: no jargon, just what happened.

    Args:
        eval_delta: Dict with keys like perplexity_delta, bleu_delta,
                    verdict ("improved"/"degraded"/"mixed").

    Returns:
        Plain-language string: "Your AI learned to give better answers."
    """
    verdict = eval_delta.get("verdict", "unknown")

    if verdict == "improved":
        parts = ["Your AI learned to give better answers."]
        pp = eval_delta.get("perplexity_improvement_pct")
        if pp is not None and pp > 0:
            parts.append(f"It's {pp:.0f}% more coherent.")
        bleu = eval_delta.get("bleu_delta")
        if bleu is not None and bleu > 0:
            parts.append("Responses are more relevant.")
        personality = eval_delta.get("personality_delta")
        if personality is not None and personality > 0.05:
            parts.append("Personality is more consistent.")
        return " ".join(parts)

    if verdict == "degraded":
        parts = ["Training made some things worse."]
        pp = eval_delta.get("perplexity_improvement_pct")
        if pp is not None and pp < 0:
            parts.append(f"Coherence dropped by {abs(pp):.0f}%.")
        parts.append("Try training for fewer epochs, or use a smaller learning rate.")
        return " ".join(parts)

    # mixed
    parts = ["Mixed results — some things improved, some got worse."]
    pp = eval_delta.get("perplexity_improvement_pct")
    if pp is not None:
        direction = "improved" if pp > 0 else "worsened"
        parts.append(f"Coherence {direction} by {abs(pp):.0f}%.")
    parts.append("Try adjusting the training settings.")
    return " ".join(parts)
