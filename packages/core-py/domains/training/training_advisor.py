"""
Training Advisor — recommends training configuration based on data and model.

Analyzes dataset size, quality, and model type to suggest optimal hyperparameters.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("slo.training.advisor")


@dataclass
class TrainingRecommendation:
    """Recommended training configuration."""
    learning_rate: float
    batch_size: int
    epochs: int
    warmup_steps: int
    early_stopping_patience: int
    reason: str
    confidence: float  # 0-1, how confident we are in this recommendation


def recommend_training_config(
    dataset_size: int,
    model_params: Optional[int] = None,
    avg_quality: Optional[float] = None,
    method: str = "distill",
) -> TrainingRecommendation:
    """
    Recommend training configuration based on dataset and model characteristics.
    
    Args:
        dataset_size: Number of training examples
        model_params: Number of model parameters (if known)
        avg_quality: Average quality score of training data (0-5)
        method: Training method ('distill', 'finetune', 'native')
    
    Returns:
        TrainingRecommendation with suggested hyperparameters
    """
    # Base recommendations by method
    if method == "finetune":
        base_lr = 1e-4
        base_batch = 8
        base_epochs = 10
    elif method == "native":
        base_lr = 3e-4
        base_batch = 16
        base_epochs = 100
    else:  # distill
        base_lr = 3e-4
        base_batch = 32
        base_epochs = 20

    # Adjust for dataset size
    if dataset_size < 100:
        # Very small dataset: lower LR, smaller batch, more epochs
        lr_factor = 0.5
        batch_factor = 0.5
        epoch_factor = 2.0
        reason = "Very small dataset: reduced learning rate and batch size for stability"
        confidence = 0.6
    elif dataset_size < 1000:
        # Small dataset: slightly lower LR
        lr_factor = 0.75
        batch_factor = 0.75
        epoch_factor = 1.5
        reason = "Small dataset: moderate reduction in learning rate"
        confidence = 0.7
    elif dataset_size < 10000:
        # Medium dataset: standard settings
        lr_factor = 1.0
        batch_factor = 1.0
        epoch_factor = 1.0
        reason = "Medium dataset: standard training configuration"
        confidence = 0.8
    elif dataset_size < 100000:
        # Large dataset: can use higher LR
        lr_factor = 1.2
        batch_factor = 1.5
        epoch_factor = 0.8
        reason = "Large dataset: increased learning rate and batch size"
        confidence = 0.85
    else:
        # Very large dataset: aggressive training
        lr_factor = 1.5
        batch_factor = 2.0
        epoch_factor = 0.5
        reason = "Very large dataset: aggressive training with larger batches"
        confidence = 0.9

    # Adjust for model size
    if model_params is not None:
        if model_params < 1_000_000:  # < 1M params
            lr_factor *= 1.2  # Small models can handle higher LR
        elif model_params > 100_000_000:  # > 100M params
            lr_factor *= 0.5  # Large models need lower LR
            batch_factor *= 0.5

    # Adjust for data quality
    if avg_quality is not None:
        if avg_quality < 2.0:
            lr_factor *= 0.5
            epoch_factor *= 0.5
            reason += "; low data quality detected"
            confidence *= 0.7
        elif avg_quality > 4.0:
            lr_factor *= 1.1
            confidence *= 1.1

    # Calculate final values
    learning_rate = base_lr * lr_factor
    batch_size = max(4, int(base_batch * batch_factor))
    epochs = max(5, int(base_epochs * epoch_factor))
    
    # Warmup steps based on dataset size
    warmup_steps = max(10, min(100, dataset_size // 100))
    
    # Early stopping patience based on epochs
    early_stopping_patience = max(3, epochs // 5)

    # Clamp values to reasonable ranges
    learning_rate = max(1e-6, min(1e-2, learning_rate))
    batch_size = max(4, min(128, batch_size))
    epochs = max(5, min(500, epochs))

    return TrainingRecommendation(
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        warmup_steps=warmup_steps,
        early_stopping_patience=early_stopping_patience,
        reason=reason,
        confidence=min(1.0, confidence),
    )


def get_training_tips(
    dataset_size: int,
    current_loss: Optional[float] = None,
    best_loss: Optional[float] = None,
    trend: Optional[float] = None,
) -> list[str]:
    """
    Generate helpful tips based on training state.
    
    Args:
        dataset_size: Number of training examples
        current_loss: Current training loss
        best_loss: Best loss achieved so far
        trend: Loss trend (negative = improving, positive = diverging)
    
    Returns:
        List of actionable tips
    """
    tips = []
    
    if dataset_size < 100:
        tips.append("Very small dataset. Consider collecting more data or using data augmentation.")
    
    if dataset_size < 1000:
        tips.append("Small dataset. Use LoRA or fine-tuning instead of training from scratch.")
    
    if current_loss is not None and best_loss is not None:
        gap = current_loss - best_loss
        if gap > 0.5:
            tips.append("Large gap between current and best loss. Model may be overfitting.")
        elif gap < 0.01:
            tips.append("Training has converged. Consider stopping or reducing learning rate.")
    
    if trend is not None:
        if trend > 0.1:
            tips.append("Loss is increasing. Lower learning rate or add warmup steps.")
        elif trend < -0.1:
            tips.append("Good progress! Loss is decreasing steadily.")
    
    if not tips:
        tips.append("Training looks healthy. Keep going!")
    
    return tips


__all__ = [
    "TrainingRecommendation",
    "recommend_training_config",
    "get_training_tips",
]
