"""
Model Pruning for SloughGPT

Implements efficient model pruning for low-end device deployment:
- Magnitude pruning
- Gradient-based importance pruning
- Structured pruning (attention heads, layers)
- Lottery Ticket Hypothesis
"""

from domains.training.slonet_compat import torch
nn = torch.nn
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger("man.pruning")


@dataclass
class PruningConfig:
    """Configuration for model pruning."""

    method: str = "magnitude"  # magnitude, gradient, lottery
    sparsity: float = 0.5  # Target sparsity (0-1)
    gradual: bool = True  # Gradual pruning schedule
    prune_epochs: int = 10
    prune_frequency: int = 100  # Steps between pruning
    use_gradient_importance: bool = False
    prune_attention_heads: bool = False
    prune_layers: bool = False


class MagnitudePruner:
    """
    Magnitude-based weight pruning.

    Prunes weights with smallest absolute values.
    """

    def __init__(self, model: nn.Module, sparsity: float = 0.5):
        self.model = model
        self.sparsity = sparsity
        self.mask_cache: Dict[str, torch.Tensor] = {}

    def compute_mask(self, layer_name: str, weight: torch.Tensor) -> torch.Tensor:
        """Compute binary mask based on weight magnitudes."""
        threshold = torch.quantile(weight.abs().float(), self.sparsity)
        mask = (weight.abs() > threshold).float()
        return mask

    def prune(self) -> Dict[str, torch.Tensor]:
        """Prune model weights and return masks."""
        masks = {}

        for name, param in self.model.named_parameters():
            if "weight" in name:
                mask = self.compute_mask(name, param.data)
                masks[name] = mask
                # Apply pruning
                param.data = param.data * mask

        self.mask_cache = masks
        return masks

    def apply_mask(self):
        """Apply cached masks to model."""
        for name, param in self.model.named_parameters():
            if name in self.mask_cache:
                param.data = param.data * self.mask_cache[name]

    def restore_weights(self):
        """Restore original weights (undo pruning)."""
        # This requires storing original weights before pruning
        pass


class GradientPruner:
    """
    Gradient-based importance pruning.

    Prunes weights with lowest gradient-based importance scores.
    """

    def __init__(self, model: nn.Module, sparsity: float = 0.5):
        self.model = model
        self.sparsity = sparsity
        self.scores: Dict[str, torch.Tensor] = {}

    def compute_importance(self) -> Dict[str, torch.Tensor]:
        """Compute importance scores based on gradients."""
        importance = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Fisher importance: gradient^2
                score = param.grad.data.abs() * param.data.abs()
                importance[name] = score

        self.scores = importance
        return importance

    def prune(self) -> Dict[str, torch.Tensor]:
        """Prune based on importance scores."""
        if not self.scores:
            self.compute_importance()

        masks = {}
        for name, score in self.scores.items():
            threshold = torch.quantile(score.float(), self.sparsity)
            mask = (score > threshold).float()
            masks[name] = mask

        return masks


class StructuredPruner:
    """
    Structured pruning for efficient inference.

    Prunes entire attention heads or layers.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.head_mask: Optional[torch.Tensor] = None
        self.layer_mask: Optional[torch.Tensor] = None

    def prune_attention_heads(
        self,
        num_heads_to_prune: int,
        importance_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prune attention heads based on importance.

        Args:
            num_heads_to_prune: Number of heads to remove
            importance_scores: Optional importance for each head

        Returns:
            Binary mask for heads (1=keep, 0=prune)
        """
        # Find number of attention layers
        num_heads = 0
        for name, module in self.model.named_modules():
            if "attention" in name.lower() and hasattr(module, "num_heads"):
                num_heads = module.num_heads
                break

        if num_heads == 0:
            logger.warning("Could not find attention layers")
            return torch.ones(num_heads)

        if importance_scores is None:
            # Random importance (uniform)
            importance_scores = torch.ones(num_heads)

        # Get indices of least important heads
        _, prune_indices = torch.topk(
            importance_scores, k=min(num_heads_to_prune, num_heads), largest=False
        )

        # Create mask
        head_mask = torch.ones(num_heads)
        head_mask[prune_indices] = 0

        self.head_mask = head_mask
        return head_mask

    def prune_layers(
        self,
        num_layers_to_prune: int,
        importance_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prune entire layers.

        Args:
            num_layers_to_prune: Number of layers to remove
            importance_scores: Optional importance for each layer

        Returns:
            Binary mask for layers (1=keep, 0=prune)
        """
        # Count transformer layers
        num_layers = 0
        for name, module in self.model.named_modules():
            if "block" in name.lower() or "layer" in name.lower():
                num_layers += 1

        if num_layers == 0:
            logger.warning("Could not find transformer layers")
            return torch.ones(num_layers)

        if importance_scores is None:
            importance_scores = torch.ones(num_layers)

        _, prune_indices = torch.topk(
            importance_scores, k=min(num_layers_to_prune, num_layers), largest=False
        )

        layer_mask = torch.ones(num_layers)
        layer_mask[prune_indices] = 0

        self.layer_mask = layer_mask
        return layer_mask

    def apply_masks(self):
        """Apply head/layer masks to model."""
        if self.head_mask is not None:
            # Apply head mask to attention layers
            head_idx = 0
            for name, module in self.model.named_modules():
                if "attention" in name.lower():
                    if hasattr(module, "head_mask"):
                        module.head_mask = self.head_mask
                    head_idx += 1


class LotteryTicketPruner:
    """
    Lottery Ticket Hypothesis pruning.

    Finds sparse subnetworks that can be trained from scratch.
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        pr_steps: int = 100,
    ):
        self.model = model
        self.sparsity = sparsity
        self.pr_steps = pr_steps
        self.original_weights: Dict[str, torch.Tensor] = {}
        self.masks: Dict[str, torch.Tensor] = {}

    def init(self):
        """Initialize by storing original weights."""
        for name, param in self.model.named_parameters():
            if "weight" in name:
                self.original_weights[name] = param.data.clone()
                self.masks[name] = torch.ones_like(param.data)

    def step(self, current_step: int, total_steps: int):
        """
        Perform one pruning step.

        Args:
            current_step: Current training step
            total_steps: Total training steps
        """
        # Calculate current sparsity target
        progress = current_step / total_steps
        current_sparsity = self.sparsity * progress

        if current_sparsity > 0:
            for name, param in self.model.named_parameters():
                if name in self.original_weights:
                    # Rescale weights
                    original = self.original_weights[name]
                    param.data = original * self.masks[name]

                    # Update mask
                    threshold = torch.quantile(original.abs().float(), current_sparsity)
                    mask = (original.abs() > threshold).float()
                    self.masks[name] = mask

    def get_final_masks(self) -> Dict[str, torch.Tensor]:
        """Get final pruning masks."""
        return self.masks


class EfficientPruner:
    """
    Unified pruner with automatic method selection based on target device.
    """

    @staticmethod
    def create_pruner(
        model: nn.Module,
        target_device: str = "cpu",
        sparsity: float = 0.5,
    ) -> MagnitudePruner:
        """
        Create appropriate pruner for target device.

        Args:
            model: Model to prune
            target_device: cpu, mobile, or embedded
            sparsity: Target sparsity (0-1)

        Returns:
            Appropriate pruner instance
        """
        if target_device in ["cpu", "mobile", "embedded"]:
            # Use structured pruning for efficiency
            return StructuredPruner(model)
        else:
            return MagnitudePruner(model, sparsity)

    @staticmethod
    def prune_for_inference(
        model: nn.Module,
        sparsity: float = 0.5,
        method: str = "magnitude",
    ) -> nn.Module:
        """
        Prune model for efficient inference.

        Args:
            model: Model to prune
            sparsity: Target sparsity (0-1)
            method: Pruning method

        Returns:
            Pruned model
        """
        if method == "magnitude":
            pruner = MagnitudePruner(model, sparsity)
        elif method == "gradient":
            pruner = GradientPruner(model, sparsity)
        else:
            pruner = MagnitudePruner(model, sparsity)

        pruner.prune()

        # Set model to inference mode
        model.eval()

        return model


def prune_model(
    model: nn.Module,
    sparsity: float = 0.5,
    method: str = "magnitude",
    prune_attention: bool = False,
    prune_layers: int = 0,
) -> nn.Module:
    """
    Convenience function to prune a model.

    Args:
        model: Model to prune
        sparsity: Target sparsity (0-1)
        method: Pruning method
        prune_attention: Whether to prune attention heads
        prune_layers: Number of layers to prune

    Returns:
        Pruned model
    """
    pruner = MagnitudePruner(model, sparsity)
    pruner.prune()

    if prune_attention:
        structured = StructuredPruner(model)
        structured.prune_attention_heads(prune_attention)

    return model


__all__ = [
    "PruningConfig",
    "MagnitudePruner",
    "GradientPruner",
    "StructuredPruner",
    "LotteryTicketPruner",
    "EfficientPruner",
    "prune_model",
]
