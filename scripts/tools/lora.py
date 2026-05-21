"""
LoRA (Low‑Rank Adaptation) utilities for NanoGPT.

This module provides a lightweight LoRA implementation that can be applied to
any ``nn.Linear`` layer in the model. When LoRA is enabled the original weights
are frozen and low‑rank adapters ``A`` and ``B`` are learned instead. The
effective weight becomes ``W + (alpha / rank) * (B @ A)``.

Usage::

    from lora import apply_lora_to_model
    model = apply_lora_to_model(model, rank=4, alpha=1.0)

The function recursively replaces ``nn.Linear`` modules with ``LoRALinear``.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LoRALinear(nn.Module):
    """LoRA‑augmented linear layer.

    The original weight ``W`` is frozen (no gradient). Two low‑rank matrices
    ``A`` (rank × in_features) and ``B`` (out_features × rank) are learned. The
    effective weight is ``W + (alpha / rank) * (B @ A)``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int = 4,
        alpha: float = 1.0,
        original_weight: Optional[torch.Tensor] = None,
        original_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        # Original weight (frozen)
        if original_weight is None:
            self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            self.weight = nn.Parameter(original_weight.clone().detach(), requires_grad=False)
        # LoRA parameters (trainable)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        # Bias handling
        if bias:
            if original_bias is None:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.bias = nn.Parameter(original_bias.clone().detach())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute LoRA contribution
        lora_weight = (self.lora_B @ self.lora_A) * (self.alpha / self.rank)
        effective_weight = self.weight + lora_weight
        return F.linear(x, effective_weight, self.bias)


def apply_lora_to_model(model: nn.Module, rank: int = 4, alpha: float = 1.0) -> nn.Module:
    """Recursively replace ``nn.Linear`` layers with ``LoRALinear``.

    Only the LoRA adapters are trainable; the original weights are frozen.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            new_linear = LoRALinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                rank=rank,
                alpha=alpha,
                original_weight=module.weight,
                original_bias=module.bias,
            )
            setattr(model, name, new_linear)
        else:
            apply_lora_to_model(module, rank=rank, alpha=alpha)
    return model
