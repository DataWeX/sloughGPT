"""
LoRA (Low-Rank Adaptation) Module for SloughGPT

LoRA implementation — SloNet/numpy is the native runtime.
Supports: Standard LoRA, IA3.
All operations use SloNet Tensor / SloLayer.
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from enum import Enum

import numpy as np

from domains.training.slonet import (
    Tensor, SloLayer, SloLinear, SloEmbedding, SloDropout,
    zeros, randn, _matmul, SloAdam,
)

logger = logging.getLogger("man.lora")


def _to_np(x):
    if isinstance(x, Tensor):
        return x.data
    if hasattr(x, 'cpu'):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x, dtype=np.float32)


def _to_tensor(arr, requires_grad=True):
    if isinstance(arr, Tensor):
        if requires_grad and not arr.requires_grad:
            arr.requires_grad = True
        return arr
    return Tensor(np.asarray(arr, dtype=np.float32), requires_grad=requires_grad)


# =============================================================================
# Enums / Config
# =============================================================================


class LoRAType(Enum):
    LORA = "lora"
    LORA_PLUS = "lora_plus"
    IA3 = "ia3"


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    lora_type: LoRAType = LoRAType.LORA
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


# =============================================================================
# LoRALinear  — SloLayer native
# =============================================================================


class LoRALinear(SloLayer):
    """
    LoRA-augmented linear layer.

    Effective weight: W + (alpha / rank) * (B @ A)
    SloLayer subclass using SloNet Tensor ops.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        lora_type: LoRAType = LoRAType.LORA,
        original_weight=None,
        original_bias=None,
    ):
        name = f"LoRALin_{in_features}x{out_features}_r{rank}"
        super().__init__(name)
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.dropout_prob = dropout
        self.lora_type = lora_type

        if original_weight is None:
            w = randn((out_features, in_features), requires_grad=False)
            w.data *= math.sqrt(2.0 / (in_features + out_features))
            self.weight = Tensor(w.data.copy(), requires_grad=False)
        else:
            self.weight = Tensor(_to_np(original_weight).copy(), requires_grad=False)

        if lora_type == LoRAType.IA3:
            self.lora_s = Tensor(np.ones(out_features, dtype=np.float32), requires_grad=True)
        else:
            self.lora_A = Tensor(
                (np.random.randn(rank, in_features) * 0.01).astype(np.float32), requires_grad=True)
            self.lora_B = Tensor(
                np.zeros((out_features, rank), dtype=np.float32), requires_grad=True)

        self.dropout = SloDropout(p=dropout) if dropout > 0 else None
        self.training = True

        if bias:
            if original_bias is None:
                self.bias = Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True)
            else:
                self.bias = Tensor(_to_np(original_bias).copy(), requires_grad=True)
        else:
            self.bias = None

        self._original_forward = None

    def train(self, mode=True):
        self.training = mode
        if self.dropout:
            self.dropout.train(mode)

    def eval(self):
        self.train(False)

    def forward(self, x):
        x_t = _to_tensor(_to_np(x), requires_grad=True)
        w = self.weight
        b = self.bias

        if self.lora_type == LoRAType.IA3:
            original = _matmul(x_t, w.T())
            if b is not None:
                original = original + b
            return original * self.lora_s

        lora_weight = _matmul(self.lora_B, self.lora_A) * (self.alpha / self.rank)
        effective_weight = self.weight + lora_weight
        out = _matmul(x_t, effective_weight.T())
        if b is not None:
            out = out + b
        if self.dropout and self.training:
            out = self.dropout.forward(out)
        return out

    def merge_weights(self):
        if self.lora_type == LoRAType.IA3:
            self.weight.data[:] *= _to_np(self.lora_s)[:, np.newaxis]
            self.lora_s.data[:] = 1.0
        else:
            lora_w = _matmul(self.lora_B, self.lora_A).data * (self.alpha / self.rank)
            self.weight.data[:] += lora_w
            self.lora_A.data[:] = 0.0
            self.lora_B.data[:] = 0.0

    def get_trainable_parameters(self):
        if self.lora_type == LoRAType.IA3:
            return [self.lora_s]
        return [self.lora_A, self.lora_B]

    def parameters(self):
        return self.get_trainable_parameters()

    def named_parameters(self, prefix=""):
        params = self.get_trainable_parameters()
        names = {
            id(self.lora_s): f"{prefix}lora_s" if self.lora_type == LoRAType.IA3 else None,
            id(self.lora_A): f"{prefix}lora_A",
            id(self.lora_B): f"{prefix}lora_B",
        }
        return [(names.get(id(p), f"{prefix}p{i}"), p) for i, p in enumerate(params) if names.get(id(p))]


# =============================================================================
# LoRAEmbedding  — SloLayer native
# =============================================================================


class LoRAEmbedding(SloLayer):
    """LoRA for embedding layers — SloNet native."""

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int = 8,
                 alpha: float = 16.0, original_weight=None):
        name = f"LoRAEmb_{num_embeddings}x{embedding_dim}_r{rank}"
        super().__init__(name)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.rank = rank
        self.alpha = alpha

        if original_weight is None:
            self.weight = SloEmbedding(num_embeddings, embedding_dim, "lora_emb")
        else:
            w_data = _to_np(original_weight)
            self.weight = SloEmbedding(num_embeddings, embedding_dim, "lora_emb")
            self.weight.weight.data[:] = w_data[:num_embeddings, :embedding_dim]
            self.weight.weight.requires_grad = False

        self.lora_A = Tensor(
            (np.random.randn(rank, embedding_dim) * 0.01).astype(np.float32), requires_grad=True)
        self.lora_B = Tensor(
            np.zeros((embedding_dim, rank), dtype=np.float32), requires_grad=True)

    def forward(self, x):
        x_t = _to_tensor(_to_np(x), requires_grad=True)
        original = self.weight.forward(x_t)
        lora_contrib = _matmul(original, _matmul(self.lora_B, self.lora_A).T())
        return original + lora_contrib * (self.alpha / self.rank)

    def merge_weights(self):
        lora_w = _matmul(self.lora_B, self.lora_A).data * (self.alpha / self.rank)
        n = min(self.num_embeddings, self.weight.weight.data.shape[0])
        d = min(self.embedding_dim, self.weight.weight.data.shape[1])
        self.weight.weight.data[:n, :d] += lora_w[:n, :d]

    def get_trainable_parameters(self):
        return [self.lora_A, self.lora_B]

    def parameters(self):
        return self.get_trainable_parameters()


# =============================================================================
# apply_lora_to_model
# =============================================================================


def apply_lora_to_model(model, config: Optional[LoRAConfig] = None,
                         rank: int = 8, alpha: float = 16.0,
                         target_modules: Optional[List[str]] = None):
    """
    Apply LoRA to a model.

    Works with both SloLayer-based models and torch nn.Module models.
    Torch Linear/Embedding layers are replaced with LoRA versions that use SloNet ops.

    Args:
        model: Model with named_modules/named_children
        config: LoRAConfig (preferred)
        rank: LoRA rank (if no config)
        alpha: LoRA alpha (if no config)
        target_modules: Module names to apply LoRA

    Returns:
        Model with LoRA applied
    """
    if config is None:
        config = LoRAConfig(rank=rank, alpha=alpha, target_modules=target_modules)
    target_modules = config.target_modules or []

    named_items = model.named_modules() if hasattr(model, 'named_modules') else model.named_children()

    for name, module in named_items:
        module_name = name.split(".")[-1]
        if module_name in target_modules or any(t in name for t in target_modules):
            in_f = getattr(module, 'in_features', getattr(module, 'in_f', None))
            out_f = getattr(module, 'out_features', getattr(module, 'out_f', None))
            if in_f is None or out_f is None:
                continue

            is_linear = (hasattr(module, 'weight') and hasattr(module, 'bias'))
            is_embedding = (hasattr(module, 'num_embeddings') and hasattr(module, 'embedding_dim'))

            if is_embedding:
                new_lora = LoRAEmbedding(
                    num_embeddings=in_f if in_f else module.num_embeddings,
                    embedding_dim=out_f if out_f else module.embedding_dim,
                    rank=config.rank, alpha=config.alpha,
                    original_weight=getattr(module, 'weight', None),
                )
            else:
                module_bias = getattr(module, 'bias', None) is not None
                new_lora = LoRALinear(
                    in_features=in_f, out_features=out_f,
                    bias=module_bias,
                    rank=config.rank, alpha=config.alpha, dropout=config.dropout,
                    lora_type=config.lora_type,
                    original_weight=getattr(module, 'weight', None),
                    original_bias=getattr(module, 'bias', None),
                )

            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_lora)
            logger.info(f"Applied LoRA to {name}")

    return model


def get_lora_parameters(model):
    """Get only LoRA parameters from a model."""
    params = {}
    named = model.named_parameters() if hasattr(model, 'named_parameters') else []
    for name, param in named:
        if "lora_" in name:
            params[name] = param
    return params


def count_lora_parameters(model) -> int:
    """Count trainable LoRA parameters."""
    total = 0
    for p in model.parameters():
        if hasattr(p, 'requires_grad') and p.requires_grad:
            total += p.data.size if hasattr(p, 'data') else 1
    return total


def print_lora_summary(model):
    """Print LoRA parameter summary."""
    lora_params = get_lora_parameters(model)
    total = sum(p.data.size if hasattr(p, 'data') else 1 for p in lora_params.values())
    logger.info(f"LoRA parameters: {len(lora_params)} tensors, {total:,} total parameters")
    param_counts = {}
    for name, p in lora_params.items():
        key = name.split(".")[-1].split("_")[0]
        sz = p.data.size if hasattr(p, 'data') else 1
        param_counts[key] = param_counts.get(key, 0) + sz
    for k, v in param_counts.items():
        logger.info(f"  {k}: {v:,} parameters")


# =============================================================================
# LoRATrainer  — SloNet native
# =============================================================================


class LoRATrainer:
    """Trainer for LoRA models using SloAdam."""

    def __init__(self, model, config: Optional[LoRAConfig] = None, lr: float = 1e-4):
        self.model = model
        self.config = config or LoRAConfig()
        self.lora_params = list(model.parameters()) if hasattr(model, 'parameters') else []
        self.optimizer = SloAdam(lr=lr)

    def step(self, loss):
        loss.backward()
        self.optimizer.step(self.lora_params)
        for p in self.lora_params:
            if hasattr(p, 'grad'):
                p.grad = None


__all__ = [
    "LoRAType",
    "LoRAConfig",
    "LoRALinear",
    "LoRAEmbedding",
    "apply_lora_to_model",
    "get_lora_parameters",
    "count_lora_parameters",
    "print_lora_summary",
    "LoRATrainer",
]
