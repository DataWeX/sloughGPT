"""
Personality Contrastive Loss for SloughGPT

Training loss functions for personality alignment.

SloNet/numpy is the native runtime. Torch tensors are converted
to numpy at the API boundary via a single helper.
"""

import math
import logging
from typing import Dict, Optional, Union, List

import numpy as np

from domains.training.slonet import (
    Tensor, cross_entropy, mse_loss, normalize,
    pairwise_distance, relu, softmax, log_softmax,
    zeros, tensor,
)

logger = logging.getLogger("sloughgpt.personality")

ArrayLike = Union[Tensor, np.ndarray]


def _to_np(x):
    """Convert any array-like to a numpy ndarray."""
    if isinstance(x, Tensor):
        return x.data
    if hasattr(x, 'cpu'):  # torch.Tensor
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x, dtype=np.float32)


def _to_tensor(arr, requires_grad=False):
    """Convert a numpy array to a SloNet Tensor."""
    if isinstance(arr, Tensor):
        if requires_grad and not arr.requires_grad:
            arr.requires_grad = True
        return arr
    if isinstance(arr, np.ndarray):
        return Tensor(arr.astype(np.float32), requires_grad=requires_grad)
    return Tensor(np.asarray(arr, dtype=np.float32), requires_grad=requires_grad)


# =============================================================================
# Personality Losses
# =============================================================================


class PersonalityContrastiveLoss:
    """
    Contrastive loss for personality training.

    Pulls samples with similar personalities together,
    pushes dissimilar ones apart in embedding space.
    """

    def __init__(self, temperature: float = 0.1, margin: float = 0.5):
        self.temperature = temperature
        self.margin = margin

    def __call__(self, embeddings, traits):
        return self.forward(embeddings, traits)

    def forward(self, embeddings, traits):
        emb = _to_np(embeddings)
        tr = _to_np(traits)

        emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        sim_matrix = emb_n @ emb_n.T / self.temperature
        trait_sim = tr @ tr.T
        labels = (trait_sim > self.margin).astype(np.float32)
        n = emb.shape[0]
        labels = labels * (1 - np.eye(n, dtype=np.float32))
        exp_sim = np.exp(sim_matrix - sim_matrix.max(axis=1, keepdims=True))
        log_prob = sim_matrix - np.log(exp_sim.sum(axis=1, keepdims=True) + 1e-8)
        loss_val = -(labels * log_prob).sum() / (labels.sum() + 1e-8)
        return Tensor(np.array(loss_val, dtype=np.float32), requires_grad=True)

    def parameters(self):
        return []

    def train(self, mode=True):
        pass

    def eval(self):
        pass

    def to(self, device):
        return self

    def state_dict(self):
        return {}

    def load_state_dict(self, d):
        pass


class PersonalityMSELoss:
    """MSE loss for predicting personality traits from embeddings."""

    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)

    def forward(self, predictions, targets):
        return mse_loss(
            _to_tensor(_to_np(predictions), requires_grad=True),
            _to_tensor(_to_np(targets)),
        )

    def parameters(self):
        return []


class PersonalityTripletLoss:
    """
    Triplet loss for personality alignment.

    Ensures anchor-positive distance < anchor-negative distance.
    """

    def __init__(self, margin: float = 0.3):
        self.margin = margin

    def __call__(self, anchor, positive, negative):
        return self.forward(anchor, positive, negative)

    def forward(self, anchor, positive, negative):
        pos_dist = pairwise_distance(
            _to_tensor(_to_np(anchor), requires_grad=True),
            _to_tensor(_to_np(positive)),
        )
        neg_dist = pairwise_distance(
            _to_tensor(_to_np(anchor), requires_grad=True),
            _to_tensor(_to_np(negative)),
        )
        loss = relu(pos_dist - neg_dist + self.margin)
        return loss

    def parameters(self):
        return []


class PersonalityLoss:
    """
    Combined personality loss.

    Combines: contrastive + MSE + optional triplet.
    """

    def __init__(
        self,
        contrastive_weight: float = 1.0,
        mse_weight: float = 0.5,
        triplet_weight: float = 0.3,
        temperature: float = 0.1,
    ):
        self.contrastive_weight = contrastive_weight
        self.mse_weight = mse_weight
        self.triplet_weight = triplet_weight
        self.contrastive_loss = PersonalityContrastiveLoss(temperature=temperature)
        self.mse_loss = PersonalityMSELoss()
        self.triplet_loss = PersonalityTripletLoss()

    def __call__(self, embeddings, traits, trait_predictions=None, positive_emb=None, negative_emb=None):
        return self.forward(embeddings, traits, trait_predictions, positive_emb, negative_emb)

    def forward(self, embeddings, traits, trait_predictions=None, positive_emb=None, negative_emb=None):
        losses = {}
        total_loss = 0.0

        if self.contrastive_weight > 0:
            cl = self.contrastive_loss(embeddings, traits)
            losses["contrastive"] = cl
            total_loss += self.contrastive_weight * float(_to_np(cl).reshape(-1)[0])

        if self.mse_weight > 0 and trait_predictions is not None:
            ml = self.mse_loss(trait_predictions, traits)
            losses["mse"] = ml
            total_loss += self.mse_weight * float(_to_np(ml).reshape(-1)[0])

        if self.triplet_weight > 0 and positive_emb is not None and negative_emb is not None:
            tl = self.triplet_loss(embeddings, positive_emb, negative_emb)
            losses["triplet"] = tl
            total_loss += self.triplet_weight * float(_to_np(tl).reshape(-1)[0])

        losses["total"] = total_loss
        return losses

    def parameters(self):
        return []


class ArchetypeAlignmentLoss:
    """Loss for aligning outputs to personality archetypes."""

    def __init__(self, num_archetypes: int = 8):
        self.num_archetypes = num_archetypes
        self.archetypes = np.array([
            [0.9, 0.8, 0.5, 0.5],
            [0.9, -0.5, 0.6, 0.9],
            [0.5, 0.9, 0.9, 0.5],
            [0.9, 0.9, 0.8, 0.5],
            [0.9, 0.8, 0.7, 0.9],
            [0.9, 0.9, 0.7, 0.8],
            [0.9, 0.8, 0.7, 0.5],
            [0.7, 0.9, 0.8, 0.9],
        ], dtype=np.float32)

    def __call__(self, traits, target_archetype=None):
        return self.forward(traits, target_archetype)

    def forward(self, traits, target_archetype=None):
        tr = _to_np(traits)
        tr_n = tr / (np.linalg.norm(tr, axis=1, keepdims=True) + 1e-8)
        arch_n = self.archetypes / (np.linalg.norm(self.archetypes, axis=1, keepdims=True) + 1e-8)
        sim = tr_n @ arch_n.T
        if target_archetype is not None:
            ta = _to_np(target_archetype)
            target_sim = (sim * ta).sum(axis=1)
            loss_val = 1.0 - target_sim.mean()
        else:
            max_sim = sim.max(axis=1)
            loss_val = 1.0 - max_sim.mean()
        return Tensor(np.array(loss_val, dtype=np.float32), requires_grad=True)

    def parameters(self):
        return []


class PersonalityFineTuningLoss:
    """
    Complete loss for personality fine-tuning.

    Combines: LM loss (cross-entropy) + personality alignment.
    """

    def __init__(self, lm_weight=1.0, personality_weight=0.5, kl_weight=0.1):
        self.lm_weight = lm_weight
        self.personality_weight = personality_weight
        self.kl_weight = kl_weight
        self.personality_loss = PersonalityLoss()
        self.archetype_loss = ArchetypeAlignmentLoss()

    def __call__(self, lm_logits, lm_targets, traits, trait_predictions=None):
        return self.forward(lm_logits, lm_targets, traits, trait_predictions)

    def forward(self, lm_logits, lm_targets, traits, trait_predictions=None):
        losses = {}
        total_loss = 0.0

        if self.lm_weight > 0:
            logits = _to_np(lm_logits)
            targets = _to_np(lm_targets)
            ll = cross_entropy(
                _to_tensor(logits.reshape(-1, logits.shape[-1]), requires_grad=True),
                _to_tensor(targets.reshape(-1).astype(np.int64)),
            )
            losses["lm"] = ll
            total_loss += self.lm_weight * float(_to_np(ll).reshape(-1)[0])

        if self.personality_weight > 0:
            logits = _to_np(lm_logits)
            embeddings = logits.mean(axis=1)
            pers_losses = self.personality_loss(
                embeddings=embeddings,
                traits=traits,
                trait_predictions=trait_predictions,
            )
            for k, v in pers_losses.items():
                losses[f"personality_{k}"] = v if isinstance(v, (int, float)) else float(_to_np(v).reshape(-1)[0])
                total_loss += self.personality_weight * (v if isinstance(v, (int, float)) else float(_to_np(v).reshape(-1)[0]))

        losses["total"] = total_loss
        return losses

    def parameters(self):
        return []


# =============================================================================
# Convenience
# =============================================================================


def create_personality_loss(loss_type: str = "combined", **kwargs):
    if loss_type == "contrastive":
        return PersonalityContrastiveLoss(**kwargs)
    elif loss_type == "mse":
        return PersonalityMSELoss()
    elif loss_type == "triplet":
        return PersonalityTripletLoss(**kwargs)
    elif loss_type == "combined":
        return PersonalityLoss(**kwargs)
    elif loss_type == "finetuning":
        return PersonalityFineTuningLoss(**kwargs)
    raise ValueError(f"Unknown loss type: {loss_type}")


__all__ = [
    "PersonalityContrastiveLoss",
    "PersonalityMSELoss",
    "PersonalityTripletLoss",
    "PersonalityLoss",
    "ArchetypeAlignmentLoss",
    "PersonalityFineTuningLoss",
    "create_personality_loss",
]
