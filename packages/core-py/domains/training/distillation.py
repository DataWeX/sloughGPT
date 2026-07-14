"""
Knowledge Distillation for SloughGPT

Knowledge distillation — SloNet/numpy is the native runtime.
Torch tensors are converted to numpy at the API boundary.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

from domains.training.slonet import (
    softmax, log_softmax, cross_entropy, mse_loss, kl_div_loss,
    Tensor, SloLinear, no_grad, SloAdam,
)

logger = logging.getLogger("man.distillation")


ArrayLike = Union[Tensor, np.ndarray]


def _to_np(x):
    if isinstance(x, Tensor):
        return x.data
    if hasattr(x, 'cpu'):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x, dtype=np.float32)


def _to_tensor(arr, requires_grad=False):
    if isinstance(arr, Tensor):
        if requires_grad and not arr.requires_grad:
            arr.requires_grad = True
        return arr
    return Tensor(np.asarray(arr, dtype=np.float32), requires_grad=requires_grad)


def _size(x, dim):
    d = _to_np(x)
    return d.shape[dim] if dim < len(d.shape) else 1


# =============================================================================
# DistillationConfig
# =============================================================================


@dataclass
class DistillationConfig:
    temperature: float = 4.0
    temperature_schedule: Optional[List[float]] = None
    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 0.0
    distillation_type: str = "logits"
    use_label_smoothing: bool = False
    label_smoothing: float = 0.1
    progressive: bool = False
    stage_weights: Optional[List[float]] = None
    hidden_layer_mapping: Optional[Dict[int, int]] = None


# =============================================================================
# DistillationLoss  — SloNet native
# =============================================================================


class DistillationLoss:
    """
    Combined loss for knowledge distillation.

    Combines hard label loss (CE), soft label loss (KL), and feature loss (MSE).
    All operations use SloNet Tensor / numpy.
    """

    def __init__(self, config: DistillationConfig):
        self.config = config
        self.projection = None

    def __call__(self, student_logits, teacher_logits, labels=None,
                 student_hidden=None, teacher_hidden=None):
        return self.forward(student_logits, teacher_logits, labels,
                            student_hidden, teacher_hidden)

    def forward(self, student_logits, teacher_logits, labels=None,
                student_hidden=None, teacher_hidden=None):
        losses = {}

        s = _to_np(student_logits)
        t = _to_np(teacher_logits)
        temp = self.config.temperature

        if self.config.beta > 0:
            student_soft = log_softmax(_to_tensor(s / temp), dim=-1)
            teacher_soft = softmax(_to_tensor(t / temp), dim=-1)
            soft_loss = kl_div_loss(student_soft, teacher_soft, reduction="batchmean")
            soft_loss = soft_loss * (temp ** 2)
            losses["soft_loss"] = float(_to_np(soft_loss).reshape(-1)[0])

        if self.config.alpha > 0 and labels is not None:
            lbl = _to_np(labels).reshape(-1).astype(np.int64)
            s_flat = s.reshape(-1, s.shape[-1])
            hard_loss = cross_entropy(_to_tensor(s_flat, requires_grad=True),
                                      _to_tensor(lbl))
            losses["hard_loss"] = float(_to_np(hard_loss).reshape(-1)[0])

        if self.config.gamma > 0 and student_hidden is not None and teacher_hidden is not None:
            sh = _to_np(student_hidden)
            th = _to_np(teacher_hidden)
            if sh.shape[-1] != th.shape[-1]:
                sh = self._project(sh, th.shape[-1])
            feat_loss = mse_loss(_to_tensor(sh, requires_grad=True), _to_tensor(th))
            losses["feature_loss"] = float(_to_np(feat_loss).reshape(-1)[0])

        total = 0.0
        if self.config.alpha > 0 and labels is not None:
            total += self.config.alpha * losses.get("hard_loss", 0.0)
        if self.config.beta > 0:
            total += self.config.beta * losses.get("soft_loss", 0.0)
        if self.config.gamma > 0 and student_hidden is not None:
            total += self.config.gamma * losses.get("feature_loss", 0.0)

        losses["total_loss"] = total
        return total, losses

    def _project(self, hidden, target_dim):
        if self.projection is None:
            self.projection = SloLinear(hidden.shape[-1], target_dim, "proj")
        return _to_np(self.projection.forward(_to_tensor(hidden, requires_grad=True)))


# =============================================================================
# DistillationTrainer
# =============================================================================


class DistillationTrainer:
    """
    Trainer for knowledge distillation.

    Uses SloNet Tensor/numpy natively. Teacher/student models may return
    torch tensors — they are converted at the boundary.
    """

    def __init__(self, teacher_model, student_model, config: DistillationConfig,
                 device: Optional[str] = None):
        self.teacher = teacher_model
        self.student = student_model
        self.config = config
        self.device = device or "cpu"

        for param in self.teacher.parameters():
            if hasattr(param, 'requires_grad'):
                param.requires_grad = False
        if hasattr(self.teacher, 'eval'):
            self.teacher.eval()

        self.optimizer = SloAdam(lr=1e-4)
        self.loss_fn = DistillationLoss(config)
        self.current_stage = 0
        self.stage_weights = config.stage_weights or [1.0]

    def step(self, inputs, labels) -> Dict[str, float]:
        with no_grad():
            teacher_outputs = self.teacher(inputs)
            t_logits = teacher_outputs if isinstance(teacher_outputs, (Tensor, np.ndarray)) else _to_np(teacher_outputs[0] if isinstance(teacher_outputs, (list, tuple)) else teacher_outputs)

        student_outputs = self.student(inputs)
        s_logits = student_outputs if isinstance(student_outputs, (Tensor, np.ndarray)) else _to_np(student_outputs[0] if isinstance(student_outputs, (list, tuple)) else student_outputs)

        if _size(t_logits, 1) != _size(s_logits, 1):
            min_len = min(_size(t_logits, 1), _size(s_logits, 1))
            t_logits = t_logits[:, :min_len]
            s_logits = s_logits[:, :min_len]
            labels = _to_np(labels)[:, :min_len]

        loss, losses_dict = self.loss_fn(s_logits, t_logits, labels)
        loss = _to_tensor(np.array(loss), requires_grad=True)
        loss.backward()
        self.optimizer.step(self.student.parameters())
        for p in self.student.parameters():
            if hasattr(p, 'grad'):
                p.grad = None

        return losses_dict

    def distill_logits(self, student_logits, teacher_logits, temperature=4.0):
        s = _to_np(student_logits)
        t = _to_np(teacher_logits)
        student_soft = log_softmax(_to_tensor(s / temperature), dim=-1)
        teacher_soft = softmax(_to_tensor(t / temperature), dim=-1)
        loss = kl_div_loss(student_soft, teacher_soft, reduction="batchmean")
        loss = loss * (temperature ** 2)
        return loss

    def distill_hidden_states(self, student_hidden, teacher_hidden, projection=None):
        sh = _to_np(student_hidden) if projection is None else _to_np(projection(student_hidden))
        th = _to_np(teacher_hidden)
        return mse_loss(_to_tensor(sh, requires_grad=True), _to_tensor(th))

    def parameters(self):
        return []


class ProgressiveDistiller:
    """Progressive knowledge distillation — layer by layer."""

    def __init__(self, teacher_model, student_model, config: DistillationConfig):
        self.teacher = teacher_model
        self.student = student_model
        self.config = config
        self.teacher_layers = self._get_layers(teacher_model)
        self.student_layers = self._get_layers(student_model)
        self.layer_mapping = self._create_layer_mapping()

    def _get_layers(self, model):
        layers = []
        for name, module in (model.named_modules() if hasattr(model, 'named_modules') else model.named_children()):
            if "block" in name.lower() or "layer" in name.lower():
                try:
                    if len(list(module.children() if hasattr(module, 'children') else module.named_children())) > 0:
                        layers.append(module)
                except Exception:
                    layers.append(module)
        return layers

    def _create_layer_mapping(self) -> Dict[int, int]:
        n_t = len(self.teacher_layers)
        n_s = len(self.student_layers)
        if n_s >= n_t:
            return {i: i for i in range(n_s)}
        return {i: int(i * n_t / n_s) for i in range(n_s)}

    def distill_intermediate(self, inputs, intermediate_losses=None):
        return 0.0


def create_distillation_trainer(teacher_model, student_model, temperature=4.0,
                                alpha=0.5, beta=0.5) -> DistillationTrainer:
    config = DistillationConfig(temperature=temperature, alpha=alpha, beta=beta)
    return DistillationTrainer(teacher_model, student_model, config)


__all__ = [
    "DistillationConfig",
    "DistillationLoss",
    "DistillationTrainer",
    "ProgressiveDistiller",
    "create_distillation_trainer",
]
