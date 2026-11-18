"""
StepStrategy — the single per-step strategy seam in SloughGPTTrainer.train_step().

One training loop (``SloughGPTTrainer``) owns forward/loss/backward/clip/step;
strategies only reshape ``(loss, metrics)`` at the seam between the composed
loss and ``backward()``. EWC, the owned objective (experience/feedback),
distillation and DPO (preference pairs) are adapters over this protocol —
``MetaTrainer`` composes them, ``ComprehensiveTrainer`` keeps phase scheduling
(*when* strategies exist at all).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np

from domain.training._internal.slonet import (
    Tensor,
    kl_div_loss,
    log_softmax,
    no_grad,
    softmax,
)

logger = logging.getLogger("slo.step_strategy")


@runtime_checkable
class StepStrategy(Protocol):
    """Per-step hook run by ``SloughGPTTrainer.train_step`` after the NaN guard.

    Contract:

    - ``loss`` is the loss composed so far (primary + earlier strategies); the
      returned loss must survive ``* scale_factor`` and ``.backward()`` when a
      gradient signal is intended (add a SloNet ``Tensor`` that carries a
      graph — a plain float shifts the scalar without teaching anything).
    - ``logits`` are this batch's fresh student logits (may be ignored).
    - ``metadata`` is the trainer's experience metadata (may be ``None``).
    - returned metric keys are **final** (already namespaced) and are merged
      into the step metrics as-is.
    - a strategy that does not apply this step returns ``(loss, {})``.
    """

    name: str

    def compute(
        self,
        model: Any,
        x: np.ndarray,
        y: np.ndarray,
        loss: Any,
        *,
        logits: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, float]]: ...


class EwcStrategy:
    """EWC penalty adapter over ``EwcContinualLearner``.

    Parity with the legacy inline block in ``train_step``: applies only when
    ``task_snapshots`` is non-empty; the legacy path dropped the stats dict
    and this adapter keeps dropping it (no metric-key changes).
    """

    name = "ewc"

    def __init__(self, learner: Any) -> None:
        self.learner = learner

    def compute(
        self,
        model: Any,
        x: np.ndarray,
        y: np.ndarray,
        loss: Any,
        *,
        logits: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, float]]:
        learner = self.learner
        if learner is None or not getattr(learner, "task_snapshots", None):
            return loss, {}
        ewc_loss, _stats = learner.ewc_loss()
        return loss + ewc_loss, {}


class OwnedObjectiveStrategy:
    """Adapter over ``OwnedObjective`` (goal 13): experience/feedback terms.

    Metric keys keep their historical ``owned_`` namespace so step metrics are
    byte-identical to the pre-strategy inline path.
    """

    name = "owned_objective"

    def __init__(self, objective: Any) -> None:
        self.objective = objective

    def compute(
        self,
        model: Any,
        x: np.ndarray,
        y: np.ndarray,
        loss: Any,
        *,
        logits: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, float]]:
        if self.objective is None:
            return loss, {}
        combined, metrics = self.objective.compute(model, x, y, loss, metadata=metadata)
        return combined, {f"owned_{k}": float(v) for k, v in metrics.items()}


class DistillationStrategy:
    """Soft-label distillation against a teacher model.

    ``DistillationLoss.forward`` converts to numpy/float at its API boundary by
    design, so its total cannot carry gradients; this adapter computes the same
    temperature-scaled soft KL directly on SloNet tensors, keeping the graph
    from student logits through to ``backward()``. The teacher runs under
    ``no_grad`` (no graph, no teacher-side accumulation).
    """

    name = "distillation"

    def __init__(self, teacher: Any, *, weight: float = 1.0, temperature: float = 2.0) -> None:
        if teacher is None:
            raise ValueError("DistillationStrategy requires a teacher")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.teacher = teacher
        self.weight = float(weight)
        self.temperature = float(temperature)

    def compute(
        self,
        model: Any,
        x: np.ndarray,
        y: np.ndarray,
        loss: Any,
        *,
        logits: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, float]]:
        if self.weight <= 0 or logits is None:
            return loss, {}
        temp = self.temperature
        with no_grad():
            teacher_out = self.teacher(x, y)
            t_logits = teacher_out[0] if isinstance(teacher_out, tuple) else teacher_out
            t_soft = softmax(t_logits / temp, dim=-1)
        s_soft = log_softmax(logits / temp, dim=-1)
        soft = kl_div_loss(s_soft, t_soft, reduction="batchmean") * (temp**2)
        metrics = {"distill_soft": float(soft.item())}
        logger.debug("distill_soft=%.6f temp=%.2f", metrics["distill_soft"], temp)
        return loss + soft * self.weight, metrics


def _as_np(x: Any) -> np.ndarray:
    """Coerce a SloNet ``Tensor`` (or array-like) into its numpy buffer."""
    data = getattr(x, "data", None)
    if isinstance(data, np.ndarray):
        return data
    return np.asarray(x)


def _seq_logprobs_np(logits: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Per-example sum of selected-token log-probs: ``[B,T,V]`` + ``[B,T]`` → ``[B]``.

    Stable log-sum-exp (the ``cross_entropy`` trick) with the same target clip.
    Pure numpy by design: the reference model's contribution must never carry
    a graph — it enters the DPO loss as constants.
    """
    if logits.ndim != 3 or logits.shape[:2] != ids.shape:
        raise ValueError(
            f"log-probs need [B, T, V] logits aligned with [B, T] ids, "
            f"got {logits.shape} and {ids.shape}"
        )
    mx = logits.max(axis=-1, keepdims=True)
    diff = logits - mx
    lp = diff - np.log(np.exp(diff).sum(axis=-1, keepdims=True))
    ids = np.clip(ids, 0, logits.shape[-1] - 1)
    b, t = np.indices(ids.shape)
    return lp[b, t, ids].sum(axis=1)


def _dpo_loss_node(
    logits_ch: Any,
    y_ch: np.ndarray,
    logits_rj: Any,
    y_rj: np.ndarray,
    ref_ch: np.ndarray,
    ref_rj: np.ndarray,
    beta: float,
) -> tuple[Tensor, float]:
    """Build ``-mean log σ(β ((π_w − π_l) − (ref_w − ref_l)))`` as one loss Tensor.

    Same pattern as ``cross_entropy``: numpy forward, a scalar ``Tensor`` with
    ``_children``/``_backward_fn`` whose backward writes each side's logits
    grad directly — the topo walk then continues from those logits into the
    policy graph on to the parameters. Reference values arrive as plain floats
    and can never acquire gradients. No ``_forward_fn``: forward-mode AD has
    no consumer on this loss; ``backward()`` is the training path.
    """
    lc = _as_np(logits_ch)
    lr = _as_np(logits_rj)
    t_ch = np.asarray(y_ch, dtype=np.int64)
    t_rj = np.asarray(y_rj, dtype=np.int64)
    # Alignment of logits vs ids is validated inside _seq_logprobs_np.
    pi_ch = _seq_logprobs_np(lc, t_ch)
    pi_rj = _seq_logprobs_np(lr, t_rj)
    t_ch = np.clip(t_ch, 0, lc.shape[-1] - 1)
    t_rj = np.clip(t_rj, 0, lr.shape[-1] - 1)
    margin = (pi_ch - pi_rj) - (np.asarray(ref_ch) - np.asarray(ref_rj))
    z = beta * margin
    out = Tensor(
        float(np.logaddexp(0.0, -z).mean()),
        requires_grad=True,
        _children=(logits_ch, logits_rj),
    )
    # Stable sigmoid: exp(-log(1 + exp(-z))) — no overflow for large |z|.
    sigmoid_z = np.exp(-np.logaddexp(0.0, -z))

    def _probs(l: np.ndarray) -> np.ndarray:
        mx = l.max(axis=-1, keepdims=True)
        e = np.exp(l - mx)
        return e / e.sum(axis=-1, keepdims=True)

    p_ch = _probs(lc)
    p_rj = _probs(lr)
    b_idx, t_idx = np.indices(t_ch.shape)
    n_pairs = margin.shape[0]

    def bk(g: np.ndarray) -> None:
        # d loss/d z = (σ(z) − 1)/B · g ; d loss/d π_w = +u ; d loss/d π_l = −u ;
        # d π/d logits = (one-hot − softmax) on each selected token.
        u = beta * (sigmoid_z - 1.0) * (np.asarray(g, dtype=np.float64) / n_pairs)
        grad_ch = -u[:, None, None] * p_ch
        grad_ch[b_idx, t_idx, t_ch] += u[:, None]
        grad_rj = u[:, None, None] * p_rj
        grad_rj[b_idx, t_idx, t_rj] -= u[:, None]
        for lg, gv in ((logits_ch, grad_ch), (logits_rj, grad_rj)):
            if lg.grad is None:
                lg.grad = Tensor(gv, _copy=False)
            else:
                lg.grad.data += gv

    out._backward_fn = bk
    return out, float(margin.mean())


class DpoStrategy:
    """Direct Preference Optimization (Rafailov et al., 2023) — the DPO mode.

    Declares the step's loss from preference pairs rather than branching the
    loop: ``pairs()`` yields ``(chosen, rejected)`` full token sequences
    ``[B, T]`` — or ``None`` when exhausted, in which case the strategy is a
    no-op — each split into the trainer's next-token convention
    ``(x, y) = (ids[:, :-1], ids[:, 1:])``. The loss is the mean negative
    log-sigmoid of the β-scaled implicit-reward margin between policy and
    reference sequence log-probs.

    - The reference runs under ``no_grad`` and its log-probs are baked in as
      floats: it never enters the autograd graph (no ref-side accumulation;
      put it in ``eval()`` yourself — the strategy never flips model modes).
    - Default ``sft_weight=0.0``: DPO *replaces* the step's training signal
      (the strategy declares the loss, so the loop's SFT term is computed by
      the one forward but not trained on); ``sft_weight > 0`` composes
      ``loss * sft_weight + dpo`` instead.
    - Metrics ``dpo_loss``/``dpo_margin`` are final (namespaced) and report
      the raw term — composition weights never leak into them.

    The RM card's Bradley–Terry objective is a sibling strategy over this
    same seam, not a flag on this class.
    """

    name = "dpo"

    def __init__(
        self,
        ref_model: Any,
        pairs: Callable[[], tuple[Any, Any] | None],
        *,
        beta: float = 0.1,
        sft_weight: float = 0.0,
    ) -> None:
        if ref_model is None:
            raise ValueError("DpoStrategy requires a reference model")
        if not callable(pairs):
            raise ValueError(
                f"pairs must be callable → (chosen, rejected) or None, got {type(pairs).__name__}"
            )
        if not beta > 0:
            raise ValueError(f"beta must be > 0, got {beta}")
        self.ref_model = ref_model
        self.pairs = pairs
        self.beta = float(beta)
        self.sft_weight = float(sft_weight)

    @staticmethod
    def _validate_pairs(batch: Any) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise ValueError(
                f"pairs() must yield (chosen, rejected) or None, got {type(batch).__name__}"
            )
        chosen, rejected = _as_np(batch[0]), _as_np(batch[1])
        if chosen.ndim != 2 or rejected.ndim != 2:
            raise ValueError(
                f"pair sequences must be 2-D [B, T], got {chosen.ndim}-D and {rejected.ndim}-D"
            )
        if chosen.shape != rejected.shape:
            raise ValueError(f"pair shapes must match, got {chosen.shape} vs {rejected.shape}")
        if chosen.shape[1] < 2:
            raise ValueError(
                f"pair sequences need at least two tokens to form (x, y), got T={chosen.shape[1]}"
            )
        if not np.issubdtype(chosen.dtype, np.integer) or not np.issubdtype(
            rejected.dtype, np.integer
        ):
            raise ValueError(
                f"pair sequences must be integer token ids, got {chosen.dtype} and {rejected.dtype}"
            )
        return chosen, rejected

    def compute(
        self,
        model: Any,
        x: np.ndarray,
        y: np.ndarray,
        loss: Any,
        *,
        logits: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, float]]:
        batch = self.pairs()
        if batch is None:
            return loss, {}
        chosen, rejected = self._validate_pairs(batch)
        x_ch, y_ch = chosen[:, :-1], chosen[:, 1:]
        x_rj, y_rj = rejected[:, :-1], rejected[:, 1:]

        out_ch = model(x_ch, y_ch)
        logits_ch = out_ch[0] if isinstance(out_ch, tuple) else out_ch
        out_rj = model(x_rj, y_rj)
        logits_rj = out_rj[0] if isinstance(out_rj, tuple) else out_rj
        with no_grad():
            ref_out_ch = self.ref_model(x_ch, y_ch)
            ref_out_rj = self.ref_model(x_rj, y_rj)
        ref_ch = ref_out_ch[0] if isinstance(ref_out_ch, tuple) else ref_out_ch
        ref_rj = ref_out_rj[0] if isinstance(ref_out_rj, tuple) else ref_out_rj

        node, margin = _dpo_loss_node(
            logits_ch,
            y_ch,
            logits_rj,
            y_rj,
            _seq_logprobs_np(_as_np(ref_ch), y_ch),
            _seq_logprobs_np(_as_np(ref_rj), y_rj),
            self.beta,
        )
        metrics = {"dpo_loss": float(node.item()), "dpo_margin": float(margin)}
        logger.debug(
            "dpo_loss=%.6f margin=%.4f beta=%.3f sft_weight=%.2f",
            metrics["dpo_loss"],
            metrics["dpo_margin"],
            self.beta,
            self.sft_weight,
        )
        if self.sft_weight > 0:
            return loss * self.sft_weight + node, metrics
        return node, metrics
