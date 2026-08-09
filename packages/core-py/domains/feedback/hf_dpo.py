"""
HFDPOTrainer — Direct Preference Optimization on user feedback pairs.

Builds (chosen, rejected) text pairs from the user feedback store and applies
the DPO preference loss to a SloNet model when a trainable model is provided.

The trainer is intentionally torch-free: preference gradients run through
SloNet's numpy autograd. When the supplied model is not SloNet-trainable the
trainer reports an honest ``rejected`` result instead of fabricating metrics.
"""
from typing import Dict, List, Optional
import time

import numpy as np

from domains.feedback.database import get_feedback_db

DPO_BETA = 0.1
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 2


class HFDPOTrainer:
    """
    Direct Preference Optimization trainer aligned to feedback records.

    Args:
        model: a SloNet model with ``forward(input_ids)``, ``parameters()``,
            and ``vocab_size`` (optional; untrainable models are detected).
        tokenizer: object exposing ``encode(text) -> List[int]`` if present.
        learning_rate: SGD learning rate for the preference update.
        beta: DPO temperature scaling the log-probability margin.

    Returns:
        dict with ``status``, ``steps``, ``avg_loss``, ``ppl_before``,
        ``ppl_after``, ``ppl_delta_pct``, ``pairs_trained``,
        ``elapsed_seconds``.

    Side effects:
        - calls ``get_feedback_db().get_all_feedback()`` for pair building
        - mutates the supplied model weights in-place when trainable
    """

    def __init__(
        self,
        model,
        tokenizer,
        learning_rate: float = DEFAULT_LR,
        beta: float = DPO_BETA,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.beta = beta

    # ── pair building ────────────────────────────────────────────────
    def prepare_dpo_pairs(self, max_pairs: Optional[int] = None) -> List[Dict]:
        """
        Pair each thumbs-down message with a thumbs-up message.

        Prefers a thumbs-up from the same conversation; falls back to any
        distinct thumbs-up when none shares the conversation.

        Args:
            max_pairs: cap on the number of pairs returned.

        Returns:
            list of ``{"chosen": str, "rejected": str}`` dicts.

        Side effects:
            - reads the feedback SQLite store
        """
        db = get_feedback_db()
        chosen = db.get_all_feedback(rating="thumbs_up", limit=200)
        rejected = db.get_all_feedback(rating="thumbs_down", limit=200)
        pairs: List[Dict] = []
        for rej in rejected:
            content_rej = (rej.get("content") or "").strip()
            if not content_rej:
                continue
            conv = rej.get("conversation_id")
            match = None
            for c in chosen:
                content_c = (c.get("content") or "").strip()
                if content_c and content_c != content_rej:
                    if conv is not None and c.get("conversation_id") == conv:
                        match = c
                        break
                    if match is None:
                        match = c
            if match is None:
                continue
            pairs.append(
                {"chosen": (match["content"] or "").strip(), "rejected": content_rej}
            )
        if max_pairs is not None and max_pairs > 0:
            pairs = pairs[:max_pairs]
        return pairs

    # ── capability detection ─────────────────────────────────────────
    @staticmethod
    def _is_trainable(model) -> bool:
        """True when the model exposes a SloNet trainable surface."""
        if model is None:
            return False
        params = getattr(model, "parameters", None)
        if params is None or not callable(params):
            return False
        try:
            params = list(params())
        except Exception:
            return False
        return len(params) > 0 and callable(getattr(model, "forward", None))

    def _encode(self, text: str) -> np.ndarray:
        """Encode text to integer ids via the tokenizer or a char vocab."""
        tokenizer = self.tokenizer
        if tokenizer is not None:
            for name in ("encode", "tokenize", "encode_as_ids"):
                fn = getattr(tokenizer, name, None)
                if callable(fn):
                    try:
                        ids = fn(text)
                        if isinstance(ids, (list, tuple, np.ndarray)) and len(ids) > 0:
                            return np.array(ids, dtype=np.int64)
                    except Exception:
                        break
        ids = [ord(ch) for ch in text if ord(ch) < 128]
        return np.array(ids, dtype=np.int64)

    def _log_probs(self, ids: np.ndarray) -> float:
        """Sum of per-token log-probabilities (numpy, no autograd graph)."""
        if len(ids) == 0:
            return 0.0
        from domains.training.slonet import Tensor

        logits = self.model.forward(Tensor(np.array(ids, dtype=np.int64).reshape(1, -1)))
        if isinstance(logits, tuple):
            logits = logits[0]
        arr = np.asarray(getattr(logits, "data", logits), dtype=np.float32)
        lp = arr[0] if arr.ndim == 3 else arr
        if lp.ndim == 1:
            lp = lp.reshape(1, -1)
        lp = lp - np.max(lp, axis=-1, keepdims=True)
        logp = lp - np.log(np.exp(lp).sum(axis=-1, keepdims=True))
        if logp.shape[0] == 1:
            return float(logp[0, ids].sum())
        return float(logp[np.arange(len(ids)), ids].sum())

    # ── training ─────────────────────────────────────────────────────
    def train(
        self,
        pairs: Optional[List[Dict]] = None,
        max_pairs: Optional[int] = None,
    ) -> Dict:
        """
        Run DPO over the preference pairs.

        Args:
            pairs: optional prebuilt (chosen, rejected) pairs; when omitted the
                pairs are built from the feedback store.
            max_pairs: cap on the number of pairs used.

        Returns:
            result dict (see class docstring). ``status`` is ``"accepted"``
            after a real preference update, or ``"rejected"`` with a reason
            when fewer than two pairs or no trainable model is available.

        Side effects:
            - updates the model weights in-place when trainable
        """
        import time

        t0 = time.time()

        if pairs is None:
            pairs = self.prepare_dpo_pairs(max_pairs=max_pairs)
        elif max_pairs is not None and max_pairs > 0:
            pairs = pairs[:max_pairs]

        if len(pairs) < 2:
            return self._reject(
                "need at least 2 preference pairs",
                elapsed=round(time.time() - t0, 1),
            )

        if not self._is_trainable(self.model):
            return self._reject(
                "no trainable SloNet model loaded",
                elapsed=round(time.time() - t0, 1),
            )

        return self._train_slonet(pairs, t0)

    def _reject(self, reason: str, elapsed: float) -> Dict:
        """Honest rejection result (no fabricated metrics)."""
        return {
            "status": "rejected",
            "reason": reason,
            "steps": 0,
            "avg_loss": None,
            "ppl_before": None,
            "ppl_after": None,
            "ppl_delta_pct": None,
            "pairs_trained": 0,
            "elapsed_seconds": elapsed,
        }

    def _train_slonet(self, pairs: List[Dict], t0: float) -> Dict:
        """Real DPO preference gradient updates on the SloNet model."""
        from domains.training.slonet import Tensor, SloSGD, cross_entropy

        model = self.model
        vocab_size = getattr(model, "vocab_size", None)
        encoded = []
        for pair in pairs:
            chosen_ids = self._encode(pair["chosen"])
            rejected_ids = self._encode(pair["rejected"])
            if len(chosen_ids) == 0 or len(rejected_ids) == 0:
                continue
            if vocab_size:
                chosen_ids = chosen_ids[chosen_ids < vocab_size]
                rejected_ids = rejected_ids[rejected_ids < vocab_size]
                if len(chosen_ids) == 0 or len(rejected_ids) == 0:
                    continue
            encoded.append((chosen_ids, rejected_ids))

        if len(encoded) < 2:
            return self._reject(
                "no usable pairs after encoding",
                elapsed=round(time.time() - t0, 1),
            )

        model.train() if hasattr(model, "train") else None
        params = list(model.parameters())

        ref_logp_chosen = [self._log_probs(ids[0]) for ids in encoded]
        ref_logp_rejected = [self._log_probs(ids[1]) for ids in encoded]
        ppl_before = float(np.exp(-np.mean(ref_logp_chosen)))

        optimizer = SloSGD(lr=self.learning_rate)
        losses: List[float] = []

        for epoch in range(DEFAULT_EPOCHS):
            epoch_losses: List[float] = []
            for (chosen_ids, rejected_ids), rc, rr in zip(
                encoded, ref_logp_chosen, ref_logp_rejected
            ):
                chosen = self._forward_logprobs(chosen_ids)
                rejected = self._forward_logprobs(rejected_ids)
                margin = (chosen - Tensor(rc)) - (rejected - Tensor(rr))
                # -log sigmoid(beta*margin) == -log_softmax([beta*margin, 0])[0]
                beta_margin = margin * self.beta
                # Build the two-class logits [beta*margin, 0] via broadcast with
                # a constant one-hot row (stack/concatenate are non-differentiable).
                pair = beta_margin * Tensor(np.array([[1.0, 0.0]]))
                target = Tensor(np.array([0], dtype=np.int64))
                loss = cross_entropy(pair, target)
                loss.backward()
                grads = [p for p in params if p.grad is not None]
                if grads:
                    optimizer.step(grads)
                for p in params:
                    p.grad = None
                epoch_losses.append(float(loss.data))
            losses.append(float(np.mean(epoch_losses)))

        ref_logp_chosen_after = [self._log_probs(ids[0]) for ids in encoded]
        ppl_after = float(np.exp(-np.mean(ref_logp_chosen_after)))
        ppl_delta_pct = 0.0
        if ppl_before > 0:
            ppl_delta_pct = round((ppl_after - ppl_before) / ppl_before * 100.0, 1)

        return {
            "status": "accepted",
            "reason": None,
            "steps": len(encoded) * DEFAULT_EPOCHS,
            "avg_loss": round(float(np.mean(losses)), 4),
            "ppl_before": round(ppl_before, 2),
            "ppl_after": round(ppl_after, 2),
            "ppl_delta_pct": ppl_delta_pct,
            "pairs_trained": len(encoded),
            "elapsed_seconds": round(time.time() - t0, 1),
        }

    def _forward_logprobs(self, ids: np.ndarray):
        """Sum of per-token log-probabilities as a differentiable Tensor."""
        from domains.training.slonet import Tensor, cross_entropy

        logits = self.model.forward(Tensor(ids.reshape(1, -1)))
        if isinstance(logits, tuple):
            logits = logits[0]
        target = Tensor(ids.reshape(1, -1))
        return -cross_entropy(logits, target) * float(len(ids))
