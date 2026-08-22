"""
Truth maintainer — self-retrain on misclassified texts post-epoch.

After each training epoch, the maintainer identifies texts whose rule-based
label disagrees with their nearest meaning point, then generates corrective
contrastive pairs to pull misclassified texts toward their correct region.

Usage:
    from domains.infrastructure.truth_maintainer import TruthMaintainer
    maintainer = TruthMaintainer()
    maintainer.run_maintenance(encoder, texts, meaning_tags, optimizer)
"""
import threading

import numpy as np
from typing import Dict, List, Optional, Tuple

from .truth_labeler import TruthLabeler, get_truth_labeler


class TruthMaintainer:
    """Self-retrain on misclassified texts post-epoch.

    Identifies texts whose rule-based label disagrees with their embedding's
    nearest meaning point, then generates corrective contrastive pairs to
    pull misclassified texts toward their correct semantic region.
    """

    def __init__(self, labeler: Optional[TruthLabeler] = None):
        self._labeler = labeler or get_truth_labeler()

    def find_misclassified(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        meaning_tags,
    ) -> List[Dict]:
        """Find texts whose rule-based label disagrees with embedding nearest point.

        Args:
            texts: list of input texts
            embeddings: (N, D) L2-normalized embedding array
            meaning_tags: MeaningTags instance

        Returns:
            List of dicts with keys: index, text, rule_label, embed_label, confidence
        """
        misclassified = []
        for i, text in enumerate(texts):
            rule_result = self._labeler.label(text)
            embed_label = meaning_tags.classify(embeddings[i].tolist())

            if rule_result.label != embed_label and rule_result.confidence > 0.4:
                misclassified.append({
                    "index": i,
                    "text": text,
                    "rule_label": rule_result.label,
                    "embed_label": embed_label,
                    "confidence": rule_result.confidence,
                })
        return misclassified

    def generate_corrective_pairs(
        self,
        misclassified: List[Dict],
        all_texts: List[str],
        all_embeddings: np.ndarray,
        meaning_tags,
        max_pairs: int = 50,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate corrective contrastive pairs for misclassified texts.

        For each misclassified text, finds texts of the SAME rule-label
        (positives) and texts of the WRONG embed-label (negatives).

        Args:
            misclassified: output of find_misclassified()
            all_texts: all training texts
            all_embeddings: all embeddings (N, D)
            meaning_tags: MeaningTags instance
            max_pairs: maximum pairs to generate

        Returns:
            Tuple of (query_texts, positive_texts, negative_texts)
        """
        queries = []
        positives = []
        negatives = []

        # Build index by embed-label
        label_to_indices: Dict[str, List[int]] = {}
        for i, emb in enumerate(all_embeddings):
            label = meaning_tags.classify(emb.tolist())
            label_to_indices.setdefault(label, []).append(i)

        for item in misclassified[:max_pairs]:
            idx = item["index"]
            rule_label = item["rule_label"]
            embed_label = item["embed_label"]

            # Find texts with same rule-label (should be close)
            rule_indices = [
                i for i, t in enumerate(all_texts)
                if self._labeler.label(t).label == rule_label and i != idx
            ]

            # Find texts with wrong embed-label (should be far)
            wrong_indices = label_to_indices.get(embed_label, [])
            wrong_indices = [i for i in wrong_indices if i != idx]

            if rule_indices and wrong_indices:
                queries.append(all_texts[idx])
                positives.append(all_texts[rule_indices[0]])
                negatives.append(all_texts[wrong_indices[0]])

        return queries, positives, negatives

    def apply_correction(
        self,
        encoder,
        query_texts: List[str],
        positive_texts: List[str],
        negative_texts: List[str],
        meaning_tags,
        vocab,
        encode_fn=None,
        max_seq_len: int = 64,
        lr: float = 3e-4,
    ) -> float:
        """Apply corrective gradient step to pull queries toward positives, away from negatives.

        Uses simple contrastive gradient: query should be close to positive,
        far from negative. This corrects misclassification by pushing the
        embedding toward the correct semantic region.

        Args:
            encoder: SloNet encoder
            query_texts, positive_texts, negative_texts: corrective pairs
            meaning_tags: MeaningTags (unused, reserved for future constraint)
            vocab: vocabulary dict
            encode_fn: optional BPE encode function
            max_seq_len: max sequence length
            lr: learning rate

        Returns:
            loss value
        """
        if not query_texts:
            return 0.0

        from domains.training.slonet import Tensor, SloAdam

        params = encoder.parameters()
        optimizer = SloAdam(lr=lr)

        B = len(query_texts)

        # Encode all three views
        if encode_fn is not None:
            query_ids = np.stack([encode_fn(t, max_seq_len) for t in query_texts])
            pos_ids = np.stack([encode_fn(t, max_seq_len) for t in positive_texts])
            neg_ids = np.stack([encode_fn(t, max_seq_len) for t in negative_texts])
        else:
            query_ids = np.stack([_encode_tokens(t, vocab, max_seq_len) for t in query_texts])
            pos_ids = np.stack([_encode_tokens(t, vocab, max_seq_len) for t in positive_texts])
            neg_ids = np.stack([_encode_tokens(t, vocab, max_seq_len) for t in negative_texts])

        query_emb = encoder.forward(query_ids)
        pos_emb = encoder.forward(pos_ids)
        neg_emb = encoder.forward(neg_ids)

        # L2-normalize
        q_norm = query_emb.data / (np.linalg.norm(query_emb.data, axis=1, keepdims=True) + 1e-10)
        p_norm = pos_emb.data / (np.linalg.norm(pos_emb.data, axis=1, keepdims=True) + 1e-10)
        n_norm = neg_emb.data / (np.linalg.norm(neg_emb.data, axis=1, keepdims=True) + 1e-10)

        # Triplet loss: query close to positive, far from negative
        pos_sim = np.sum(q_norm * p_norm, axis=1)  # (B,)
        neg_sim = np.sum(q_norm * n_norm, axis=1)  # (B,)
        margin = 0.3
        loss = np.maximum(0.0, margin + pos_sim - neg_sim).mean()

        # Gradient: d(loss)/d(query) pulls toward positive, pushes from negative
        grad_q = ((neg_sim - pos_sim - margin) > 0).astype(np.float64)[:, np.newaxis]
        grad_query = grad_q * (p_norm - n_norm) / B

        # Backprop through L2 norm
        norms = np.linalg.norm(query_emb.data, axis=1, keepdims=True) + 1e-10
        grad_emb = (grad_query / norms) - (query_emb.data * (grad_query * query_emb.data).sum(axis=1, keepdims=True) / (norms ** 3))

        query_emb.grad = Tensor(grad_emb)
        query_emb.backward()

        # Clip and step
        for p in params:
            if p.grad is not None:
                g = p.grad.data
                norm = np.linalg.norm(g)
                if norm > 1.0:
                    p.grad.data = g / norm
        optimizer.step(params)

        return float(loss)


def _encode_tokens(text: str, vocab: dict, max_len: int) -> np.ndarray:
    """Fallback whitespace tokenization."""
    tokens = text.lower().split()[:max_len]
    ids = [vocab.get(t, vocab.get("<unk>", 0)) for t in tokens]
    padded = np.zeros(max_len, dtype=np.int64)
    padded[:len(ids)] = ids
    return padded


# Module-level singleton
_maintainer: Optional[TruthMaintainer] = None
_maintainer_lock = threading.Lock()


def get_truth_maintainer() -> TruthMaintainer:
    """Get or create the truth maintainer singleton."""
    global _maintainer
    if _maintainer is None:
        with _maintainer_lock:
            if _maintainer is None:
                _maintainer = TruthMaintainer()
    return _maintainer


def reset_truth_maintainer() -> None:
    """Reset the singleton (for testing)."""
    global _maintainer
    with _maintainer_lock:
        _maintainer = None
