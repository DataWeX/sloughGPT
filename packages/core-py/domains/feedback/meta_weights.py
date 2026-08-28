"""
Meta-weight manager for live feedback-based generation adjustment.

Uses feedback database to retrieve similar good responses and
adjust generation parameters accordingly.
"""

import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timezone

import logging

from .database import get_feedback_db, SimilarPattern

logger = logging.getLogger("slo.feedback.meta_weights")


@dataclass
class MetaWeights:
    """Adjustable weights for generation."""

    temperature: float = 0.7
    repetition_penalty: float = 1.15
    top_p: float = 0.85
    top_k: int = 40
    length_penalty: float = 1.0
    style_bias: float = 0.0  # -1 to 1, creative to conservative
    confidence_boost: float = 0.0  # increase for more confident responses


class MetaWeightManager:
    """
    Manages meta-weights based on user feedback.

    Retrieves similar past good responses and adjusts generation
    parameters to bias towards patterns that worked well.
    """

    def __init__(
        self,
        db_path: str = "data/feedback.db",
        embedding_dim: int = 384,
        use_simple_search: bool = True,
    ):
        self.db = get_feedback_db(db_path)
        self.embedding_dim = embedding_dim
        self.use_simple_search = use_simple_search

        # Running averages for meta-weights
        self._weight_history: List[Dict[str, float]] = []
        self._default_weights = MetaWeights()

        # Decay factor for historical weights (higher = more weight on recent)
        self.decay_factor = 0.9

        # Embedding model (lazy loaded)
        self._embed_model = None
        self._embedder = None

    def _get_embedder(self):
        """Lazy load embedding model (sentence-transformers)."""
        if self._embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.embedding_dim = 384
                logger.info(
                    "MetaWeightManager: Using sentence-transformers embeddings (dim=%d)", self.embedding_dim, extra={"tag": "INFRA"}
                )
            except ImportError:
                logger.warning(
                    "MetaWeightManager: sentence-transformers not available, using simple embeddings", extra={"tag": "INFRA"}
                )
                self._embed_model = "simple"
        return self._embed_model

    def _embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.
        Uses sentence-transformers if available, falls back to simple hash.
        """
        embedder = self._get_embedder()

        if embedder == "simple" or embedder is None:
            return self._simple_embed(text)

        try:
            # SentenceTransformer returns (1, dim) array
            embedding = embedder.encode(text, convert_to_numpy=True, show_progress_bar=False)
            if len(embedding.shape) > 1:
                embedding = embedding[0]
            return embedding.astype(np.float32)
        except Exception as e:
            logger.warning("Embedding error: %s, falling back to simple", e, extra={"tag": "INFRA"})
            return self._simple_embed(text)

    def _simple_embed(self, text: str) -> np.ndarray:
        """
        Simple embedding using word hash (fallback when no sentence-transformers).
        """
        words = text.lower().split()
        vector = np.zeros(self.embedding_dim)

        for i, word in enumerate(words[: self.embedding_dim]):
            vector[i] = hash(word) % 100 / 100.0

        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def _aggregate_patterns(self, patterns: List[SimilarPattern]) -> Dict[str, float]:
        """Aggregate patterns to get adjustment values.

        Each similar message adjusts generation parameters based on
        whether it received thumbs_up or thumbs_down:

        - temperature: ↑ creativity on good, ↓ on bad
        - repetition_penalty: ↓ on good (allow some repetition), ↑ on bad
        - top_p: ↑ (more diverse sampling) on good, ↓ on bad
        - top_k: ↓ (tighter focus) on good, ↑ (broader) on bad
        - style_bias: ↑ (creative) on good, ↓ (conservative) on bad
        - confidence_boost: ↑ on good, ↓ on bad
        """
        if not patterns:
            return {}

        weighted: Dict[str, float] = {
            "temperature_boost": 0.0,
            "repetition_boost": 0.0,
            "top_p_boost": 0.0,
            "top_k_boost": 0.0,
            "style_bias": 0.0,
            "confidence_boost": 0.0,
        }
        total_weight = 0.0

        for p in patterns:
            w = p.similarity
            total_weight += w

            if p.rating == "thumbs_up":
                weighted["temperature_boost"] += w * 0.05
                weighted["repetition_boost"] += w * -0.05
                weighted["top_p_boost"] += w * 0.03
                weighted["top_k_boost"] += w * -2
                weighted["style_bias"] += w * 0.05
                weighted["confidence_boost"] += w * 0.03
            else:
                weighted["temperature_boost"] += w * -0.05
                weighted["repetition_boost"] += w * 0.05
                weighted["top_p_boost"] += w * -0.03
                weighted["top_k_boost"] += w * 2
                weighted["style_bias"] += w * -0.05
                weighted["confidence_boost"] += w * -0.03

        if total_weight > 0:
            return {k: v / total_weight for k, v in weighted.items()}
        return {}

    def get_adjustment(
        self,
        user_message: str,
        k: int = 5,
        rating: Optional[str] = "thumbs_up",
        user_id: str = "default",
    ) -> MetaWeights:
        """
        Get meta-weight adjustment based on similar past feedback.

        Combines two signals:
        1. Per-user accumulated boosts (from all historical feedback)
        2. Pattern-based adjustments (from k nearest similar messages)

        Args:
            user_message: Current user message
            k: Number of similar patterns to retrieve
            rating: Which rating to prioritize ("thumbs_up", "thumbs_down", or None)
            user_id: User identifier for per-user weights

        Returns:
            MetaWeights with adjustments to apply
        """
        weights = MetaWeights(
            temperature=self._default_weights.temperature,
            repetition_penalty=self._default_weights.repetition_penalty,
            top_p=self._default_weights.top_p,
            top_k=self._default_weights.top_k,
        )

        try:
            # Layer 1: per-user accumulated boosts
            user_weights = self.db.get_user_meta_weights(user_id)
            if user_weights:
                weights.temperature += user_weights.get("temperature_boost", 0)
                weights.repetition_penalty += user_weights.get("repetition_boost", 0)
                weights.top_p += user_weights.get("top_p_boost", 0)
                weights.top_k += int(user_weights.get("top_k_boost", 0))
                weights.style_bias += user_weights.get("style_bias", 0)
                weights.confidence_boost += user_weights.get("confidence_boost", 0)

            # Layer 2: pattern-based adjustments from similar messages
            query_embedding = self._embed(user_message)

            patterns = self.db.find_similar_messages(
                query_embedding, k=k, rating=rating, min_similarity=0.3
            )

            if not patterns and self.use_simple_search:
                patterns = self.db.find_similar_by_text(user_message, k=k, rating=rating)

            adjustments = self._aggregate_patterns(patterns)

            weights.temperature += adjustments.get("temperature_boost", 0)
            weights.repetition_penalty += adjustments.get("repetition_boost", 0)
            weights.top_p += adjustments.get("top_p_boost", 0)
            weights.top_k += int(adjustments.get("top_k_boost", 0))
            weights.style_bias += adjustments.get("style_bias", 0)
            weights.confidence_boost += adjustments.get("confidence_boost", 0)

            # Clamp to safe ranges
            weights.temperature = max(0.1, min(1.5, weights.temperature))
            weights.repetition_penalty = max(0.8, min(1.3, weights.repetition_penalty))
            weights.top_p = max(0.1, min(1.0, weights.top_p))
            weights.top_k = max(5, min(200, weights.top_k))
            weights.style_bias = max(-1.0, min(1.0, weights.style_bias))
            weights.confidence_boost = max(-1.0, min(1.0, weights.confidence_boost))

            # Store in history
            self._weight_history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "temperature": weights.temperature,
                "repetition_penalty": weights.repetition_penalty,
                "top_p": weights.top_p,
                "top_k": weights.top_k,
                "style_bias": weights.style_bias,
                "confidence_boost": weights.confidence_boost,
                "pattern_count": len(patterns),
            })

            if len(self._weight_history) > 100:
                self._weight_history = self._weight_history[-50:]

        except Exception as e:
            logger.warning("MetaWeightManager: %s", e, extra={"tag": "INFRA"})

        return weights

    def record_feedback(
        self,
        user_message: str,
        assistant_response: str,
        rating: str,
        conversation_id: Optional[str] = None,
        quality_score: Optional[float] = None,
        user_id: str = "default",
    ) -> str:
        """
        Record feedback and update meta-weights.

        Args:
            user_message: The user's message
            assistant_response: The assistant's response
            rating: "thumbs_up" or "thumbs_down"
            conversation_id: Optional conversation ID
            quality_score: Optional 0-1 quality score
            user_id: User identifier for per-user meta-weights

        Returns:
            Feedback ID
        """
        # Create conversation if needed
        if conversation_id is None:
            conversation_id = self.db.create_conversation(user_id=user_id)

        # Add messages with embeddings
        user_msg_id = self.db.add_message(
            conversation_id=conversation_id,
            role="user",
            content=user_message,
            embedding=self._embed(user_message),
        )

        assistant_id = self.db.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_response,
            embedding=self._embed(assistant_response),
        )

        # Add feedback
        context = f"{user_message[:100]} -> {assistant_response[:100]}"
        feedback_id = self.db.add_feedback(
            message_id=assistant_id,
            rating=rating,
            quality_score=quality_score,
            context_snippet=context,
        )

        # Update user meta-weights
        try:
            self.db.update_user_meta_weights(
                user_id=user_id,
                rating=rating,
                temperature_delta=0.02,
                repetition_delta=0.02,
                top_p_delta=0.01,
                top_k_delta=1,
                style_bias_delta=0.02,
                confidence_delta=0.01,
            )
        except Exception as e:
            logger.warning("Could not update user meta-weights: %s", e, extra={"tag": "INFRA"})

        return feedback_id

    def get_quality_trend(self, window: int = 10) -> Dict[str, float]:
        """Get quality trend from recent feedback."""
        feedback = self.db.get_all_feedback(rating=None, limit=window)

        if not feedback:
            return {"trend": 0.0, "thumbs_up_ratio": 0.0}

        thumbs_up = sum(1 for f in feedback if f["rating"] == "thumbs_up")
        return {
            "trend": thumbs_up / len(feedback),
            "thumbs_up_ratio": thumbs_up / len(feedback),
            "sample_count": len(feedback),
        }

    def export_training_data(self, filepath: str, format: str = "jsonl"):
        """Export feedback as training data."""
        if format == "jsonl":
            self.db.export_feedback_jsonl(filepath)
        elif format == "dpo":
            self.db.export_dpo_format(filepath)

    def get_stats(self) -> Dict[str, Any]:
        """Get meta-weight statistics."""
        db_stats = self.db.get_stats()
        trend = self.get_quality_trend()

        avg = {field: 0.0 for field in (
            "temperature", "repetition_penalty", "top_p", "top_k",
            "style_bias", "confidence_boost",
        )}
        if self._weight_history:
            n = len(self._weight_history)
            for field in avg:
                avg[field] = sum(w.get(field, 0) for w in self._weight_history) / n

        return {
            "db_stats": db_stats,
            "quality_trend": trend,
            "current_weights": {
                "temperature": avg["temperature"] or self._default_weights.temperature,
                "repetition_penalty": avg["repetition_penalty"] or self._default_weights.repetition_penalty,
                "top_p": avg["top_p"] or self._default_weights.top_p,
                "top_k": int(avg["top_k"]) or self._default_weights.top_k,
                "style_bias": avg["style_bias"],
                "confidence_boost": avg["confidence_boost"],
            },
            "history_length": len(self._weight_history),
        }


# Global instance
_meta_weight_manager: Optional[MetaWeightManager] = None


def get_meta_weight_manager() -> MetaWeightManager:
    """Get or create the global meta-weight manager."""
    global _meta_weight_manager
    if _meta_weight_manager is None:
        _meta_weight_manager = MetaWeightManager()
    return _meta_weight_manager
