"""
Meta-weight manager for live feedback-based generation adjustment.

Uses feedback database to retrieve similar good responses and
adjust generation parameters accordingly.
"""

import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

from .database import FeedbackDB, get_feedback_db, SimilarPattern


@dataclass
class MetaWeights:
    """Adjustable weights for generation."""

    temperature: float = 0.8
    repetition_penalty: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
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
                print(
                    f"MetaWeightManager: Using sentence-transformers embeddings (dim={self.embedding_dim})"
                )
            except ImportError:
                print(
                    "MetaWeightManager: sentence-transformers not available, using simple embeddings"
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
            print(f"Embedding error: {e}, falling back to simple")
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
        """Aggregate patterns to get adjustment values."""
        if not patterns:
            return {}

        # Calculate weighted average based on similarity
        weighted_temp = 0.0
        weighted_rep = 0.0
        total_weight = 0.0

        for p in patterns:
            weight = p.similarity

            if p.rating == "thumbs_up":
                # Good responses: slightly more creative, less repetition
                weighted_temp += weight * 0.05
                weighted_rep += weight * -0.05
            else:
                # Bad responses: opposite
                weighted_temp += weight * -0.05
                weighted_rep += weight * 0.05

            total_weight += weight

        if total_weight > 0:
            return {
                "temperature_boost": weighted_temp / total_weight,
                "repetition_boost": weighted_rep / total_weight,
            }
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
            # Get per-user meta weights first
            user_weights = self.db.get_user_meta_weights(user_id)
            if user_weights:
                weights.temperature += user_weights.get("temperature_boost", 0)
                weights.repetition_penalty += user_weights.get("repetition_boost", 0)

            # Generate embedding for the message
            query_embedding = self._embed(user_message)

            # Find similar messages using vector search
            patterns = self.db.find_similar_messages(
                query_embedding, k=k, rating=rating, min_similarity=0.3
            )

            # If no vector results, fall back to text search
            if not patterns and self.use_simple_search:
                patterns = self.db.find_similar_by_text(user_message, k=k, rating=rating)

            # Aggregate patterns
            adjustments = self._aggregate_patterns(patterns)

            # Apply adjustments
            weights.temperature += adjustments.get("temperature_boost", 0)
            weights.repetition_penalty += adjustments.get("repetition_boost", 0)

            # Clamp to reasonable ranges
            weights.temperature = max(0.1, min(2.0, weights.temperature))
            weights.repetition_penalty = max(0.8, min(1.5, weights.repetition_penalty))

            # Store in history
            self._weight_history.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "temperature": weights.temperature,
                    "repetition_penalty": weights.repetition_penalty,
                    "pattern_count": len(patterns),
                }
            )

            # Keep only recent history
            if len(self._weight_history) > 100:
                self._weight_history = self._weight_history[-50:]

        except Exception as e:
            print(f"MetaWeightManager warning: {e}")

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
            )
        except Exception as e:
            print(f"Warning: Could not update user meta-weights: {e}")

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

        # Calculate average weights from history
        avg_temp = 0.0
        avg_rep = 0.0
        if self._weight_history:
            avg_temp = sum(w["temperature"] for w in self._weight_history) / len(
                self._weight_history
            )
            avg_rep = sum(w["repetition_penalty"] for w in self._weight_history) / len(
                self._weight_history
            )

        return {
            "db_stats": db_stats,
            "quality_trend": trend,
            "current_weights": {
                "temperature": avg_temp or self._default_weights.temperature,
                "repetition_penalty": avg_rep or self._default_weights.repetition_penalty,
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
