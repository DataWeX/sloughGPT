"""
Feedback database with vector search for meta-weight learning.

Stores conversations, messages, feedback, and uses embeddings
to retrieve similar good responses for biasing generation.

Backed by MogDB (the project's embedded document database). Embeddings
are stored as JSON lists of floats (``float32`` values); there are no SQL
joins in MogDB, so feedback↔message relations are resolved with two-step
Python lookups that preserve the original JOIN semantics.
"""

import json
import uuid
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import numpy as np

from mogdb import MogDB


@dataclass
class Message:
    id: str
    conversation_id: str
    role: str  # "user" or "assistant"
    content: str
    embedding: Optional[np.ndarray] = None
    created_at: Optional[str] = None


@dataclass
class Feedback:
    id: str
    message_id: str
    rating: str  # "thumbs_up" or "thumbs_down"
    quality_score: Optional[float] = None
    created_at: Optional[str] = None


@dataclass
class SimilarPattern:
    content: str
    rating: str
    similarity: float
    pattern_type: str


class FeedbackDB:
    """MogDB database for feedback with vector search."""

    def __init__(self, db_path: str = "data/feedback.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize MogDB collections."""
        with self._lock:
            self._db = MogDB(self.db_path)
            self._conversations = self._db.collection("conversations")
            self._messages = self._db.collection("messages")
            self._feedback = self._db.collection("feedback")
            self._meta_weights = self._db.collection("user_meta_weights")

    def _embedding_to_list(self, embedding: np.ndarray) -> List[float]:
        """Convert a numpy embedding to a JSON-serialisable float list."""
        return [float(x) for x in embedding.astype(np.float32).ravel()]

    def _list_to_embedding(self, values: List[float]) -> np.ndarray:
        """Convert a stored float list back to a float32 numpy array."""
        return np.array(values, dtype=np.float32)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def _strip_meta(doc: Dict[str, Any]) -> Dict[str, Any]:
        """Return *doc* without MogDB-internal fields (``_id``/``_created``/``_updated``)."""
        return {k: v for k, v in doc.items() if k not in ("_id", "_created", "_updated")}

    # ============ Conversations ============

    def create_conversation(self, user_id: str = "default", title: str = "New Chat") -> str:
        """Create a new conversation."""
        conv_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            self._conversations.insert_one(
                {
                    "_id": conv_id,
                    "id": conv_id,
                    "user_id": user_id,
                    "title": title,
                    "created_at": now,
                    "updated_at": now,
                }
            )

        return conv_id

    def get_conversation(self, conv_id: str) -> Optional[Dict]:
        """Get conversation by ID."""
        doc = self._conversations.find_one({"_id": conv_id})
        if not doc:
            return None
        return {
            "id": doc["_id"],
            "user_id": doc["user_id"],
            "title": doc["title"],
            "created_at": doc["created_at"],
            "updated_at": doc["updated_at"],
        }

    def list_conversations(self, user_id: str = "default", limit: int = 50) -> List[Dict]:
        """List conversations for a user."""
        docs = self._conversations.find(
            {"user_id": user_id},
            sort=[("updated_at", -1)],
            limit=limit,
        )
        return [
            {
                "id": d["_id"],
                "user_id": d["user_id"],
                "title": d["title"],
                "created_at": d["created_at"],
                "updated_at": d["updated_at"],
            }
            for d in docs
        ]

    # ============ Messages ============

    def add_message(
        self, conversation_id: str, role: str, content: str, embedding: Optional[np.ndarray] = None
    ) -> str:
        """Add a message to a conversation."""
        msg_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        stored_embedding = self._embedding_to_list(embedding) if embedding is not None else None

        with self._lock:
            self._messages.insert_one(
                {
                    "_id": msg_id,
                    "id": msg_id,
                    "conversation_id": conversation_id,
                    "role": role,
                    "content": content,
                    "embedding": stored_embedding,
                    "created_at": now,
                }
            )
            self._conversations.update_one({"_id": conversation_id}, {"$set": {"updated_at": now}})

        return msg_id

    def get_messages(self, conversation_id: str) -> List[Dict]:
        """Get all messages in a conversation."""
        docs = self._messages.find(
            {"conversation_id": conversation_id},
            sort=[("created_at", 1)],
        )
        return [
            {
                "id": d["_id"],
                "conversation_id": d["conversation_id"],
                "role": d["role"],
                "content": d["content"],
                "created_at": d["created_at"],
            }
            for d in docs
        ]

    def get_message_embedding(self, message_id: str) -> Optional[np.ndarray]:
        """Get embedding for a message."""
        doc = self._messages.find_one({"_id": message_id})
        if doc and doc.get("embedding") is not None:
            return self._list_to_embedding(doc["embedding"])
        return None

    # ============ Feedback ============

    def add_feedback(
        self,
        message_id: str,
        rating: str,
        quality_score: Optional[float] = None,
        context_snippet: Optional[str] = None,
    ) -> str:
        """Add feedback for a message."""
        fb_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            self._feedback.insert_one(
                {
                    "_id": fb_id,
                    "id": fb_id,
                    "message_id": message_id,
                    "rating": rating,
                    "quality_score": quality_score,
                    "context_snippet": context_snippet,
                    "created_at": now,
                }
            )

        return fb_id

    def get_feedback(self, message_id: str) -> List[Dict]:
        """Get all feedback for a message."""
        docs = self._feedback.find(
            {"message_id": message_id},
            sort=[("created_at", -1)],
        )
        return [
            {
                "id": d["_id"],
                "message_id": d["message_id"],
                "rating": d["rating"],
                "quality_score": d.get("quality_score"),
                "context_snippet": d.get("context_snippet"),
                "created_at": d["created_at"],
            }
            for d in docs
        ]

    def get_all_feedback(self, rating: Optional[str] = None, limit: int = 1000) -> List[Dict]:
        """Get all feedback, optionally filtered by rating.

        Mimics the previous SQL JOIN (feedback ⋈ messages): each feedback
        document is paired with its message's ``content``/``conversation_id``;
        feedback pointing at a missing message is skipped.
        """
        query: Dict[str, Any] = {}
        if rating:
            query["rating"] = rating

        docs = self._feedback.find(query, sort=[("created_at", -1)], limit=limit)

        results = []
        for doc in docs:
            message = self._messages.find_one({"_id": doc["message_id"]})
            if not message:
                continue
            results.append(
                {
                    "id": doc["_id"],
                    "message_id": doc["message_id"],
                    "rating": doc["rating"],
                    "quality_score": doc.get("quality_score"),
                    "context_snippet": doc.get("context_snippet"),
                    "created_at": doc["created_at"],
                    "content": message["content"],
                    "conversation_id": message["conversation_id"],
                }
            )
        return results

    # ============ Vector Search ============

    def find_similar_messages(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        rating: Optional[str] = None,
        min_similarity: float = 0.0,
    ) -> List[SimilarPattern]:
        """Find similar messages using cosine similarity on embeddings.

        Reproduces the previous SQL path: messages with an embedding, limited
        to 1000 rows, optionally restricted to messages carrying feedback of
        the given ``rating`` (two-step lookup; each message counted once).
        """
        messages = self._messages.find({"embedding": {"$exists": True}}, limit=1000)

        if rating:
            rated_ids = {
                fb["message_id"]
                for fb in self._feedback.find({"rating": rating})
            }
            messages = [m for m in messages if m["_id"] in rated_ids]

        results = []
        for doc in messages:
            embedding = self._list_to_embedding(doc["embedding"])
            similarity = self._cosine_similarity(query_embedding, embedding)

            if similarity >= min_similarity:
                results.append(
                    SimilarPattern(
                        content=doc["content"],
                        rating=rating if rating else "neutral",
                        similarity=similarity,
                        pattern_type="message",
                    )
                )

        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:k]

    def find_similar_by_text(
        self, query: str, k: int = 5, rating: Optional[str] = None
    ) -> List[SimilarPattern]:
        """Find similar messages by text content (simple keyword matching)."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        messages = self._messages.find(limit=500)

        if rating:
            rated_ids = {
                fb["message_id"]
                for fb in self._feedback.find({"rating": rating})
            }
            messages = [m for m in messages if m["_id"] in rated_ids]

        # Score by word overlap
        results = []
        for doc in messages:
            content = doc["content"].lower()
            content_words = set(content.split())

            # Jaccard similarity
            if query_words:
                intersection = len(query_words & content_words)
                union = len(query_words | content_words)
                similarity = intersection / union if union > 0 else 0
            else:
                similarity = 0

            if similarity > 0.1:  # Minimum threshold
                results.append(
                    SimilarPattern(
                        content=doc["content"][:200],  # Truncate
                        rating=rating if rating else "neutral",
                        similarity=similarity,
                        pattern_type="keyword_match",
                    )
                )

        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:k]

    # ============ Statistics ============

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        conv_count = len(self._conversations.find())
        msg_count = len(self._messages.find())

        all_feedback = self._feedback.find()
        fb_count = len(all_feedback)
        thumbs_up = sum(1 for f in all_feedback if f["rating"] == "thumbs_up")
        thumbs_down = sum(1 for f in all_feedback if f["rating"] == "thumbs_down")

        return {
            "conversations": conv_count,
            "messages": msg_count,
            "feedback_total": fb_count,
            "thumbs_up": thumbs_up,
            "thumbs_down": thumbs_down,
            "ratio": thumbs_up / max(thumbs_down, 1),
        }

    # ============ Export ============

    def export_feedback_jsonl(self, filepath: str, rating: Optional[str] = None):
        """Export feedback as JSONL for training."""
        with open(filepath, "w") as f:
            feedback_list = self.get_all_feedback(rating=rating)
            for fb in feedback_list:
                # Get previous message (user) for context
                prev_docs = self._messages.find(
                    {"conversation_id": fb["conversation_id"], "created_at": {"$lt": fb.get("created_at", "")}},
                    sort=[("created_at", -1)],
                    limit=1,
                )
                prev_content = prev_docs[0]["content"] if prev_docs else ""

                record = {
                    "prompt": prev_content,
                    "response": fb["content"],
                    "rating": fb["rating"],
                    "quality_score": fb.get("quality_score"),
                    "message_id": fb["message_id"],
                }
                f.write(json.dumps(record) + "\n")

    def export_dpo_format(self, filepath: str):
        """Export as DPO format: chosen/rejected pairs."""
        thumbs_up = self.get_all_feedback(rating="thumbs_up")
        thumbs_down = self.get_all_feedback(rating="thumbs_down")

        with open(filepath, "w") as f:
            for up_fb in thumbs_up:
                # Find corresponding rejected response in same conversation
                for down_fb in thumbs_down:
                    if up_fb["conversation_id"] == down_fb["conversation_id"]:
                        record = {
                            "chosen": up_fb["content"],
                            "rejected": down_fb["content"],
                            "prompt": up_fb.get("context_snippet", "")[:500],
                        }
                        f.write(json.dumps(record) + "\n")
                        break

    # ============ User Meta Weights ============

    def get_user_meta_weights(self, user_id: str) -> Optional[Dict]:
        """Get meta weights for a specific user."""
        doc = self._meta_weights.find_one({"_id": user_id})
        if not doc:
            return None
        return {
            "user_id": doc["user_id"],
            "temperature_boost": doc.get("temperature_boost", 0.0),
            "repetition_boost": doc.get("repetition_boost", 0.0),
            "top_p_boost": doc.get("top_p_boost", 0.0),
            "top_k_boost": doc.get("top_k_boost", 0),
            "thumbs_up_count": doc.get("thumbs_up_count", 0),
            "thumbs_down_count": doc.get("thumbs_down_count", 0),
            "last_updated": doc.get("last_updated"),
            "created_at": doc.get("created_at"),
        }

    def update_user_meta_weights(
        self,
        user_id: str,
        rating: str,
        temperature_delta: float = 0.01,
        repetition_delta: float = 0.01,
    ) -> Dict:
        """Update meta weights for a user based on feedback."""
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            existing = self._meta_weights.find_one({"_id": user_id})

            if existing:
                temp_boost = existing.get("temperature_boost", 0.0) + (
                    temperature_delta if rating == "thumbs_up" else -temperature_delta
                )
                rep_boost = existing.get("repetition_boost", 0.0) + (
                    -repetition_delta if rating == "thumbs_up" else repetition_delta
                )
                up_count = existing.get("thumbs_up_count", 0) + (1 if rating == "thumbs_up" else 0)
                down_count = existing.get("thumbs_down_count", 0) + (
                    1 if rating == "thumbs_down" else 0
                )

                self._meta_weights.update_one(
                    {"_id": user_id},
                    {
                        "$set": {
                            "temperature_boost": temp_boost,
                            "repetition_boost": rep_boost,
                            "thumbs_up_count": up_count,
                            "thumbs_down_count": down_count,
                            "last_updated": now,
                        }
                    },
                )
            else:
                temp_boost = temperature_delta if rating == "thumbs_up" else -temperature_delta
                rep_boost = -repetition_delta if rating == "thumbs_up" else repetition_delta
                up_count = 1 if rating == "thumbs_up" else 0
                down_count = 1 if rating == "thumbs_down" else 0

                self._meta_weights.insert_one(
                    {
                        "_id": user_id,
                        "user_id": user_id,
                        "temperature_boost": temp_boost,
                        "repetition_boost": rep_boost,
                        "top_p_boost": 0,
                        "top_k_boost": 0,
                        "thumbs_up_count": up_count,
                        "thumbs_down_count": down_count,
                        "last_updated": now,
                        "created_at": now,
                    }
                )

        return self.get_user_meta_weights(user_id)

    def get_all_user_meta_weights(self) -> List[Dict]:
        """Get meta weights for all users."""
        docs = self._meta_weights.find(sort=[("last_updated", -1)])
        return [self._strip_meta(d) for d in docs]


# Global instance
_feedback_db: Optional[FeedbackDB] = None


def get_feedback_db(db_path: str = "data/feedback.db") -> FeedbackDB:
    """Get or create the global feedback database instance."""
    global _feedback_db
    if _feedback_db is None:
        _feedback_db = FeedbackDB(db_path)
    return _feedback_db
