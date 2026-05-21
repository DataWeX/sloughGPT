"""
Feedback database with vector search for meta-weight learning.

Stores conversations, messages, feedback, and uses embeddings
to retrieve similar good responses for biasing generation.
"""

import sqlite3
import pickle
import json
import uuid
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np


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
    """SQLite database for feedback with vector search."""

    def __init__(self, db_path: str = "data/feedback.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)

            # Messages table with embeddings stored as blob
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    role TEXT,
                    content TEXT,
                    embedding BLOB,
                    created_at TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)

            # Feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    message_id TEXT,
                    rating TEXT,
                    quality_score REAL,
                    context_snippet TEXT,
                    created_at TEXT,
                    FOREIGN KEY (message_id) REFERENCES messages(id)
                )
            """)

            # Patterns table for extracted patterns
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id TEXT PRIMARY KEY,
                    message_id TEXT,
                    pattern_type TEXT,
                    pattern_text TEXT,
                    embedding BLOB,
                    quality_score REAL,
                    created_at TEXT,
                    FOREIGN KEY (message_id) REFERENCES messages(id)
                )
            """)

            # User meta-weights table for per-user preferences
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_meta_weights (
                    id TEXT PRIMARY KEY,
                    user_id TEXT UNIQUE,
                    temperature_boost REAL DEFAULT 0.0,
                    repetition_boost REAL DEFAULT 0.0,
                    top_p_boost REAL DEFAULT 0.0,
                    top_k_boost INTEGER DEFAULT 0,
                    thumbs_up_count INTEGER DEFAULT 0,
                    thumbs_down_count INTEGER DEFAULT 0,
                    last_updated TEXT,
                    created_at TEXT
                )
            """)

            # Create indexes for faster queries
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_message ON feedback(message_id)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type)")

            conn.commit()
            conn.close()

    def _embeddings_to_blob(self, embedding: np.ndarray) -> bytes:
        """Convert numpy array to blob for storage."""
        return pickle.dumps(embedding.astype(np.float32))

    def _blob_to_embeddings(self, blob: bytes) -> np.ndarray:
        """Convert blob back to numpy array."""
        return pickle.loads(blob)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # ============ Conversations ============

    def create_conversation(self, user_id: str = "default", title: str = "New Chat") -> str:
        """Create a new conversation."""
        conv_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (conv_id, user_id, title, now, now),
            )
            conn.commit()
            conn.close()

        return conv_id

    def get_conversation(self, conv_id: str) -> Optional[Dict]:
        """Get conversation by ID."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,))
            row = cursor.fetchone()
            conn.close()

        if row:
            return {
                "id": row[0],
                "user_id": row[1],
                "title": row[2],
                "created_at": row[3],
                "updated_at": row[4],
            }
        return None

    def list_conversations(self, user_id: str = "default", limit: int = 50) -> List[Dict]:
        """List conversations for a user."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
                (user_id, limit),
            )
            rows = cursor.fetchall()
            conn.close()

        return [
            {"id": r[0], "user_id": r[1], "title": r[2], "created_at": r[3], "updated_at": r[4]}
            for r in rows
        ]

    # ============ Messages ============

    def add_message(
        self, conversation_id: str, role: str, content: str, embedding: Optional[np.ndarray] = None
    ) -> str:
        """Add a message to a conversation."""
        msg_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        embedding_blob = self._embeddings_to_blob(embedding) if embedding is not None else None

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO messages (id, conversation_id, role, content, embedding, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (msg_id, conversation_id, role, content, embedding_blob, now),
            )

            # Update conversation timestamp
            cursor.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
            )

            conn.commit()
            conn.close()

        return msg_id

    def get_messages(self, conversation_id: str) -> List[Dict]:
        """Get all messages in a conversation."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, conversation_id, role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at",
                (conversation_id,),
            )
            rows = cursor.fetchall()
            conn.close()

        return [
            {"id": r[0], "conversation_id": r[1], "role": r[2], "content": r[3], "created_at": r[4]}
            for r in rows
        ]

    def get_message_embedding(self, message_id: str) -> Optional[np.ndarray]:
        """Get embedding for a message."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT embedding FROM messages WHERE id = ?", (message_id,))
            row = cursor.fetchone()
            conn.close()

        if row and row[0]:
            return self._blob_to_embeddings(row[0])
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
        now = datetime.utcnow().isoformat()

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO feedback (id, message_id, rating, quality_score, context_snippet, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (fb_id, message_id, rating, quality_score, context_snippet, now),
            )
            conn.commit()
            conn.close()

        return fb_id

    def get_feedback(self, message_id: str) -> List[Dict]:
        """Get all feedback for a message."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM feedback WHERE message_id = ? ORDER BY created_at DESC",
                (message_id,),
            )
            rows = cursor.fetchall()
            conn.close()

        return [
            {
                "id": r[0],
                "message_id": r[1],
                "rating": r[2],
                "quality_score": r[3],
                "context_snippet": r[4],
                "created_at": r[5],
            }
            for r in rows
        ]

    def get_all_feedback(self, rating: Optional[str] = None, limit: int = 1000) -> List[Dict]:
        """Get all feedback, optionally filtered by rating."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if rating:
                cursor.execute(
                    "SELECT f.*, m.content, m.conversation_id FROM feedback f JOIN messages m ON f.message_id = m.id WHERE f.rating = ? ORDER BY f.created_at DESC LIMIT ?",
                    (rating, limit),
                )
            else:
                cursor.execute(
                    "SELECT f.*, m.content, m.conversation_id FROM feedback f JOIN messages m ON f.message_id = m.id ORDER BY f.created_at DESC LIMIT ?",
                    (limit,),
                )

            rows = cursor.fetchall()
            conn.close()

        return [
            {
                "id": r[0],
                "message_id": r[1],
                "rating": r[2],
                "quality_score": r[3],
                "context_snippet": r[4],
                "created_at": r[5],
                "content": r[6],
                "conversation_id": r[7],
            }
            for r in rows
        ]

    # ============ Vector Search ============

    def find_similar_messages(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        rating: Optional[str] = None,
        min_similarity: float = 0.0,
    ) -> List[SimilarPattern]:
        """Find similar messages using cosine similarity on embeddings."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all messages with embeddings
            if rating:
                cursor.execute(
                    """SELECT m.id, m.content, m.embedding, f.rating 
                       FROM messages m 
                       JOIN feedback f ON m.id = f.message_id 
                       WHERE m.embedding IS NOT NULL AND f.rating = ?
                       LIMIT 1000""",
                    (rating,),
                )
            else:
                cursor.execute(
                    "SELECT id, content, embedding, role FROM messages WHERE embedding IS NOT NULL LIMIT 1000"
                )

            rows = cursor.fetchall()
            conn.close()

        # Calculate similarities
        results = []
        for row in rows:
            if len(row) >= 3 and row[2]:
                embedding = self._blob_to_embeddings(row[2])
                similarity = self._cosine_similarity(query_embedding, embedding)

                if similarity >= min_similarity:
                    results.append(
                        SimilarPattern(
                            content=row[1],
                            rating=row[3] if len(row) > 3 else "neutral",
                            similarity=similarity,
                            pattern_type="message",
                        )
                    )

        # Sort by similarity and return top k
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:k]

    def find_similar_by_text(
        self, query: str, k: int = 5, rating: Optional[str] = None
    ) -> List[SimilarPattern]:
        """Find similar messages by text content (simple keyword matching)."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if rating:
                cursor.execute(
                    """SELECT m.id, m.content, f.rating 
                       FROM messages m 
                       JOIN feedback f ON m.id = f.message_id 
                       WHERE f.rating = ?
                       LIMIT 500""",
                    (rating,),
                )
            else:
                cursor.execute("SELECT id, content, role FROM messages LIMIT 500")

            rows = cursor.fetchall()
            conn.close()

        # Score by word overlap
        results = []
        for row in rows:
            content = row[1].lower()
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
                        content=row[1][:200],  # Truncate
                        rating=row[2] if len(row) > 2 else "neutral",
                        similarity=similarity,
                        pattern_type="keyword_match",
                    )
                )

        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:k]

    # ============ Statistics ============

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM conversations")
            conv_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM messages")
            msg_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM feedback")
            fb_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM feedback WHERE rating = 'thumbs_up'")
            thumbs_up = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM feedback WHERE rating = 'thumbs_down'")
            thumbs_down = cursor.fetchone()[0]

            conn.close()

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
                with self._lock:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute(
                        """SELECT content FROM messages WHERE conversation_id = ? AND created_at < ? ORDER BY created_at DESC LIMIT 1""",
                        (fb["conversation_id"], fb.get("created_at", "")),
                    )
                    prev_row = cursor.fetchone()
                    conn.close()

                record = {
                    "prompt": prev_row[0] if prev_row else "",
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
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_meta_weights WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            conn.close()

        if row:
            return {
                "user_id": row[1],
                "temperature_boost": row[2],
                "repetition_boost": row[3],
                "top_p_boost": row[4],
                "top_k_boost": row[5],
                "thumbs_up_count": row[6],
                "thumbs_down_count": row[7],
                "last_updated": row[8],
                "created_at": row[9],
            }
        return None

    def update_user_meta_weights(
        self,
        user_id: str,
        rating: str,
        temperature_delta: float = 0.01,
        repetition_delta: float = 0.01,
    ) -> Dict:
        """Update meta weights for a user based on feedback."""
        now = datetime.utcnow().isoformat()

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if user exists
            cursor.execute("SELECT * FROM user_meta_weights WHERE user_id = ?", (user_id,))
            existing = cursor.fetchone()

            if existing:
                # Update existing
                temp_boost = existing[2] + (
                    temperature_delta if rating == "thumbs_up" else -temperature_delta
                )
                rep_boost = existing[3] + (
                    -repetition_delta if rating == "thumbs_up" else repetition_delta
                )
                up_count = existing[6] + (1 if rating == "thumbs_up" else 0)
                down_count = existing[7] + (1 if rating == "thumbs_down" else 0)

                cursor.execute(
                    """
                    UPDATE user_meta_weights 
                    SET temperature_boost = ?, repetition_boost = ?,
                        thumbs_up_count = ?, thumbs_down_count = ?, last_updated = ?
                    WHERE user_id = ?
                """,
                    (temp_boost, rep_boost, up_count, down_count, now, user_id),
                )
            else:
                # Create new
                temp_boost = temperature_delta if rating == "thumbs_up" else -temperature_delta
                rep_boost = -repetition_delta if rating == "thumbs_up" else repetition_delta
                up_count = 1 if rating == "thumbs_up" else 0
                down_count = 1 if rating == "thumbs_down" else 0

                cursor.execute(
                    """
                    INSERT INTO user_meta_weights 
                    (id, user_id, temperature_boost, repetition_boost, top_p_boost, top_k_boost,
                     thumbs_up_count, thumbs_down_count, last_updated, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        str(uuid.uuid4()),
                        user_id,
                        temp_boost,
                        rep_boost,
                        0,
                        0,
                        up_count,
                        down_count,
                        now,
                        now,
                    ),
                )

            conn.commit()
            conn.close()

        return self.get_user_meta_weights(user_id)

    def get_all_user_meta_weights(self) -> List[Dict]:
        """Get meta weights for all users."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_meta_weights ORDER BY last_updated DESC")
            rows = cursor.fetchall()
            conn.close()

        return [
            {
                "user_id": r[1],
                "temperature_boost": r[2],
                "repetition_boost": r[3],
                "top_p_boost": r[4],
                "top_k_boost": r[5],
                "thumbs_up_count": r[6],
                "thumbs_down_count": r[7],
                "last_updated": r[8],
                "created_at": r[9],
            }
            for r in rows
        ]


# Global instance
_feedback_db: Optional[FeedbackDB] = None


def get_feedback_db(db_path: str = "data/feedback.db") -> FeedbackDB:
    """Get or create the global feedback database instance."""
    global _feedback_db
    if _feedback_db is None:
        _feedback_db = FeedbackDB(db_path)
    return _feedback_db
