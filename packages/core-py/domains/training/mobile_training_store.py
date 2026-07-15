"""Training data store backed by MogDB.

Provides structured storage for (user, assistant) conversation pairs
collected from the mobile app.  Supports indexed queries by session,
quality, sync status, and timestamp.

Usage::

    from domains.training.mobile_training_store import get_training_store

    store = get_training_store()
    store.add_pair("hello", "hi there", "s1", quality=1)
    pending = store.get_pending_pairs()
    store.mark_synced([pair["_id"] for pair in pending])
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("slo.training.mobile_store")

# Singleton
_store = None


class MobileTrainingStore:
    """MogDB-backed store for mobile training pairs.

    Args:
        db_path: Directory for MogDB journal files.
    """

    def __init__(self, db_path: str):
        from mogdb import MogDB

        self._db = MogDB(db_path)
        self._col = self._db.collection("training_pairs")

    def add_pair(
        self,
        user_msg: str,
        assistant_msg: str,
        session_id: str,
        quality: float = 0.0,
        model: str = "",
    ) -> str:
        """Insert a single training pair.

        Args:
            user_msg: The user's message.
            assistant_msg: The assistant's response.
            session_id: Chat session this pair came from.
            quality: Quality signal (-1, 0, or 1).
            model: Model that generated the response.

        Returns:
            The inserted document's _id.
        """
        doc = {
            "user_msg": user_msg,
            "assistant_msg": assistant_msg,
            "session_id": session_id,
            "quality": quality,
            "model": model,
            "synced": False,
            "used_for_training": False,
            "timestamp": time.time(),
        }
        doc_id = self._col.insert_one(doc)
        logger.debug("added training pair %s (session=%s)", doc_id, session_id)
        return doc_id

    def add_batch(
        self,
        pairs: List[Dict[str, Any]],
    ) -> List[str]:
        """Insert multiple training pairs.

        Args:
            pairs: List of dicts with user_msg, assistant_msg, session_id, etc.

        Returns:
            List of inserted _ids.
        """
        ids = []
        for p in pairs:
            doc_id = self.add_pair(
                user_msg=p.get("user_msg", ""),
                assistant_msg=p.get("assistant_msg", ""),
                session_id=p.get("session_id", ""),
                quality=p.get("quality", 0.0),
                model=p.get("model", ""),
            )
            ids.append(doc_id)
        logger.info("inserted %d training pairs", len(ids),
            extra={"tag": "TRAIN"},)
        return ids

    def get_pair(self, pair_id: str) -> Optional[Dict[str, Any]]:
        """Get a single pair by ID."""
        return self._col.find_one({"_id": pair_id})

    def get_pending_pairs(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all pairs not yet synced to server for training.

        Args:
            limit: Max pairs to return.

        Returns:
            List of un-synced training pair documents.
        """
        return self._col.find(
            {"synced": False},
            sort=[("timestamp", 1)],
            limit=limit,
        )

    def get_pairs_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all pairs from a specific session."""
        return self._col.find(
            {"session_id": session_id},
            sort=[("timestamp", 1)],
        )

    def get_pairs_by_quality(self, min_quality: float = 0.0) -> List[Dict[str, Any]]:
        """Get pairs with quality >= min_quality."""
        return self._col.find(
            {"quality": {"$gte": min_quality}},
            sort=[("quality", -1)],
        )

    def get_training_ready(
        self, min_pairs: int = 10, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get pairs ready for training (not yet used, not synced).

        Args:
            min_pairs: Minimum pairs needed before returning any.
            limit: Max pairs to return.

        Returns:
            List of training-ready documents, or empty if below threshold.
        """
        total_pending = self._col.count({"synced": False})
        if total_pending < min_pairs:
            return []
        return self.get_pending_pairs(limit=limit)

    def mark_synced(self, pair_ids: List[str]) -> int:
        """Mark pairs as synced (sent to server for training).

        Args:
            pair_ids: List of document _ids.

        Returns:
            Number of documents updated.
        """
        count = 0
        for pid in pair_ids:
            count += self._col.update_one(
                {"_id": pid},
                {"$set": {"synced": True}},
            )
        return count

    def mark_used(self, pair_ids: List[str]) -> int:
        """Mark pairs as used for training (consumed by fine-tune).

        Args:
            pair_ids: List of document _ids.

        Returns:
            Number of documents updated.
        """
        count = 0
        for pid in pair_ids:
            count += self._col.update_one(
                {"_id": pid},
                {"$set": {"used_for_training": True}},
            )
        return count

    def update_quality(self, pair_id: str, quality: float) -> bool:
        """Update quality signal on a pair (e.g. thumbs up/down).

        Args:
            pair_id: Document _id.
            quality: New quality value.

        Returns:
            True if updated, False if not found.
        """
        return self._col.update_one(
            {"_id": pair_id},
            {"$set": {"quality": quality}},
        ) > 0

    def delete_pair(self, pair_id: str) -> bool:
        """Delete a single pair."""
        return self._col.delete_one({"_id": pair_id}) > 0

    def delete_synced(self) -> int:
        """Delete all synced pairs (already used for training)."""
        return self._col.delete_many({"synced": True})

    def list_pairs(
        self,
        limit: int = 50,
        offset: int = 0,
        min_quality: Optional[float] = None,
        session_id: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List training pairs with optional filters.

        Args:
            limit: Max pairs to return.
            offset: Skip first N pairs (for pagination).
            min_quality: Filter to quality >= this value.
            session_id: Filter to specific session.
            search: Search in user_msg and assistant_msg content.

        Returns:
            List of training pair documents, newest first.
        """
        query: Dict[str, Any] = {}
        if min_quality is not None:
            query["quality"] = {"$gte": min_quality}
        if session_id:
            query["session_id"] = session_id
        if search:
            query["$or"] = [
                {"user_msg": {"$regex": search, "$options": "i"}},
                {"assistant_msg": {"$regex": search, "$options": "i"}},
            ]
        return self._col.find(
            query,
            sort=[("timestamp", -1)],
            limit=limit,
            skip=offset,
        )

    def count(self, query: Optional[Dict] = None) -> int:
        """Count pairs matching query (or all if None)."""
        return self._col.count(query)

    def stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        total = self.count()
        synced = self.count({"synced": True})
        used = self.count({"used_for_training": True})
        return {
            "total": total,
            "pending": total - synced,
            "synced": synced,
            "used": used,
        }

    def quality_breakdown(self) -> Dict[str, int]:
        """Get count of pairs per quality value.

        Returns:
            Dict mapping quality string ("0", "1", "-1") to count.
        """
        pairs = self._col.find({})
        counts: Dict[str, int] = {}
        for p in pairs:
            q = str(p.get("quality", 0))
            counts[q] = counts.get(q, 0) + 1
        return counts

    def compact(self) -> int:
        """Compact the underlying journal. Returns doc count."""
        return self._col.compact()

    def close(self) -> None:
        """Compact and close the database."""
        self.compact()


def get_training_store(db_path: Optional[str] = None) -> MobileTrainingStore:
    """Get or create the singleton training data store.

    Args:
        db_path: Override path (default: packages/data/mobile_training/).
    """
    global _store
    if _store is None:
        if db_path is None:
            repo_root = Path(__file__).resolve().parents[4]
            db_path = str(repo_root / "packages" / "data" / "mobile_training")
        _store = MobileTrainingStore(db_path)
    return _store
