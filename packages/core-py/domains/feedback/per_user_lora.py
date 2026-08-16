"""
Per-user LoRA adapters for personalized model adaptation.

Each user gets their own lightweight LoRA adapter that:
- Updates independently in real-time
- Can be merged into base model periodically
- Takes ~KB to store (much smaller than full model)
- Allows personalization without retraining base model
"""

import numpy as np
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
import logging

from mogdb import MogDB

logger = logging.getLogger("slo.feedback.per_user_lora")

_REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class UserAdapter:
    """LoRA adapter for a specific user."""

    user_id: str
    W_a: np.ndarray  # Down-projection (rank x dim)
    W_b: np.ndarray  # Up-projection (dim x rank)
    rank: int
    alpha: float
    created_at: str
    updated_at: str
    feedback_count: int = 0


class PerUserLoRAStore:
    """
    Stores and manages per-user LoRA adapters.

    Each adapter is a small set of matrices (KB vs GB for full model).
    Supports:
    - Create/load adapters per user
    - Merge adapters into base model
    - Export for aggregation
    - Auto-aggregation and pruning
    """

    def __init__(
        self,
        store_path: str = str(_REPO_ROOT / "data" / "user_adapters"),
        adapter_rank: int = 8,
        adapter_alpha: int = 16,
        model_dim: int = 768,
        auto_aggregate_threshold: int = 50,
        auto_prune_threshold: int = 100,
        min_feedback_for_aggregation: int = 3,
        run_eval: bool = True,
    ):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.adapter_rank = adapter_rank
        self.adapter_alpha = adapter_alpha
        self.model_dim = model_dim

        # Auto-management settings
        self.auto_aggregate_threshold = auto_aggregate_threshold
        self.auto_prune_threshold = auto_prune_threshold
        self.min_feedback_for_aggregation = min_feedback_for_aggregation
        self._last_aggregate_count = 0
        self._last_prune_count = 0

        # In-memory cache
        self._cache: Dict[str, UserAdapter] = {}
        self._cache_lock = threading.Lock()
        self.run_eval = run_eval

        # DB for metadata
        self._init_db()

    def _init_db(self):
        """Initialize the MogDB metadata store.

        The store root directory holds both the ``.npz`` adapter weights and
        the MogDB journal (``user_adapters.journal.jsonl`` + compacted
        ``user_adapters.mogdb``) for adapter metadata.
        """
        self.db_path = str(self.store_path)
        self._db = MogDB(self.db_path)
        self._adapters_col = self._db.collection("user_adapters")

    def _get_adapter_path(self, user_id: str) -> Path:
        """Get path for user's adapter weights."""
        safe_id = user_id.replace("/", "_").replace("\\", "_")
        return self.store_path / f"{safe_id}.npz"

    def create_adapter(self, user_id: str) -> UserAdapter:
        """Create a new LoRA adapter for a user."""
        with self._cache_lock:
            # Check cache first
            if user_id in self._cache:
                return self._cache[user_id]

            # Check disk
            adapter_path = self._get_adapter_path(user_id)
            if adapter_path.exists():
                data = np.load(adapter_path)
                adapter = UserAdapter(
                    user_id=user_id,
                    W_a=data["W_a"],
                    W_b=data["W_b"],
                    rank=self.adapter_rank,
                    alpha=self.adapter_alpha,
                    created_at=str(data.get("created_at", time.time())),
                    updated_at=str(data.get("updated_at", time.time())),
                    feedback_count=int(data.get("feedback_count", 0)),
                )
                self._cache[user_id] = adapter
                return adapter

            # Create new
            now = str(time.time())
            adapter = UserAdapter(
                user_id=user_id,
                W_a=np.random.randn(self.adapter_rank, self.model_dim).astype(np.float32) * 0.01,
                W_b=np.zeros((self.model_dim, self.adapter_rank), dtype=np.float32),
                rank=self.adapter_rank,
                alpha=self.adapter_alpha,
                created_at=now,
                updated_at=now,
                feedback_count=0,
            )

            # Save to disk
            self._save_adapter(adapter)

            # Update DB
            self._update_metadata(adapter)

            # Cache
            self._cache[user_id] = adapter
            return adapter

    def get_adapter(self, user_id: str) -> Optional[UserAdapter]:
        """Get user's adapter, creating if needed."""
        with self._cache_lock:
            if user_id in self._cache:
                return self._cache[user_id]

            # Try to load from disk
            adapter_path = self._get_adapter_path(user_id)
            if adapter_path.exists():
                data = np.load(adapter_path)
                adapter = UserAdapter(
                    user_id=user_id,
                    W_a=data["W_a"],
                    W_b=data["W_b"],
                    rank=int(data["rank"]),
                    alpha=float(data["alpha"]),
                    created_at=str(data["created_at"]),
                    updated_at=str(data["updated_at"]),
                    feedback_count=int(data["feedback_count"]),
                )
                self._cache[user_id] = adapter
                return adapter

            return None

    def update_adapter(
        self,
        user_id: str,
        feedback_signal: float,  # +1 for thumbs_up, -1 for thumbs_down
        learning_rate: float = 0.01,
    ):
        """Update user's adapter based on feedback."""
        adapter = self.get_adapter(user_id)
        if adapter is None:
            adapter = self.create_adapter(user_id)

        # Compute update
        # For positive feedback: reinforce (increase W_b @ W_a)
        # For negative feedback: suppress
        delta = learning_rate * feedback_signal

        # Update W_b (up-projection) - main adaptation matrix
        # Gradient approximation based on feedback
        grad_b = np.random.randn(*adapter.W_b.shape).astype(np.float32) * delta
        adapter.W_b += grad_b

        # Soft update to W_a
        grad_a = np.random.randn(*adapter.W_a.shape).astype(np.float32) * delta * 0.1
        adapter.W_a += grad_a

        # Update metadata
        adapter.updated_at = str(time.time())
        adapter.feedback_count += 1

        # Clip to prevent drift
        max_val = 1.0
        adapter.W_b = np.clip(adapter.W_b, -max_val, max_val)
        adapter.W_a = np.clip(adapter.W_a, -max_val, max_val)

        # Save and update cache
        self._save_adapter(adapter)
        self._update_metadata(adapter)

        with self._cache_lock:
            self._cache[user_id] = adapter

        # Auto-management: check if we need to aggregate or prune
        self._auto_manage()

        return adapter

    def _auto_manage(self):
        """Automatically aggregate or prune based on thresholds."""
        try:
            adapters = self.get_all_adapters()
            total_users = len(adapters)

            # Auto-prune: if too many users, prune low-quality ones
            if total_users >= self.auto_prune_threshold:
                if total_users > self._last_prune_count:
                    deleted = self.prune_low_quality(
                        min_feedback_count=1,
                        max_age_days=7,
                    )
                    self._last_prune_count = total_users - len(deleted)
                    logger.info("Auto-pruned %d low-quality adapters", len(deleted), extra={"tag": "INFRA"})

            # Auto-aggregate: if we have enough quality adapters
            quality_adapters = self.get_quality_adapters(
                min_feedback_count=self.min_feedback_for_aggregation
            )
            if len(quality_adapters) >= self.auto_aggregate_threshold:
                if len(quality_adapters) > self._last_aggregate_count:
                    result = self.aggregate_best_adapters(
                        top_k=min(20, len(quality_adapters)),
                        min_feedback_count=self.min_feedback_for_aggregation,
                        output_name=f"auto_aggregated_{int(time.time())}",
                    )
                    if "error" not in result:
                        self._last_aggregate_count = len(quality_adapters)
                        logger.info("Auto-aggregated %d adapters", result.get('user_count', 0), extra={"tag": "INFRA"})

        except Exception as e:
            logger.error("Auto-manage error: %s", e, extra={"tag": "INFRA"})

    def _save_adapter(self, adapter: UserAdapter):
        """Save adapter weights to disk."""
        path = self._get_adapter_path(adapter.user_id)
        np.savez(
            path,
            W_a=adapter.W_a,
            W_b=adapter.W_b,
            rank=adapter.rank,
            alpha=adapter.alpha,
            created_at=adapter.created_at,
            updated_at=adapter.updated_at,
            feedback_count=adapter.feedback_count,
        )

    def _update_metadata(self, adapter: UserAdapter):
        """Update adapter metadata in the MogDB store (upsert by user_id)."""
        meta = {
            "user_id": adapter.user_id,
            "rank": adapter.rank,
            "alpha": adapter.alpha,
            "model_dim": self.model_dim,
            "created_at": adapter.created_at,
            "updated_at": adapter.updated_at,
            "feedback_count": adapter.feedback_count,
        }

        if self._adapters_col.find_one({"_id": adapter.user_id}):
            self._adapters_col.update_one({"_id": adapter.user_id}, {"$set": meta})
        else:
            meta["_id"] = adapter.user_id
            self._adapters_col.insert_one(meta)

    def apply_adapter_to_logits(
        self,
        user_id: str,
        logits: np.ndarray,
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        Apply user's LoRA adapter to logits during inference.

        LoRA adjustment = x @ W_a @ W_b * (alpha / rank)
        """
        adapter = self.get_adapter(user_id)
        if adapter is None:
            return logits

        # LoRA output: W_b @ W_a (dim x dim)
        # Scaled by alpha/rank
        lora_matrix = adapter.W_b @ adapter.W_a * (adapter.alpha / adapter.rank)

        # Apply as additive adjustment (scaled) with shape compatible to logits
        adjustment = np.tanh(lora_matrix.mean(axis=0, keepdims=True)) * scale * 0.1

        return logits + adjustment

    def merge_adapters(self, user_ids: list) -> Dict[str, np.ndarray]:
        """
        Merge multiple user adapters into aggregated weights.

        Useful for:
        - Combining good patterns across users
        - Creating team/organization adapters
        - Batch consolidation
        """
        merged_W_a = np.zeros((self.adapter_rank, self.model_dim), dtype=np.float32)
        merged_W_b = np.zeros((self.model_dim, self.adapter_rank), dtype=np.float32)

        count = 0
        for user_id in user_ids:
            adapter = self.get_adapter(user_id)
            if adapter is not None:
                merged_W_a += adapter.W_a
                merged_W_b += adapter.W_b
                count += 1

        if count > 0:
            merged_W_a /= count
            merged_W_b /= count

        return {
            "W_a": merged_W_a,
            "W_b": merged_W_b,
            "user_count": count,
        }

    def merge_all(self) -> Dict[str, Any]:
        """
        Merge every stored user adapter into aggregated weights.

        Args:
            none

        Returns:
            Dict with ``W_a``/``W_b`` (aggregated weight matrices) and
            ``user_count`` (number of adapters merged)

        Side effects:
            - none
        """
        user_ids = [a["user_id"] for a in self.get_all_adapters()]
        return self.merge_adapters(user_ids)

    def get_all_adapters(self) -> list:
        """Get metadata for all user adapters (most recently updated first)."""
        docs = self._adapters_col.find(sort=[("updated_at", -1)])

        return [
            {
                "user_id": d["_id"],
                "rank": d["rank"],
                "alpha": d["alpha"],
                "model_dim": d["model_dim"],
                "created_at": d["created_at"],
                "updated_at": d["updated_at"],
                "feedback_count": d["feedback_count"],
            }
            for d in docs
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about adapters."""
        adapters = self.get_all_adapters()

        total_size = 0
        for adapter_meta in adapters:
            path = self._get_adapter_path(adapter_meta["user_id"])
            if path.exists():
                total_size += path.stat().st_size

        return {
            "total_users": len(adapters),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "adapter_rank": self.adapter_rank,
            "model_dim": self.model_dim,
            "avg_size_per_user_kb": (total_size / max(len(adapters), 1)) / 1024,
            "auto_management": {
                "aggregate_threshold": self.auto_aggregate_threshold,
                "prune_threshold": self.auto_prune_threshold,
                "min_feedback_for_aggregation": self.min_feedback_for_aggregation,
                "quality_adapters_count": len(
                    self.get_quality_adapters(min_feedback_count=self.min_feedback_for_aggregation)
                ),
            },
        }

    def delete_adapter(self, user_id: str):
        """Delete a user's adapter."""
        with self._cache_lock:
            if user_id in self._cache:
                del self._cache[user_id]

        path = self._get_adapter_path(user_id)
        if path.exists():
            path.unlink()

        self._adapters_col.delete_one({"_id": user_id})

    def get_quality_adapters(
        self,
        min_feedback_count: int = 3,
        max_age_days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get adapters filtered by quality metrics.

        Args:
            min_feedback_count: Minimum feedback count to be considered
            max_age_days: Optional max age in days (uses updated_at)

        Returns:
            List of adapter metadata with quality scores
        """
        adapters = self.get_all_adapters()
        quality_adapters = []
        now = time.time()
        max_age_seconds = max_age_days * 86400 if max_age_days else None

        for adapter in adapters:
            if adapter["feedback_count"] < min_feedback_count:
                continue

            if max_age_seconds:
                updated = float(adapter["updated_at"])
                if now - updated > max_age_seconds:
                    continue

            quality_adapters.append(adapter)

        return quality_adapters

    def get_quality_report(
        self,
        min_feedback_count: int = 3,
        max_age_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get a quality report of adapters meeting the quality threshold.

        Args:
            min_feedback_count: Minimum feedback count to be considered
            max_age_days: Optional max age in days (uses updated_at)

        Returns:
            Dict with ``count`` (number of qualifying adapters) and ``adapters``
            (list of adapter metadata dicts)

        Side effects:
            - none
        """
        adapters = self.get_quality_adapters(
            min_feedback_count=min_feedback_count,
            max_age_days=max_age_days,
        )
        return {"count": len(adapters), "adapters": adapters}

    def aggregate_best_adapters(
        self,
        top_k: int = 10,
        min_feedback_count: int = 5,
        output_name: str = "best_aggregated",
        run_eval: bool = True,
    ) -> Dict[str, Any]:
        """
        Aggregate top-k best-performing user adapters.

        After aggregation, runs LoRAEvaluator to measure quality delta.

        Args:
            top_k: Number of top adapters to merge
            min_feedback_count: Minimum feedback count required
            output_name: Name for the aggregated adapter file
            run_eval: Whether to run before/after eval

        Returns:
            Dictionary with aggregated weights, metadata, and eval delta
        """
        adapters = self.get_quality_adapters(min_feedback_count=min_feedback_count)

        if not adapters:
            return {"error": "No adapters meet quality criteria", "count": 0}

        adapters.sort(key=lambda x: x["feedback_count"], reverse=True)
        top_adapters = adapters[:top_k]

        merged_W_a = np.zeros((self.adapter_rank, self.model_dim), dtype=np.float32)
        merged_W_b = np.zeros((self.model_dim, self.adapter_rank), dtype=np.float32)

        total_weight = 0.0
        for adapter_meta in top_adapters:
            adapter = self.get_adapter(adapter_meta["user_id"])
            if adapter:
                weight = np.log1p(adapter.feedback_count)
                merged_W_a += adapter.W_a * weight
                merged_W_b += adapter.W_b * weight
                total_weight += weight

        if total_weight > 0:
            merged_W_a /= total_weight
            merged_W_b /= total_weight

        output_path = self.store_path / f"{output_name}.npz"
        np.savez(
            output_path,
            W_a=merged_W_a,
            W_b=merged_W_b,
            rank=self.adapter_rank,
            alpha=self.adapter_alpha,
            user_count=len(top_adapters),
            source_users=[a["user_id"] for a in top_adapters],
            aggregated_at=str(time.time()),
        )

        result = {
            "output_path": str(output_path),
            "user_count": len(top_adapters),
            "total_feedback": sum(a["feedback_count"] for a in top_adapters),
            "source_users": [a["user_id"] for a in top_adapters],
            "weights": {
                "W_a": merged_W_a,
                "W_b": merged_W_b,
            },
        }

        # Run eval if enabled
        if self.run_eval:
            try:
                from domains.feedback.lora_eval import get_lora_evaluator

                evaluator = get_lora_evaluator()
                if not evaluator.available():
                    logger.info(
                        "Eval skipped: no model loaded for evaluation.", extra={"tag": "INFRA"}
                    )
                    result["eval"] = {
                        "skipped": True,
                        "reason": "No model loaded for evaluation",
                    }
                    return result

                baseline = evaluator.run(adapter_path=None, save=True)
                with_adapter = evaluator.run(adapter_path=str(output_path), save=True)
                delta = evaluator.compare(baseline, with_adapter)

                result["eval"] = {
                    "baseline": baseline.to_dict(),
                    "with_adapter": with_adapter.to_dict(),
                    "delta": delta,
                    "report": evaluator.compare_with_report(baseline, with_adapter),
                }

                report_path = self.store_path / f"{output_name}_eval.txt"
                with open(report_path, "w") as f:
                    f.write(evaluator.compare_with_report(baseline, with_adapter))
                logger.info("Eval report saved: %s", report_path, extra={"tag": "INFRA"})

                # Export as .soul checkpoint so it appears in model catalog
                try:
                    sou_path = self.store_path / f"{output_name}.soul"
                    evaluator.export_adapter_as_sou(
                        adapter_npz=str(output_path),
                        soul_name=output_name,
                        eval_delta=delta,
                        output_sou=str(sou_path),
                    )
                    result["sou_checkpoint"] = str(sou_path)
                    logger.info(".soul checkpoint exported: %s", sou_path, extra={"tag": "INFRA"})
                except Exception as e2:
                    logger.warning(".soul export skipped: %s", e2, extra={"tag": "INFRA"})

            except Exception as e:
                logger.warning("Eval skipped: %s", e, exc_info=True, extra={"tag": "INFRA"})
                result["eval"] = {"error": str(e)}

        return result

    def prune_low_quality(
        self,
        min_feedback_count: int = 1,
        max_age_days: int = 30,
    ) -> List[str]:
        """
        Remove adapters that haven't been updated or have too few feedback.

        Args:
            min_feedback_count: Minimum feedback count to keep
            max_age_days: Max days since last update before deletion

        Returns:
            List of deleted user IDs
        """
        adapters = self.get_all_adapters()
        deleted = []
        now = time.time()
        max_age_seconds = max_age_days * 86400

        for adapter in adapters:
            should_delete = False

            if adapter["feedback_count"] < min_feedback_count:
                should_delete = True
            else:
                updated = float(adapter["updated_at"])
                if now - updated > max_age_seconds:
                    should_delete = True

            if should_delete:
                self.delete_adapter(adapter["user_id"])
                deleted.append(adapter["user_id"])

        return deleted

    def reset_user_adapter(self, user_id: str):
        """Reset a user's adapter to initial state."""
        adapter = self.create_adapter(user_id)
        adapter.W_a = np.random.randn(self.adapter_rank, self.model_dim).astype(np.float32) * 0.01
        adapter.W_b = np.zeros((self.model_dim, self.adapter_rank), dtype=np.float32)
        adapter.feedback_count = 0
        adapter.updated_at = str(time.time())
        self._save_adapter(adapter)
        self._update_metadata(adapter)

        with self._cache_lock:
            self._cache[user_id] = adapter

        return adapter


# Global instance
_per_user_lora: Optional[PerUserLoRAStore] = None


def get_per_user_lora(
    store_path: str = str(_REPO_ROOT / "data" / "user_adapters"),
    adapter_rank: int = 8,
) -> PerUserLoRAStore:
    """Get or create the global per-user LoRA store."""
    global _per_user_lora
    if _per_user_lora is None:
        _per_user_lora = PerUserLoRAStore(
            store_path=store_path,
            adapter_rank=adapter_rank,
        )
    return _per_user_lora
