"""Model Health Repository — typed repository for model health snapshots.

Replaces the ad-hoc JSON file I/O in model_health.py with a structured repository.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

from domains.infrastructure.repository import FileRepository, JsonSerializer

logger = logging.getLogger("slo.model_health_repo")


@dataclass
class HealthSnapshot:
    """A single model health measurement."""
    timestamp: float
    perplexity: float
    loss: float
    num_sentences: int
    model_name: str = ""
    quality_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HealthSnapshot:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ModelHealthRepository:
    """Repository for model health snapshots with JSONL persistence.

    Replaces the ad-hoc db_path.read_text() / write_text() usage
    in model_health.py with a structured repository.
    """

    def __init__(self, data_dir: str | Path, *, max_history: int = 200):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._max_history = max_history

        self._repo = FileRepository[dict](
            self._data_dir / "snapshots",
            serializer=dict,
        )
        self._repo.enable_cache(30.0)

        # Also keep a legacy single-file for backward compatibility
        self._legacy_path = self._data_dir / "model_health.json"
        self._migrate_legacy()

    def _migrate_legacy(self) -> None:
        """Migrate legacy single-file format to repository format."""
        if not self._legacy_path.exists():
            return
        try:
            data = json.loads(self._legacy_path.read_text())
            if not isinstance(data, list):
                return
            for i, entry in enumerate(data[-self._max_history:]):
                snapshot = HealthSnapshot.from_dict(entry)
                snapshot_id = f"snapshot_{snapshot.timestamp:.0f}_{i}"
                self._repo.save(snapshot_id, snapshot.to_dict())
            logger.info("Migrated %d legacy health snapshots", len(data))
            # Rename legacy file as backup
            backup = self._legacy_path.with_suffix(".json.bak")
            self._legacy_path.rename(backup)
        except Exception as e:
            logger.warning("Failed to migrate legacy health data: %s", e)

    def add_snapshot(self, snapshot: HealthSnapshot) -> bool:
        """Add a health snapshot to the repository."""
        snapshot_id = f"snapshot_{snapshot.timestamp:.0f}"
        return self._repo.save(snapshot_id, snapshot.to_dict())

    def get_latest(self) -> Optional[HealthSnapshot]:
        """Get the most recent health snapshot."""
        all_snapshots = self.list_snapshots()
        if not all_snapshots:
            return None
        return max(all_snapshots, key=lambda s: s.timestamp)

    def list_snapshots(self, limit: int | None = None) -> list[HealthSnapshot]:
        """List health snapshots, optionally limited to most recent N."""
        all_data = self._repo.list()
        snapshots = [HealthSnapshot.from_dict(d) for d in all_data]
        snapshots.sort(key=lambda s: s.timestamp, reverse=True)
        if limit:
            return snapshots[:limit]
        return snapshots

    def get_trend(self, hours: float = 24.0) -> list[HealthSnapshot]:
        """Get health snapshots from the last N hours."""
        cutoff = time.time() - (hours * 3600)
        return [s for s in self.list_snapshots() if s.timestamp >= cutoff]

    def detect_drift(self, threshold: float = 0.15) -> Optional[dict[str, Any]]:
        """Detect significant perplexity drift compared to recent average.

        Returns drift info dict if drift detected, None otherwise.
        """
        recent = self.list_snapshots(limit=10)
        if len(recent) < 2:
            return None

        latest = recent[0]
        baseline = recent[1:]
        avg_baseline = sum(s.perplexity for s in baseline) / len(baseline)

        if avg_baseline == 0:
            return None

        drift_pct = (latest.perplexity - avg_baseline) / avg_baseline
        if abs(drift_pct) >= threshold:
            return {
                "drift_detected": True,
                "latest_ppl": latest.perplexity,
                "baseline_ppl": avg_baseline,
                "drift_percent": drift_pct,
                "timestamp": latest.timestamp,
            }
        return None

    def get_stats(self) -> dict[str, Any]:
        """Get aggregate statistics over all snapshots."""
        snapshots = self.list_snapshots()
        if not snapshots:
            return {"count": 0}
        ppls = [s.perplexity for s in snapshots]
        losses = [s.loss for s in snapshots]
        return {
            "count": len(snapshots),
            "avg_perplexity": sum(ppls) / len(ppls),
            "min_perplexity": min(ppls),
            "max_perplexity": max(ppls),
            "avg_loss": sum(losses) / len(losses),
            "oldest": min(s.timestamp for s in snapshots),
            "newest": max(s.timestamp for s in snapshots),
        }

    def clear(self) -> int:
        """Clear all snapshots. Returns count of deleted items."""
        snapshots = self.list_snapshots()
        count = len(snapshots)
        for s in snapshots:
            snapshot_id = f"snapshot_{s.timestamp:.0f}"
            self._repo.delete(snapshot_id)
        return count
