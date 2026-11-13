"""Durable streaming metrics for long unattended pretraining (goal 18).

``TrainingMonitor`` keeps loss history in RAM only, and ``ExperimentTracker``
needs MLflow/WandB. A months-long run needs a file you can ``tail`` after a
crash: append-only JSONL under the checkpoint directory.

This is a recorder, not a training loop — no forward/loss/backward/optimize.
Wire it from ``on_progress`` (or any progress dict producer) in the long-run
supervisor.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

__all__ = ["LongRunRecorder"]


class LongRunRecorder:
    """Append-only JSONL metrics log for a long training run.

    Each ``record()`` writes one JSON object per line under ``path`` (created
    if missing). Keys are preserved as given; a ``ts`` field is added when the
    caller does not supply one. Safe for concurrent readers (single writer).

    Example:
        rec = LongRunRecorder(ckpt_dir / "metrics.jsonl")

        def on_progress(info: dict) -> None:
            rec.record(**info)

        trainer.train(on_progress=on_progress)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def record(self, **fields: Any) -> dict[str, Any]:
        """Append one metrics line. Returns the dict that was written."""
        row: dict[str, Any] = dict(fields)
        if "ts" not in row:
            row["ts"] = time.time()
        line = json.dumps(row, default=float, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
        self._count += 1
        return row

    @property
    def count(self) -> int:
        """Number of lines written by this process instance."""
        return self._count

    def read_all(self) -> list[dict[str, Any]]:
        """Parse every JSONL row currently on disk (for tests / monitoring)."""
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return rows
