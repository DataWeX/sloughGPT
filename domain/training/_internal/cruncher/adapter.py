"""Cruncher adapter — sealed snapshot → trainer data_path.

The bridge between the cruncher's versioned snapshots and the one training
loop.  Resolution reuses ``resolve_training_data_path`` (SloughGPT Standard
v1 manifest), so the trainer consumes snapshots exactly like any other
manifest dataset: point ``data_path`` at the manifest and train.  No trainer
changes, no second loader.
"""

from __future__ import annotations

import logging
from pathlib import Path

from domain.training._internal.dataset_manifest import (
    ManifestError,
    load_manifest,
    resolve_training_data_path,
)

from .snapshot import SnapshotRef, SnapshotStore

logger = logging.getLogger("slo.training.cruncher")


def resolve_snapshot(ref: SnapshotRef) -> tuple[Path, dict]:
    """Resolve a sealed snapshot to ``(train_text_path, manifest_dict)``.

    Raises:
        ManifestError: manifest missing/invalid or train split unresolvable —
            i.e. the snapshot is not trainable, fail loud before a run starts.
    """
    text_path, manifest = resolve_training_data_path(ref.manifest_path)
    return text_path, manifest


def resolve_latest(dataset_id: str, store: SnapshotStore | None = None) -> tuple[Path, dict]:
    """Resolve the newest sealed snapshot of *dataset_id*.

    Raises:
        ManifestError: no sealed snapshot exists for the dataset.
    """
    store = store or SnapshotStore()
    ref = store.latest(dataset_id)
    if ref is None:
        raise ManifestError(f"no sealed snapshot for dataset {dataset_id!r} under {store.root}")
    return resolve_snapshot(ref)


def trainer_data_path(ref: SnapshotRef) -> str:
    """The ``data_path`` string to hand the trainer for *ref*.

    Trainer call sites accept a filesystem path to the manifest; this keeps
    that contract in one place.
    """
    if not ref.manifest_path.is_file():
        raise ManifestError(f"snapshot manifest missing: {ref.manifest_path}")
    # Validate early so a bad snapshot fails at adapter time, not mid-run.
    load_manifest(ref.manifest_path)
    return str(ref.manifest_path)
