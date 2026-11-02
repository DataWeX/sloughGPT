"""MogDB journal for multimodal (video-caption) checkpoint metadata.

The ``*_meta.json`` files beside each checkpoint remain the human-readable
artifact and the write-time source of truth; this journal makes checkpoint
listing MogDB-first without ever resurrecting a checkpoint whose files were
deleted behind its back — ``list_journal_checkpoints`` validates every entry
against the filesystem and prunes stale docs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from domain.shared import find_repo_root

logger = logging.getLogger("slo.checkpoint_journal")

_JOURNAL_DB = find_repo_root(Path(__file__).resolve()) / "data" / "multimodal_mogdb"
_COLLECTION = "video_checkpoints"


def _collection():
    """Open the journal collection, or None when MogDB is unavailable."""
    try:
        from mogdb import MogDB

        db = MogDB(str(_JOURNAL_DB))
        return db.collection(_COLLECTION)
    except Exception as e:
        logger.warning("Checkpoint journal unavailable (MogDB failed): %s", e)
        return None


def journal_checkpoint(meta: dict[str, Any], npz_path: Path) -> None:
    """Upsert one checkpoint's metadata into the journal.

    Best-effort: a journal failure never fails the save — the ``*_meta.json``
    file is already written and remains authoritative for the file scan.
    """
    try:
        coll = _collection()
        if coll is None:
            return
        doc = dict(meta)
        doc["path"] = str(npz_path)
        existing = coll.find_one({"name": meta.get("name")})
        if existing is not None:
            coll.find_one_and_replace({"_id": existing["_id"]}, doc)
        else:
            coll.insert_one(doc)
    except Exception as e:
        logger.warning("Failed to journal checkpoint %s: %s", meta.get("name"), e)


def list_journal_checkpoints(ckpt_dir: Path) -> list[dict] | None:
    """Journal-backed checkpoint list for *ckpt_dir*, or None to fall back.

    Returns ``None`` when the journal is empty/unavailable (caller scans
    ``*_meta.json``). Entries whose ``.npz`` no longer exists are dropped and
    their stale docs pruned, so a direct file delete always wins.
    """
    coll = _collection()
    if coll is None:
        return None
    try:
        docs = coll.find(sort=[("epoch", -1), ("step", -1)])
    except Exception as e:
        logger.warning("Checkpoint journal read failed, falling back to files: %s", e)
        return None
    if not docs:
        return None

    checkpoints: list[dict[str, Any]] = []
    for doc in docs:
        npz = Path(doc.get("path", ""))
        if not npz.exists():
            try:
                coll.delete_one({"_id": doc["_id"]})
            except Exception:
                pass
            continue
        checkpoints.append(
            {
                "name": doc.get("name", npz.stem),
                "path": str(npz),
                "size_mb": round(npz.stat().st_size / (1024 * 1024), 3),
                "epoch": doc.get("epoch", 0),
                "step": doc.get("step", 0),
                "loss": doc.get("loss"),
                "vocab_size": doc.get("vocab_size", 0),
            }
        )
    return checkpoints
