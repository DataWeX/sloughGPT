"""Cruncher snapshots — immutable, versioned training datasets.

Layout under ``data/crunched/``::

    data/crunched/<dataset_id>/
        v1/
            manifest.json      # SloughGPT Standard v1 (+ cruncher extras)
            train.txt          # UTF-8 text the char trainer consumes
            dedup_state.json   # MinHash index AFTER v1's records (basis for v2+)
        v2/
            ...

Immutability contract: sealing ``v<N+1>`` never touches ``v<N>``. Re-running
a crunch on the same dataset bumps the version instead of overwriting, so a
bad data pack is rolled back by pointing at the previous version — the running
model is never corrupted in place.

The manifest is a superset of SloughGPT Standard v1
(``domain/training/_internal/dataset_manifest.py``): required keys stay
required, cruncher extras (``mix``, ``stats``, ``cruncher``) are additive.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from domain.shared import find_repo_root

logger = logging.getLogger("slo.training.cruncher")

MANIFEST_SCHEMA_VERSION = "1.0"
DEFAULT_DOMAIN = "general"
DEFAULT_PII_POLICY = "scrub-typed-tokens"


def _default_root() -> Path:
    return find_repo_root(Path(__file__).resolve()) / "data" / "crunched"


@dataclass(frozen=True)
class SnapshotRef:
    """Pointer to one sealed snapshot."""

    dataset_id: str
    version: int
    path: Path

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    @property
    def train_path(self) -> Path:
        return self.path / "train.txt"

    @property
    def dedup_state_path(self) -> Path:
        return self.path / "dedup_state.json"


class SnapshotError(RuntimeError):
    """Snapshot store violation (missing dataset, bad version, overwrite)."""


class SnapshotStore:
    """Immutable versioned snapshot store rooted at ``data/crunched/``."""

    def __init__(self, root: Path | None = None):
        self._root = Path(root) if root else _default_root()

    @property
    def root(self) -> Path:
        return self._root

    def dataset_dir(self, dataset_id: str) -> Path:
        return self._root / dataset_id

    def versions(self, dataset_id: str) -> list[int]:
        """Sealed versions of *dataset_id*, ascending. Empty if none."""
        ds = self.dataset_dir(dataset_id)
        if not ds.is_dir():
            return []
        out = []
        for p in ds.iterdir():
            if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit():
                if (p / "manifest.json").is_file():
                    out.append(int(p.name[1:]))
        return sorted(out)

    def latest(self, dataset_id: str) -> SnapshotRef | None:
        """Most recent sealed snapshot, or None."""
        vs = self.versions(dataset_id)
        if not vs:
            return None
        v = vs[-1]
        return SnapshotRef(dataset_id, v, self.dataset_dir(dataset_id) / f"v{v}")

    def seal(
        self,
        dataset_id: str,
        records: Iterable[str],
        *,
        sources: list[str],
        mix: dict[str, float] | None = None,
        domain: str = DEFAULT_DOMAIN,
        pii_policy: str = DEFAULT_PII_POLICY,
        dedup_state: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> SnapshotRef:
        """Seal *records* as the next immutable version of *dataset_id*.

        Writes to a temp directory then renames into place — a crash mid-seal
        never leaves a half-written ``v<N>`` visible to the store.

        Args:
            dataset_id: dataset identity (directory name).
            records: cleaned training-text records, one per line in train.txt.
            sources: provenance strings (e.g. ``"kaggle:owner/name"``).
            mix: reserved pack-mix weights (consumed when multi-pack lands).
            domain: dataset domain (general unless a knowledge-pack says so).
            pii_policy: recorded PII handling policy.
            dedup_state: serializable DedupIndex state saved alongside, so the
                next version dedups against this corpus.
            stats: ingest stats recorded in the manifest for auditability.

        Returns:
            SnapshotRef of the newly sealed version.

        Raises:
            SnapshotError: empty record set or dataset_id contains a path
                separator.
        """
        if not dataset_id or any(c in dataset_id for c in "/\\"):
            raise SnapshotError(f"invalid dataset_id {dataset_id!r}")
        texts = [r for r in records if r]
        if not texts:
            raise SnapshotError(f"cannot seal {dataset_id!r}: no records")

        vs = self.versions(dataset_id)
        version = (vs[-1] + 1) if vs else 1
        final_dir = self.dataset_dir(dataset_id) / f"v{version}"
        if final_dir.exists():
            raise SnapshotError(f"snapshot {final_dir} already exists — refusing to overwrite")

        self.dataset_dir(dataset_id).mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(
            tempfile.mkdtemp(prefix=f".v{version}.tmp-", dir=self.dataset_dir(dataset_id))
        )
        try:
            (tmp_dir / "train.txt").write_text("\n".join(texts) + "\n", encoding="utf-8")
            manifest = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "dataset_id": dataset_id,
                "version": version,
                "domain": domain,
                "pii_policy": pii_policy,
                "sources": sources,
                "splits": {"train": "train.txt"},
                # Reserved: pack-mix weights, consumed when multi-pack training lands.
                "mix": mix or {},
                "cruncher": {
                    "sealed_at": time.time(),
                    "record_count": len(texts),
                    "stats": stats or {},
                },
            }
            (tmp_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            if dedup_state is not None:
                (tmp_dir / "dedup_state.json").write_text(json.dumps(dedup_state), encoding="utf-8")
            os.rename(tmp_dir, final_dir)
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        logger.info(
            "sealed %s v%d (%d records)", dataset_id, version, len(texts), extra={"tag": "TRAIN"}
        )
        return SnapshotRef(dataset_id, version, final_dir)

    def load_manifest(self, ref: SnapshotRef) -> dict[str, Any]:
        """Read a snapshot's manifest.json."""
        data: dict[str, Any] = json.loads(ref.manifest_path.read_text(encoding="utf-8"))
        return data

    def load_dedup_state(self, ref: SnapshotRef) -> dict[str, Any] | None:
        """Read the dedup index state sealed with *ref*, or None."""
        if not ref.dedup_state_path.is_file():
            return None
        state: dict[str, Any] = json.loads(ref.dedup_state_path.read_text(encoding="utf-8"))
        return state
