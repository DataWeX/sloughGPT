"""Resolve training corpus paths from legacy folders or v1 dataset manifests.

This module does not write checkpoints. Trainer ``*.soul`` char vocab on disk is
documented under *Checkpoint vocabulary* in ``docs/policies/CONTRIBUTING.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .schemas import TrainDatasetRef


def _repo_root() -> Path:
    from pathlib import Path

    from domain.shared import find_repo_root

    return find_repo_root(Path(__file__).resolve())


def materialize_source_text(text: str, name: str) -> str:
    """Turn pasted training text into a just-cache dataset; return its id.

    The paste-text flow has no dataset on disk, so writing one through the
    normal dataset controller keeps resolution, pre-flight size checks,
    preview and recovery on the single existing path — instead of teaching
    every downstream step a second source kind.
    """
    import re
    import time

    size_bytes = len(text.encode("utf-8"))
    if size_bytes < 100:
        raise ValueError(f"Pasted text is too small ({size_bytes} bytes). Need at least 100 bytes.")

    from controllers.datasets import get_datasets_controller

    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", name or "").strip("-_")[:48].strip("-_") or "text"
    dataset_id = f"paste-{slug}-{int(time.time())}"

    ctrl = get_datasets_controller()
    ctrl.create_dataset(dataset_id, description="Pasted text for training")
    rows = [line for line in text.splitlines() if line.strip()] or [text]
    ctrl.add_data(dataset_id, rows)
    return dataset_id


def resolve_legacy_corpus_path(stem: str) -> Path | None:
    """Find a training corpus file for a dataset id across legacy locations.

    Search order (first hit wins, preserving the historic priority):
    ``datasets/{stem}/input.txt`` (exact legacy contract), then
    ``data/{stem}`` and ``data/datasets/{stem}`` via the shared
    ``find_corpus_file`` priority (corpus.jsonl, input.txt, train.txt,
    text.txt, *.txt, *.jsonl).

    Returns the corpus file Path, or None when nothing resolves.
    """
    from domain.training._internal.cache_tags import find_corpus_file

    root = _repo_root()
    legacy = root / "datasets" / stem / "input.txt"
    if legacy.is_file():
        return legacy
    for base in ("data", "data/datasets"):
        candidate_dir = root / base / stem
        if candidate_dir.is_dir():
            hit = find_corpus_file(candidate_dir)
            if hit is not None:
                return hit
    return None


def resolve_training_inputs(
    dataset: str | None,
    manifest_uri: str | None,
    dataset_ref: TrainDatasetRef | None,
) -> tuple[str, str, dict[str, Any] | None, str]:
    """
    Returns (data_path_str, out_stem, manifest_meta | None, source_kind).

    source_kind is ``legacy`` | ``manifest`` | ``ref``.
    """

    from domain.training._internal.dataset_manifest import ManifestError, resolve_training_data_path

    manifest_meta: dict[str, Any] | None = None

    if dataset_ref is not None:
        ref = dataset_ref
        data_path, manifest = resolve_training_data_path(ref.manifest_uri)
        if manifest.get("dataset_id") != ref.dataset_id:
            raise ManifestError(
                f"dataset_ref.dataset_id {ref.dataset_id!r} does not match "
                f"manifest dataset_id {manifest.get('dataset_id')!r}"
            )
        if str(manifest.get("version")) != str(ref.version):
            raise ManifestError(
                f"dataset_ref.version {ref.version!r} does not match "
                f"manifest version {manifest.get('version')!r}"
            )
        manifest_meta = {"dataset_id": manifest["dataset_id"], "version": manifest["version"]}
        return str(data_path), ref.dataset_id, manifest_meta, "ref"

    if manifest_uri is not None and str(manifest_uri).strip():
        data_path, manifest = resolve_training_data_path(manifest_uri)
        manifest_meta = {"dataset_id": manifest["dataset_id"], "version": manifest["version"]}
        out_stem = str(manifest.get("dataset_id", "dataset"))
        return str(data_path), out_stem, manifest_meta, "manifest"

    stem = str(dataset).strip()
    # Just-cache first; legacy datasets/ dir is a read-only fallback.
    try:
        from domain.training._internal.cache_tags import resolve_in_cache

        hit = resolve_in_cache(stem)
        if hit:
            return hit, stem, None, "cache"
    except ValueError:
        pass
    hit = resolve_legacy_corpus_path(stem)
    if hit is None:
        searched = ["datasets/{s}/input.txt", "data/{s}/", "data/datasets/{s}/"]
        locations = ", ".join(loc.format(s=stem) for loc in searched)
        raise ManifestError(f"Missing training file for {stem!r} (searched: {locations})")
    return str(hit.resolve()), stem, None, "legacy"
