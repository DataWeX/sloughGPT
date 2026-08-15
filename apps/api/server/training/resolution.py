"""Resolve training corpus paths from legacy folders or v1 dataset manifests.

This module does not write checkpoints. Trainer ``*.soul`` char vocab on disk is
documented under *Checkpoint vocabulary* in ``docs/policies/CONTRIBUTING.md``.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Dict

from .schemas import TrainDatasetRef


def resolve_training_inputs(
    dataset: Optional[str],
    manifest_uri: Optional[str],
    dataset_ref: Optional[TrainDatasetRef],
) -> Tuple[str, str, Optional[Dict[str, Any]], str]:
    """
    Returns (data_path_str, out_stem, manifest_meta | None, source_kind).

    source_kind is ``legacy`` | ``manifest`` | ``ref``.
    """
    from pathlib import Path

    from domains.training.dataset_manifest import ManifestError, resolve_training_data_path

    manifest_meta: Optional[Dict[str, Any]] = None

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
    p = Path("datasets") / stem / "input.txt"
    if not p.is_file():
        raise ManifestError(f"Missing training file: {p}")
    return str(p.resolve()), stem, None, "legacy"
