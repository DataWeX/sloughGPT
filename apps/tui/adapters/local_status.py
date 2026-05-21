"""Local filesystem snapshot for TUI read-only “status” room (mirrors ``cmd_status`` layout)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class LocalStatusSnapshot:
    """Counts under ``repo_root`` for models/ and datasets/."""

    repo_root: Path
    models_dir_found: bool
    model_file_count: int
    datasets_dir_found: bool
    dataset_entry_count: int
    model_sample_paths: List[str] = field(default_factory=list)
    dataset_sample_names: List[str] = field(default_factory=list)


def scan_local_repo(repo_root: Path, *, model_sample_limit: int = 5, dataset_sample_limit: int = 5) -> LocalStatusSnapshot:
    """
    Scan ``models/**.pt`` / ``**.pth`` and top-level ``datasets/`` entries (same idea as ``cmd_status``).
    Paths in samples are POSIX strings relative to ``repo_root``.
    """
    root = repo_root.resolve()
    models_dir = root / "models"
    datasets_dir = root / "datasets"

    model_files: List[Path] = []
    if models_dir.is_dir():
        model_files = list(models_dir.rglob("*.pt")) + list(models_dir.rglob("*.pth"))

    rel_models = []
    for p in model_files[:model_sample_limit]:
        try:
            rel_models.append(p.relative_to(root).as_posix())
        except ValueError:
            rel_models.append(str(p))

    ds_names: List[str] = []
    ds_count = 0
    if datasets_dir.is_dir():
        entries = sorted(datasets_dir.iterdir(), key=lambda x: x.name.lower())
        ds_count = len(entries)
        ds_names = [e.name for e in entries[:dataset_sample_limit]]

    return LocalStatusSnapshot(
        repo_root=root,
        models_dir_found=models_dir.is_dir(),
        model_file_count=len(model_files),
        model_sample_paths=rel_models,
        datasets_dir_found=datasets_dir.is_dir(),
        dataset_entry_count=ds_count,
        dataset_sample_names=ds_names,
    )
