"""
Dataset Manager - Ported from recovered dataset_manager.py
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np


class DatasetInfo:
    """Dataset information container."""

    def __init__(self, name: str, path: Path, meta: Optional[Dict[str, Any]] = None):
        self.name = name
        self.path = path
        self.meta = meta or {}
        self.vocab_size = self.meta.get("vocab_size", 0) if self.meta else 0
        self.train_tokens = self.meta.get("train_tokens", 0) if self.meta else 0
        self.val_tokens = self.meta.get("val_tokens", 0) if self.meta else 0
        self.total_characters = self.meta.get("total_characters", 0) if self.meta else 0
        self.source_files = self.meta.get("source_files", []) if self.meta else []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "vocab_size": self.vocab_size,
            "train_tokens": self.train_tokens,
            "val_tokens": self.val_tokens,
            "total_characters": self.total_characters,
            "source_files": self.source_files,
            "meta": self.meta,
        }


class DatasetRegistry:
    """Central registry for all datasets."""

    def __init__(self, registry_file: str = "datasets/registry.json"):
        self.registry_file = Path(registry_file)
        self.datasets: Dict[str, DatasetInfo] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r") as f:
                    data = json.load(f)
                    for name, info in data.items():
                        self.datasets[name] = DatasetInfo(
                            name=name, path=Path(info["path"]), meta=info.get("meta")
                        )
            except Exception:
                pass

    def _save_registry(self) -> None:
        """Save registry to file."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        data = {name: info.to_dict() for name, info in self.datasets.items()}
        with open(self.registry_file, "w") as f:
            json.dump(data, f, indent=2)

    def register(
        self, name: str, path: Union[str, Path], meta: Optional[Dict[str, Any]] = None
    ) -> DatasetInfo:
        """Register a new dataset."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")

        meta_path = path / "meta.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                loaded_meta = pickle.load(f)
                meta = {**loaded_meta, **(meta or {})}

        info = DatasetInfo(name, path, meta)
        self.datasets[name] = info
        self._save_registry()
        return info

    def get(self, name: str) -> Optional[DatasetInfo]:
        """Get dataset by name."""
        return self.datasets.get(name)

    def list_datasets(self) -> List[DatasetInfo]:
        """List all registered datasets."""
        return list(self.datasets.values())

    def load_data(self, name: str, split: str = "train") -> Optional[np.ndarray]:
        """Load dataset data."""
        info = self.get(name)
        if not info:
            return None

        filename = f"{split}.bin"
        data_path = info.path / filename

        if not data_path.exists():
            return None

        return np.fromfile(data_path, dtype=np.uint16)

    def unregister(self, name: str) -> bool:
        """Unregister a dataset."""
        if name in self.datasets:
            del self.datasets[name]
            self._save_registry()
            return True
        return False


class DatasetMixer:
    """Mix multiple datasets with different weights."""

    def __init__(self, registry: DatasetRegistry):
        self.registry = registry
        self.mix_config: Dict[str, float] = {}

    def set_mix(self, config: Dict[str, float]) -> None:
        """Set dataset mixing configuration."""
        total = sum(config.values())
        self.mix_config = {k: v / total for k, v in config.items()}

    def sample_batch(self, batch_size: int, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a batch from mixed datasets."""
        datasets = list(self.mix_config.keys())
        weights = list(self.mix_config.values())

        selected = np.random.choice(datasets, p=weights)
        data = self.registry.load_data(selected, "train")

        if data is None or len(data) < block_size + 1:
            raise ValueError(f"Dataset {selected} not available")

        ix = np.random.randint(0, len(data) - block_size - 1, (batch_size,))

        x = np.array([data[i : i + block_size] for i in ix])
        y = np.array([data[i + 1 : i + block_size + 1] for i in ix])

        return x, y


__all__ = ["DatasetInfo", "DatasetRegistry", "DatasetMixer"]
