"""
Unified Training Module
Provides DataLoader and utilities for the SloughGPT framework.

Note: For training, use SloughGPTTrainer from train_pipeline.
TrainingConfig is deprecated - use TrainerConfig from train_pipeline.
"""

import json
import logging
from typing import Optional, List, Dict, Any, Iterator, Callable
from pathlib import Path

from domains.training.slonet_compat import torch, nn
Dataset = torch.utils.data.Dataset
DataLoader = torch.utils.data.DataLoader


logger = logging.getLogger(__name__)


# Canonical imports
from domains.training.train_pipeline import TrainerConfig, TextDataset

TrainingConfig = TrainerConfig  # Alias for backwards compatibility


class UniversalDataLoader:
    """Universal data loader supporting multiple formats."""

    def __init__(self, data_path: str, block_size: int = 128):
        self.data_path = Path(data_path)
        self.block_size = block_size
        self.texts: List[str] = []
        self.data: List[int] = []
        self.vocab_size: int = 0
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}

        self._load_data()

    def _load_data(self):
        """Load data from file."""
        suffix = self.data_path.suffix.lower()

        if suffix == ".txt":
            self._load_text()
        elif suffix == ".json":
            self._load_json()
        elif suffix == ".jsonl":
            self._load_jsonl()
        elif suffix == ".csv":
            self._load_csv()
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _load_text(self):
        """Load plain text file."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            text = f.read()

        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.data = [self.stoi[ch] for ch in text]
        self.texts = [text]

        logger.info(f"Loaded text: {len(self.data)} tokens, {self.vocab_size} vocab")

    def _load_json(self):
        """Load JSON file."""
        with open(self.data_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            self.texts = [item["text"] if isinstance(item, dict) else str(item) for item in data]
        elif isinstance(data, dict):
            self.texts = [data.get("text", str(data))]

        self._build_vocab()

    def _load_jsonl(self):
        """Load JSONL file."""
        self.texts = []
        with open(self.data_path, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    text = item["text"] if isinstance(item, dict) else str(item)
                    self.texts.append(text)

        self._build_vocab()

    def _load_csv(self):
        """Load CSV file."""
        import csv

        self.texts = []
        with open(self.data_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.texts.append(row.get("text", ""))

        self._build_vocab()

    def _build_vocab(self):
        """Build vocabulary from texts."""
        chars = set()
        for text in self.texts:
            chars.update(text)

        chars = sorted(chars)
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        # Encode all texts
        self.data = []
        for text in self.texts:
            self.data.extend([self.stoi.get(ch, 0) for ch in text])

        logger.info(
            f"Loaded {len(self.texts)} texts: {len(self.data)} tokens, {self.vocab_size} vocab"
        )

    def get_batch(self, batch_size: int, block_size: Optional[int] = None) -> Iterator:
        """Get batch generator."""
        block_size = block_size or self.block_size

        # Create indices
        n = len(self.data) - block_size
        indices = torch.randperm(n)[: batch_size * 100]  # Limit to avoid memory issues

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]

            x = torch.stack(
                [
                    torch.tensor(self.data[idx : idx + block_size], dtype=torch.long)
                    for idx in batch_indices
                ]
            )
            y = torch.stack(
                [
                    torch.tensor(self.data[idx + 1 : idx + block_size + 1], dtype=torch.long)
                    for idx in batch_indices
                ]
            )

            yield x, y

    def create_dataloader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        """Create PyTorch DataLoader."""
        dataset = TextDataset(self.data, self.block_size)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class ModelWrapper:
    """Model wrapper base class."""

    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config


class TorchModelWrapper(ModelWrapper):
    """PyTorch model wrapper."""

    def __init__(self, model: nn.Module, config: TrainingConfig):
        super().__init__(model, config)
        self.model = model.to(config.device)


__all__ = [
    "TrainingConfig",
    "TextDataset",
    "UniversalDataLoader",
    "ModelWrapper",
    "TorchModelWrapper",
]


def train(config: TrainingConfig):
    """Train function for quick training using SloughGPTTrainer."""
    from domains.training.train_pipeline import SloughGPTTrainer
    trainer = SloughGPTTrainer(data_path=config.data_path, config=config)
    trainer.train()
    return trainer
