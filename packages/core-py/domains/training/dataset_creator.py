"""
Dataset Creator - Ported from recovered create_dataset.py
Simple utility to create training datasets.
"""

import shutil
from pathlib import Path
from typing import Dict, Any
import pickle
import numpy as np


class DatasetCreator:
    """Simple dataset creator for training."""

    def __init__(self, output_dir: str = "datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_from_text(self, name: str, text: str, train_split: float = 0.9) -> Dict[str, Any]:
        """Create dataset from text."""
        dataset_dir = self.output_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        input_file = dataset_dir / "input.txt"
        with open(input_file, "w", encoding="utf-8") as f:
            f.write(text)

        return self._process_dataset(name, train_split)

    def create_from_file(
        self, name: str, file_path: str, train_split: float = 0.9
    ) -> Dict[str, Any]:
        """Create dataset from file."""
        dataset_dir = self.output_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        input_file = dataset_dir / "input.txt"
        shutil.copy(file_path, input_file)

        return self._process_dataset(name, train_split)

    def create_from_folder(
        self, name: str, folder_path: str, train_split: float = 0.9
    ) -> Dict[str, Any]:
        """Create dataset from folder of text files."""
        dataset_dir = self.output_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        folder = Path(folder_path)
        texts = []

        for file_path in folder.rglob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                texts.append(f.read())

        combined = "\n\n".join(texts)

        return self.create_from_text(name, combined, train_split)

    def _process_dataset(self, name: str, train_split: float = 0.9) -> Dict[str, Any]:
        """Process text into training data."""
        dataset_dir = self.output_dir / name
        input_file = dataset_dir / "input.txt"

        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()

        chars = sorted(set(text))
        vocab_size = len(chars)

        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        encoded = np.array([stoi[ch] for ch in text], dtype=np.uint16)

        n = len(encoded)
        split = int(n * train_split)

        train_data = encoded[:split]
        val_data = encoded[split:]

        train_data.tofile(dataset_dir / "train.bin")
        val_data.tofile(dataset_dir / "val.bin")

        meta = {
            "vocab_size": vocab_size,
            "train_tokens": len(train_data),
            "val_tokens": len(val_data),
            "total_characters": len(text),
            "chars": chars,
        }

        with open(dataset_dir / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        return {
            "success": True,
            "name": name,
            "vocab_size": vocab_size,
            "train_tokens": len(train_data),
            "val_tokens": len(val_data),
        }


def create_dataset(
    name: str,
    text: str = None,
    file_path: str = None,
    folder_path: str = None,
    output_dir: str = "datasets",
) -> Dict[str, Any]:
    """Create a dataset from text, file, or folder."""
    creator = DatasetCreator(output_dir)

    if text:
        return creator.create_from_text(name, text)
    elif file_path:
        return creator.create_from_file(name, file_path)
    elif folder_path:
        return creator.create_from_folder(name, folder_path)
    else:
        return {"success": False, "error": "No input provided"}


__all__ = ["DatasetCreator", "create_dataset"]
