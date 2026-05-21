"""
Dataset Preparation Tools
Prepare, clean, and tokenize datasets for training.
"""

import re
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from collections import Counter

import numpy as np

logger = logging.getLogger("sloughgpt.dataset_prep")


@dataclass
class DatasetStats:
    """Statistics about a dataset."""

    name: str
    total_chars: int
    total_words: int
    total_lines: int
    unique_chars: int
    vocab_size: int
    avg_word_length: float
    avg_line_length: float


class TextCleaner:
    """Clean and preprocess text data."""

    def __init__(self):
        self.steps = []

    def lowercase(self) -> "TextCleaner":
        self.steps.append(lambda x: x.lower())
        return self

    def remove_extra_whitespace(self) -> "TextCleaner":
        self.steps.append(lambda x: re.sub(r"\s+", " ", x))
        return self

    def remove_urls(self) -> "TextCleaner":
        self.steps.append(lambda x: re.sub(r"https?://\S+|www\.\S+", "", x))
        return self

    def remove_emails(self) -> "TextCleaner":
        self.steps.append(lambda x: re.sub(r"\S+@\S+", "", x))
        return self

    def remove_html(self) -> "TextCleaner":
        self.steps.append(lambda x: re.sub(r"<[^>]+>", "", x))
        return self

    def remove_special_chars(self, keep: str = "") -> "TextCleaner":
        pattern = rf"[^a-zA-Z0-9\s{re.escape(keep)}]"
        self.steps.append(lambda x: re.sub(pattern, "", x))
        return self

    def normalize_unicode(self) -> "TextCleaner":
        import unicodedata

        self.steps.append(lambda x: unicodedata.normalize("NFKC", x))
        return self

    def clean(self, text: str) -> str:
        for step in self.steps:
            text = step(text)
        return text.strip()


class Tokenizer:
    """Simple character and word tokenizers."""

    def __init__(self, vocab_size: int = 500):
        self.vocab_size = vocab_size
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}

    def build_char_vocab(self, text: str) -> "Tokenizer":
        chars = sorted(set(text))

        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

        for i, token in enumerate(special_tokens):
            self.char_to_idx[token] = i
            self.idx_to_char[i] = token

        for i, char in enumerate(chars[: self.vocab_size - 4]):
            idx = i + 4
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char

        return self

    def build_word_vocab(self, text: str, min_freq: int = 2) -> "Tokenizer":
        words = text.split()
        word_counts = Counter(words)

        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

        for i, token in enumerate(special_tokens):
            self.word_to_idx[token] = i
            self.idx_to_word[i] = token

        idx = 4
        for word, count in word_counts.most_common(self.vocab_size - 4):
            if count >= min_freq:
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                idx += 1

        return self

    def encode_chars(self, text: str) -> List[int]:
        return [self.char_to_idx.get(c, 1) for c in text]

    def decode_chars(self, tokens: List[int]) -> str:
        return "".join(self.idx_to_char.get(t, "?") for t in tokens)

    def encode_words(self, text: str) -> List[int]:
        return [self.word_to_idx.get(w, 1) for w in text.split()]

    def decode_words(self, tokens: List[int]) -> str:
        return " ".join(self.idx_to_word.get(t, "?") for t in tokens)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "char_to_idx": self.char_to_idx,
                    "idx_to_char": self.idx_to_char,
                    "word_to_idx": self.word_to_idx,
                    "idx_to_word": self.idx_to_word,
                    "vocab_size": self.vocab_size,
                },
                f,
            )

    def load(self, path: str) -> "Tokenizer":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.char_to_idx = data["char_to_idx"]
        self.idx_to_char = data["idx_to_char"]
        self.word_to_idx = data.get("word_to_idx", {})
        self.idx_to_word = data.get("idx_to_word", {})
        self.vocab_size = data["vocab_size"]
        return self


class DatasetPreparer:
    """Prepare datasets for training."""

    def __init__(self, output_dir: str = "datasets"):
        self.output_dir = Path(output_dir)
        self.cleaner = TextCleaner()
        self.tokenizer = Tokenizer()

    def prepare_from_text(
        self,
        text: str,
        name: str,
        clean: bool = True,
        split_ratio: float = 0.9,
    ) -> DatasetStats:
        if clean:
            text = self.cleaner.remove_extra_whitespace().clean(text)

        self.tokenizer.build_char_vocab(text)

        tokens = self.tokenizer.encode_chars(text)
        tokens = np.array(tokens, dtype=np.uint16)

        split_idx = int(len(tokens) * split_ratio)
        train_tokens = tokens[:split_idx]
        val_tokens = tokens[split_idx:]

        dataset_dir = self.output_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        train_tokens.tofile(dataset_dir / "train.bin")
        val_tokens.tofile(dataset_dir / "val.bin")

        with open(dataset_dir / "input.txt", "w") as f:
            f.write(text)

        self.tokenizer.save(dataset_dir / "meta.pkl")

        stats = DatasetStats(
            name=name,
            total_chars=len(text),
            total_words=len(text.split()),
            total_lines=len(text.splitlines()),
            unique_chars=len(set(text)),
            vocab_size=len(self.tokenizer.char_to_idx),
            avg_word_length=len(text) / max(len(text.split()), 1),
            avg_line_length=len(text) / max(len(text.splitlines()), 1),
        )

        logger.info(
            f"Dataset '{name}' prepared: {stats.total_chars:,} chars, {stats.vocab_size} vocab"
        )

        return stats

    def prepare_from_file(
        self,
        file_path: str,
        name: str,
        clean: bool = True,
        split_ratio: float = 0.9,
    ) -> DatasetStats:
        text = Path(file_path).read_text(encoding="utf-8")
        return self.prepare_from_text(text, name, clean, split_ratio)

    def prepare_from_directory(
        self,
        dir_path: str,
        name: str,
        patterns: List[str] = None,
        clean: bool = True,
        split_ratio: float = 0.9,
    ) -> DatasetStats:
        patterns = patterns or ["*.txt", "*.md"]

        texts = []
        dir_path = Path(dir_path)

        for pattern in patterns:
            for file in dir_path.rglob(pattern):
                try:
                    texts.append(file.read_text(encoding="utf-8"))
                except Exception as e:
                    logger.warning(f"Failed to read {file}: {e}")

        combined = "\n\n".join(texts)
        return self.prepare_from_text(combined, name, clean, split_ratio)

    def prepare_from_jsonl(
        self,
        file_path: str,
        name: str,
        text_field: str = "text",
        clean: bool = True,
        split_ratio: float = 0.9,
    ) -> DatasetStats:
        texts = []

        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    text = data.get(text_field) or data.get("content") or str(data)
                    texts.append(text)

        combined = "\n\n".join(texts)
        return self.prepare_from_text(combined, name, clean, split_ratio)


def prepare_dataset(
    source: str,
    name: str,
    output_dir: str = "datasets",
    clean: bool = True,
    split_ratio: float = 0.9,
) -> DatasetStats:
    """Convenience function to prepare a dataset."""
    preparer = DatasetPreparer(output_dir)
    source_path = Path(source)

    if source_path.is_dir():
        return preparer.prepare_from_directory(source, name, clean=clean, split_ratio=split_ratio)
    elif source_path.suffix == ".jsonl":
        return preparer.prepare_from_jsonl(source, name, clean=clean, split_ratio=split_ratio)
    else:
        return preparer.prepare_from_file(source, name, clean=clean, split_ratio=split_ratio)


__all__ = [
    "TextCleaner",
    "Tokenizer",
    "DatasetPreparer",
    "DatasetStats",
    "prepare_dataset",
]
