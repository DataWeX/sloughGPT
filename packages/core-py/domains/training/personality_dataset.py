"""
Personality Dataset Loader for SloughGPT

Loads and processes personality training data.

SloDataset / SloDataLoader from slonet is the native runtime.
Torch Dataset/DataLoader can be used when HuggingFace tokenizers are needed.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
import random

import numpy as np

from domains.training.slonet import SloDataset, SloDataLoader

logger = logging.getLogger("sloughgpt.personality")


@dataclass
class PersonalitySample:
    """Single personality training sample."""
    text: str
    traits: Dict[str, float]
    embedding: Optional[np.ndarray] = None


class PersonalityDataset(SloDataset):
    """
    Dataset for personality training.

    Supports:
    — Text + trait labels
    — Archetype labels
    — Contrastive pairs
    Yields numpy arrays (not torch tensors).
    """

    def __init__(self, samples: List[PersonalitySample], tokenizer: Optional[Callable] = None,
                 max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        result: Dict[str, Any] = {"text": sample.text, "traits": sample.traits}

        if self.tokenizer:
            tokens = self.tokenizer(
                sample.text, max_length=self.max_length,
                padding="max_length", truncation=True, return_tensors="np",
            )
            result["input_ids"] = tokens["input_ids"].squeeze(0)
            result["attention_mask"] = tokens["attention_mask"].squeeze(0)

        return result


class PersonalityDatasetLoader:
    """Loads personality datasets from various sources."""

    @staticmethod
    def from_jsonl(path: str) -> List[PersonalitySample]:
        samples = []
        with open(path, "r") as f:
            for line in f:
                data = json.loads(line)
                samples.append(PersonalitySample(text=data["text"], traits=data.get("traits", {})))
        logger.info(f"Loaded {len(samples)} samples from {path}")
        return samples

    @staticmethod
    def from_json(path: str) -> List[PersonalitySample]:
        with open(path, "r") as f:
            data = json.load(f)
        samples = [PersonalitySample(text=item["text"], traits=item.get("traits", {}))
                   for item in data]
        logger.info(f"Loaded {len(samples)} samples from {path}")
        return samples

    @staticmethod
    def from_csv(path: str) -> List[PersonalitySample]:
        import csv
        samples = []
        with open(path, "r") as f:
            for row in csv.DictReader(f):
                traits = {}
                for key, val in row.items():
                    if key != "text" and val:
                        try:
                            traits[key] = float(val)
                        except ValueError:
                            pass
                samples.append(PersonalitySample(text=row["text"], traits=traits))
        logger.info(f"Loaded {len(samples)} samples from {path}")
        return samples

    @staticmethod
    def create_synthetic(num_samples: int = 1000,
                         templates: Optional[List] = None) -> List[PersonalitySample]:
        if templates is None:
            templates = [
                ("Hello! How are you?", {"warmth": 0.9, "formality": -0.5}),
                ("Good morning.", {"warmth": 0.3, "formality": 0.8}),
                ("OMG that's so cool!", {"warmth": 0.9, "creativity": 0.7, "humor": 0.9}),
                ("I understand your concern.", {"empathy": 0.9, "patience": 0.8}),
                ("The solution is simple.", {"confidence": 0.9, "directness": 0.8}),
                ("Let me explain step by step.", {"patience": 0.9, "formality": 0.3}),
                ("Absolutely! Great idea!", {"warmth": 0.9, "creativity": 0.5}),
                ("I disagree with that approach.", {"confidence": 0.7, "directness": 0.9}),
            ]
        variations = [
            "Hey there!", "Hi!", "Hello", "Greetings", "Good day",
            "Please", "Kindly", "Would you mind", "Could you",
            "Definitely", "Absolutely", "Sure thing", "Of course",
            "That's interesting", "Really?", "I see", "Interesting point",
        ]
        samples = []
        for _ in range(num_samples):
            template, base_traits = random.choice(templates)
            for var in random.sample(variations, min(2, len(variations))):
                if random.random() > 0.5:
                    template = var + " " + template.lower()
            traits = {k: max(0.0, min(1.0, v + random.gauss(0, 0.1)))
                      for k, v in base_traits.items()}
            samples.append(PersonalitySample(text=template, traits=traits))
        logger.info(f"Created {len(samples)} synthetic samples")
        return samples

    @staticmethod
    def create_archetype_samples(archetypes: Optional[Dict[str, Dict[str, float]]] = None,
                                 samples_per_archetype: int = 50) -> List[PersonalitySample]:
        if archetypes is None:
            archetypes = {
                "sage": {"wisdom": 0.9, "patience": 0.8, "formality": 0.5},
                "innocent": {"warmth": 0.9, "humor": 0.6, "creativity": 0.5},
                "explorer": {"creativity": 0.9, "adventure": 0.9, "warmth": 0.5},
                "caregiver": {"empathy": 0.9, "warmth": 0.9, "patience": 0.8},
                "ruler": {"confidence": 0.9, "formality": 0.8, "directness": 0.7},
                "rebel": {"confidence": 0.9, "directness": 0.9, "creativity": 0.7},
                "magician": {"creativity": 0.9, "confidence": 0.8, "humor": 0.5},
                "jester": {"humor": 0.9, "warmth": 0.7, "creativity": 0.8},
            }
        archetype_texts = {
            "sage": ["Let me explain", "The answer is", "Research shows", "Based on evidence"],
            "innocent": ["How wonderful!", "I believe", "Let's be friends", "Everything will be okay"],
            "explorer": ["What if we", "Let's try", "Adventure awaits", "Discover new"],
            "caregiver": ["I understand", "How can I help", "You're not alone", "Take care"],
            "ruler": ["The rules are", "Order must be", "Follow the", "Structure is"],
            "rebel": ["Break the rules", "Question authority", "Think different", "Challenge norms"],
            "magician": ["Imagine if", "The possibilities", "Make it happen", "Transform this"],
            "jester": ["Laugh at", "Fun time!", "Joke time", "Enjoy the"],
        }
        samples = []
        for archetype, traits in archetypes.items():
            texts = archetype_texts.get(archetype, ["Sample text"])
            for _ in range(samples_per_archetype):
                text = random.choice(texts)
                if random.random() > 0.5:
                    text += " " + random.choice(["today", "now", "please", "together"])
                samples.append(PersonalitySample(text=text, traits=traits))
        logger.info(f"Created {len(samples)} archetype samples")
        return samples


class PersonalityDataCollator:
    """Collates personality batches. Returns numpy arrays."""

    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [item["text"] for item in batch]
        encoded = self.tokenizer(
            texts, max_length=self.max_length, padding=True,
            truncation=True, return_tensors="np",
        )

        all_traits = [item["traits"] for item in batch]
        trait_keys = set()
        for t in all_traits:
            trait_keys.update(t.keys())

        trait_dict = {k: [] for k in trait_keys}
        for traits in all_traits:
            for k in trait_keys:
                trait_dict[k].append(traits.get(k, 0.0))

        trait_arr = np.array([trait_dict[k] for k in sorted(trait_keys)], dtype=np.float32)

        return {
            "input_ids": np.asarray(encoded["input_ids"]),
            "attention_mask": np.asarray(encoded["attention_mask"]),
            "traits": trait_arr,
        }


def create_personality_dataloader(
    data_source: str,
    batch_size: int = 8,
    tokenizer: Optional[Callable] = None,
    max_length: int = 512,
    shuffle: bool = True,
):
    """
    Create a SloDataLoader for personality training.

    Args:
        data_source: Path to data file or "synthetic" or "archetype"
        batch_size: Batch size
        tokenizer: Tokenizer for encoding
        max_length: Max sequence length
        shuffle: Whether to shuffle

    Returns:
        SloDataLoader
    """
    path = Path(data_source)

    if data_source == "synthetic":
        samples = PersonalityDatasetLoader.create_synthetic()
    elif data_source == "archetype":
        samples = PersonalityDatasetLoader.create_archetype_samples()
    elif path.suffix == ".jsonl":
        samples = PersonalityDatasetLoader.from_jsonl(str(path))
    elif path.suffix == ".json":
        samples = PersonalityDatasetLoader.from_json(str(path))
    elif path.suffix == ".csv":
        samples = PersonalityDatasetLoader.from_csv(str(path))
    else:
        raise ValueError(f"Unknown data source: {data_source}")

    dataset = PersonalityDataset(samples, tokenizer, max_length)
    collator = PersonalityDataCollator(tokenizer, max_length) if tokenizer else None

    return SloDataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator,
    )


__all__ = [
    "PersonalitySample",
    "PersonalityDataset",
    "PersonalityDatasetLoader",
    "PersonalityDataCollator",
    "create_personality_dataloader",
]
