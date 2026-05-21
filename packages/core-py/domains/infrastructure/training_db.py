"""
Simple Training Database - Save conversations and train directly from them.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path
import threading
import random


@dataclass
class ConvPair:
    """Single conversation pair (prompt + response)."""

    id: str
    session_id: str
    prompt: str
    response: str
    model: str
    timestamp: str
    quality: float
    feedback: Optional[str] = None
    used: bool = False


class TrainingDB:
    """Database for training data with balanced sampling."""

    def __init__(self, path: str = None):
        if path is None:
            # Default to repo root data dir (go up 5 levels from this file to repo root)
            path = Path(__file__).parents[4] / "data" / "training.db"
        else:
            path = Path(path)

        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: Optional[List[Dict]] = None
        print(f"[TrainingDB] Initialized at: {self.path}")

    def _load(self) -> List[Dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text())
        except:
            return []

    def _save(self, data: List[Dict]):
        with self._lock:
            self.path.write_text(json.dumps(data, indent=2))
            self._cache = data

    def _ensure(self) -> List[Dict]:
        if self._cache is None:
            self._cache = self._load()
        return self._cache

    def add(self, session_id: str, prompt: str, response: str, model: str = "gpt2") -> ConvPair:
        data = self._ensure()
        pair = ConvPair(
            id=f"pair_{len(data)}_{int(datetime.now().timestamp() * 1000)}",
            session_id=session_id,
            prompt=prompt,
            response=response,
            model=model,
            timestamp=datetime.now().isoformat(),
            quality=0.5,
        )
        data.append(asdict(pair))
        self._save(data)
        print(f"[TrainingDB] Added pair to {self.path}: {pair.id}")
        return pair

    def set_feedback(self, pair_id: str, feedback: str) -> bool:
        data = self._ensure()
        for pair in data:
            if pair["id"] == pair_id:
                pair["feedback"] = feedback
                pair["quality"] = 1.0 if feedback == "up" else 0.0
                self._save(data)
                return True
        return False

    def get_pairs(
        self, min_quality: float = 0.0, unused_only: bool = False, limit: int = None
    ) -> List[ConvPair]:
        data = self._ensure()
        pairs = [ConvPair(**p) for p in data if p.get("quality", 0.5) >= min_quality]
        if unused_only:
            pairs = [p for p in pairs if not p.used]
        if limit:
            pairs = pairs[-limit:]
        return pairs

    def search(self, query: str, top_k: int = 5) -> List[ConvPair]:
        """Simple text search in prompts and responses."""
        data = self._ensure()
        query_lower = query.lower()

        results = []
        for p in data:
            if (
                query_lower in p.get("prompt", "").lower()
                or query_lower in p.get("response", "").lower()
            ):
                results.append(ConvPair(**p))

        return results[:top_k]

    def get_balanced_pairs(self, target_count: int = 100) -> List[ConvPair]:
        """Get balanced pairs: equal positive/negative/neutral."""
        data = self._ensure()
        pairs = [ConvPair(**p) for p in data]

        positive = [p for p in pairs if p.quality >= 0.8]
        negative = [p for p in pairs if p.quality <= 0.2]
        neutral = [p for p in pairs if 0.2 < p.quality < 0.8]

        target_per = target_count // 3
        balanced = []

        random.shuffle(positive)
        balanced.extend(positive[:target_per])

        random.shuffle(negative)
        balanced.extend(negative[:target_per])

        random.shuffle(neutral)
        balanced.extend(neutral[:target_per])

        random.shuffle(balanced)
        return balanced

    def get_weighted_pairs(self, target_count: int = 100, strategy: str = "inverse") -> List[Dict]:
        """Get weighted pairs for training."""
        data = self._ensure()
        pairs = [ConvPair(**p) for p in data]

        if not pairs:
            return []

        weighted = []
        for p in pairs:
            if strategy == "inverse":
                weight = 1 - abs(p.quality - 0.5) * 2
            elif strategy == "positive":
                weight = p.quality
            else:
                weight = 0.5 - abs(p.quality - 0.5)

            weighted.append({"pair": p, "weight": max(0.1, weight)})

        weighted.sort(key=lambda x: x["weight"], reverse=True)
        selected = weighted[:target_count]

        return [
            {
                "prompt": item["pair"].prompt,
                "response": item["pair"].response,
                "quality": item["pair"].quality,
                "weight": item["weight"],
            }
            for item in selected
        ]

    def mark_used(self, pair_ids: List[str]):
        data = self._ensure()
        for pair in data:
            if pair["id"] in pair_ids:
                pair["used"] = True
        self._save(data)

    def get_stats(self) -> Dict:
        """Get database stats."""
        data = self._ensure()
        total = len(data)

        if total == 0:
            return {
                "total_pairs": 0,
                "positive_pairs": 0,
                "negative_pairs": 0,
                "neutral_pairs": 0,
                "unused_pairs": 0,
                "file_size_bytes": 0,
            }

        positive = len([p for p in data if p.get("quality", 0.5) >= 0.8])
        negative = len([p for p in data if p.get("quality", 0.5) <= 0.2])
        neutral = total - positive - negative
        unused = len([p for p in data if not p.get("used", False)])

        return {
            "total_pairs": total,
            "positive_pairs": positive,
            "negative_pairs": negative,
            "neutral_pairs": neutral,
            "unused_pairs": unused,
            "file_size_bytes": self.path.stat().st_size if self.path.exists() else 0,
        }


_db: Optional[TrainingDB] = None
_db_lock = threading.Lock()


def get_training_db(path: str = None) -> TrainingDB:
    global _db
    with _db_lock:
        if _db is None:
            _db = TrainingDB(path)
        return _db


__all__ = ["ConvPair", "TrainingDB", "get_training_db"]
