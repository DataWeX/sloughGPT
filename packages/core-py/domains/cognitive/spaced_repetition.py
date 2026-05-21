"""
Spaced Repetition Learning System
Ported from recovered slo_spaced_repetition.py
"""

import time
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum
import sqlite3


class Difficulty(Enum):
    VERY_EASY = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    VERY_HARD = 5


class MemoryStrength(Enum):
    WEAK = 1
    LEARNING = 2
    YOUNG = 3
    MATURE = 4
    STRONG = 5


@dataclass
class LearningItem:
    """Represents a single learning item in the spaced repetition system"""

    id: str
    content: str
    concept: str
    difficulty: Difficulty
    created_at: float
    last_reviewed: float
    next_review: float
    review_count: int
    success_count: int
    failure_count: int
    memory_strength: MemoryStrength
    ease_factor: float
    interval: float

    def __post_init__(self):
        if self.ease_factor == 0:
            self.ease_factor = 2.5
        if self.interval == 0:
            self.interval = 1.0


class SpacedRepetitionScheduler:
    """Spaced Repetition Learning System"""

    def __init__(self, db_path: str = "learning.db"):
        self.db_path = db_path
        self.conn = None
        self.items: Dict[str, LearningItem] = {}
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_items (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                concept TEXT NOT NULL,
                difficulty INTEGER NOT NULL,
                created_at REAL NOT NULL,
                last_reviewed REAL NOT NULL,
                next_review REAL NOT NULL,
                review_count INTEGER NOT NULL,
                success_count INTEGER NOT NULL,
                failure_count INTEGER NOT NULL,
                memory_strength INTEGER NOT NULL,
                ease_factor REAL NOT NULL,
                interval REAL NOT NULL
            )
        """)

        self.conn.commit()

    def add_item(
        self, content: str, concept: str, difficulty: Difficulty = Difficulty.MEDIUM
    ) -> str:
        """Add a new learning item."""
        item_id = f"item_{len(self.items)}_{int(time.time())}"

        now = time.time()
        item = LearningItem(
            id=item_id,
            content=content,
            concept=concept,
            difficulty=difficulty,
            created_at=now,
            last_reviewed=now,
            next_review=now,
            review_count=0,
            success_count=0,
            failure_count=0,
            memory_strength=MemoryStrength.WEAK,
            ease_factor=2.5,
            interval=1.0,
        )

        self.items[item_id] = item
        self._save_item(item)

        return item_id

    def _save_item(self, item: LearningItem) -> None:
        """Save item to database."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO learning_items
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                item.id,
                item.content,
                item.concept,
                item.difficulty.value,
                item.created_at,
                item.last_reviewed,
                item.next_review,
                item.review_count,
                item.success_count,
                item.failure_count,
                item.memory_strength.value,
                item.ease_factor,
                item.interval,
            ),
        )
        self.conn.commit()

    def get_due_items(self) -> List[LearningItem]:
        """Get items due for review."""
        now = time.time()
        return [item for item in self.items.values() if item.next_review <= now]

    def review_item(self, item_id: str, quality: int) -> None:
        """Review an item (quality 0-5)."""
        item = self.items.get(item_id)
        if not item:
            return

        item.review_count += 1
        item.last_reviewed = time.time()

        if quality >= 3:
            item.success_count += 1
            item.interval *= item.ease_factor
        else:
            item.failure_count += 1
            item.interval = max(1.0, item.interval * 0.5)

        item.next_review = time.time() + (item.interval * 86400)

        if item.review_count > 5:
            item.memory_strength = MemoryStrength.STRONG
        elif item.review_count > 3:
            item.memory_strength = MemoryStrength.MATURE
        elif item.review_count > 1:
            item.memory_strength = MemoryStrength.YOUNG
        else:
            item.memory_strength = MemoryStrength.LEARNING

        self._save_item(item)

    def get_statistics(self) -> Dict:
        """Get learning statistics."""
        return {
            "total_items": len(self.items),
            "due_items": len(self.get_due_items()),
            "total_reviews": sum(i.review_count for i in self.items.values()),
            "success_rate": sum(i.success_count for i in self.items.values())
            / max(1, sum(i.review_count for i in self.items.values())),
        }

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()


__all__ = ["SpacedRepetitionScheduler", "LearningItem", "Difficulty", "MemoryStrength"]
