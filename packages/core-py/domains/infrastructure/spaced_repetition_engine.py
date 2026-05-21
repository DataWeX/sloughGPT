"""Spaced Repetition Learning System as a modular engine."""

import time as time_module
from typing import Dict, List, Optional
from collections import defaultdict


class SpacedRepetitionScheduler:
    """Spaced Repetition Learning System.

    Schedules reviews based on performance:
    - Good performance (≥80%) → longer interval (up to 1 week)
    - Poor performance (<80%) → shorter interval (down to 1 day)
    """

    def __init__(self):
        self.review_schedule: Dict[str, float] = {}
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        self.intervals = {
            "day": 1 * 24 * 3600,
            "week": 7 * 24 * 3600,
            "month": 30 * 24 * 3600,
        }

    def schedule_review(self, doc_id: str, performance: float) -> float:
        """Schedule next review based on performance.

        Args:
            doc_id: Document ID
            performance: Score 0-1

        Returns:
            Next review timestamp
        """
        self.performance_history[doc_id].append(performance)
        avg_performance = sum(self.performance_history[doc_id]) / len(
            self.performance_history[doc_id]
        )
        if avg_performance >= 0.9:
            interval = self.intervals["month"]
        elif avg_performance >= 0.8:
            interval = self.intervals["week"]
        elif avg_performance >= 0.6:
            interval = 3 * 24 * 3600
        else:
            interval = self.intervals["day"]
        next_review = time_module.time() + interval
        self.review_schedule[doc_id] = next_review
        return next_review

    def get_due_reviews(self) -> List[str]:
        """Get list of documents due for review."""
        current_time = time_module.time()
        return [
            doc_id
            for doc_id, review_time in self.review_schedule.items()
            if current_time >= review_time
        ]

    def get_next_review_time(self, doc_id: str) -> Optional[float]:
        """Get next review time for a document."""
        return self.review_schedule.get(doc_id)

    def get_review_stats(self) -> Dict[str, any]:
        """Get spaced repetition statistics."""
        current_time = time_module.time()
        due = self.get_due_reviews()
        upcoming: Dict[str, float] = {}
        for doc_id, review_time in self.review_schedule.items():
            if review_time > current_time:
                days_until = (review_time - current_time) / (24 * 3600)
                upcoming[doc_id] = round(days_until, 1)
        return {
            "due_count": len(due),
            "due_documents": due,
            "total_scheduled": len(self.review_schedule),
            "upcoming_reviews": upcoming,
        }
