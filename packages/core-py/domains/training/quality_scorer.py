"""
Quality Scorer - Ported from recovered quality_scorer.py
Automated dataset quality evaluation
"""

from pathlib import Path
from typing import Dict, List, Any


class DatasetQualityScorer:
    """Evaluates and scores dataset quality across multiple dimensions."""

    def __init__(self):
        self.scores = {
            "content_quality": 0.0,
            "diversity": 0.0,
            "completeness": 0.0,
            "formatting": 0.0,
            "size": 0.0,
            "overall": 0.0,
        }

        self.weights = {
            "content_quality": 0.3,
            "diversity": 0.2,
            "completeness": 0.2,
            "formatting": 0.15,
            "size": 0.15,
        }

        self.recommendations: List[str] = []

    def score_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Score a dataset."""
        path = Path(dataset_path)

        if not path.exists():
            return {"error": "Dataset not found"}

        scores = {}

        # Size score
        scores["size"] = self._score_size(path)

        # Completeness
        scores["completeness"] = self._score_completeness(path)

        # Formatting
        scores["formatting"] = self._score_formatting(path)

        # Content quality
        scores["content_quality"] = self._score_content_quality(path)

        # Diversity
        scores["diversity"] = self._score_diversity(path)

        # Overall
        overall = sum(scores[k] * self.weights[k] for k in self.weights)
        scores["overall"] = overall

        return scores

    def _score_size(self, path: Path) -> float:
        """Score based on size."""
        files = list(path.rglob("*.bin")) + list(path.rglob("*.txt"))
        if not files:
            return 0.0

        total_size = sum(f.stat().st_size for f in files)

        if total_size < 1024 * 1024:  # < 1MB
            return 0.3
        elif total_size < 10 * 1024 * 1024:  # < 10MB
            return 0.6
        elif total_size < 100 * 1024 * 1024:  # < 100MB
            return 0.8
        else:
            return 1.0

    def _score_completeness(self, path: Path) -> float:
        """Score based on completeness."""
        required = ["input.txt", "meta.pkl"]
        train = path / "train.bin"

        has_meta = (path / "meta.pkl").exists()
        has_train = train.exists()

        if has_meta and has_train:
            return 1.0
        elif has_train:
            return 0.7
        else:
            return 0.3

    def _score_formatting(self, path: Path) -> float:
        """Score based on formatting."""
        txt_file = path / "input.txt"

        if not txt_file.exists():
            return 0.5

        content = txt_file.read_text()

        if len(content) < 100:
            return 0.3

        # Check for common issues
        if "\x00" in content:  # Null bytes
            return 0.2

        return 0.8

    def _score_content_quality(self, path: Path) -> float:
        """Score based on content quality."""
        txt_file = path / "input.txt"

        if not txt_file.exists():
            return 0.5

        content = txt_file.read_text()
        lines = content.split("\n")

        # Check for empty lines
        non_empty = sum(1 for l in lines if l.strip())

        if non_empty == 0:
            return 0.0

        quality = min(non_empty / len(lines), 1.0)

        return quality

    def _score_diversity(self, path: Path) -> float:
        """Score based on diversity."""
        txt_file = path / "input.txt"

        if not txt_file.exists():
            return 0.5

        content = txt_file.read_text()

        # Simple diversity check - unique characters
        unique_chars = len(set(content))
        total_chars = len(content)

        if total_chars == 0:
            return 0.0

        diversity = unique_chars / total_chars

        return min(diversity * 10, 1.0)

    def get_recommendations(self) -> List[str]:
        """Get recommendations for improvement."""
        return self.recommendations


__all__ = ["DatasetQualityScorer"]
