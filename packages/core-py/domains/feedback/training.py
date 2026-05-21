"""
Training pipeline for feedback-based fine-tuning.

Uses collected feedback data to fine-tune the model using:
1. DPO (Direct Preference Optimization)
2. Supervised Fine-Tuning (SFT) on positive examples
3. RLHF-style reward modeling
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class TrainingExample:
    prompt: str
    response: str
    rating: str
    quality_score: Optional[float] = None


@dataclass
class DPOPair:
    chosen: str
    rejected: str
    prompt: str


class FeedbackTrainer:
    """
    Prepares training data from feedback for fine-tuning.

    Supports:
    - DPO format: chosen/rejected pairs
    - SFT format: supervised fine-tuning on positive examples
    - Reward training: preference pairs
    """

    def __init__(self, db_path: str = "data/feedback.db"):
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def get_training_examples(
        self, min_quality: float = 0.0, limit: int = 10000
    ) -> List[TrainingExample]:
        """Get training examples from feedback database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT m.content, f.rating, f.quality_score,
                   (SELECT mm.content FROM messages mm 
                    WHERE mm.conversation_id = m.conversation_id 
                    AND mm.role = 'user'
                    AND mm.created_at < m.created_at
                    ORDER BY mm.created_at DESC LIMIT 1) as prompt
            FROM messages m
            JOIN feedback f ON m.id = f.message_id
            WHERE m.role = 'assistant'
            AND (f.quality_score IS NULL OR f.quality_score >= ?)
            ORDER BY f.created_at DESC
            LIMIT ?
        """,
            (min_quality, limit),
        )

        rows = cursor.fetchall()
        conn.close()

        examples = []
        for row in rows:
            examples.append(
                TrainingExample(
                    prompt=row[3] or "", response=row[0], rating=row[1], quality_score=row[2]
                )
            )

        return examples

    def prepare_dpo_pairs(self, min_pairs: int = 10) -> List[DPOPair]:
        """
        Prepare DPO (Direct Preference Optimization) training pairs.

        Creates (chosen, rejected) pairs from feedback where:
        - chosen = response with thumbs_up
        - rejected = response with thumbs_down in same conversation
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT m.conversation_id 
            FROM feedback f
            JOIN messages m ON f.message_id = m.id
            GROUP BY m.conversation_id
            HAVING COUNT(DISTINCT f.rating) > 1
            LIMIT 1000
        """)

        conv_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        pairs = []
        for conv_id in conv_ids:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT m.content, f.rating,
                       (SELECT mm.content FROM messages mm 
                        WHERE mm.conversation_id = m.conversation_id 
                        AND mm.role = 'user'
                        AND mm.created_at < m.created_at
                        ORDER BY mm.created_at DESC LIMIT 1) as prompt
                FROM messages m
                JOIN feedback f ON m.id = f.message_id
                WHERE m.conversation_id = ? AND m.role = 'assistant'
            """,
                (conv_id,),
            )

            rows = cursor.fetchall()
            conn.close()

            chosen = None
            rejected = None
            prompt = ""

            for content, rating, pr in rows:
                if rating == "thumbs_up" and chosen is None:
                    chosen = content
                    prompt = pr or ""
                elif rating == "thumbs_down" and rejected is None:
                    rejected = content

            if chosen and rejected:
                pairs.append(DPOPair(chosen=chosen, rejected=rejected, prompt=prompt))

        return pairs

    def prepare_sft_data(self, min_quality: float = 0.5) -> List[Dict]:
        """
        Prepare Supervised Fine-Tuning data from positive feedback.

        Format: [{"prompt": "...", "response": "..."}]
        """
        examples = self.get_training_examples(min_quality=min_quality)

        sft_data = []
        for ex in examples:
            if ex.rating == "thumbs_up" and ex.prompt:
                sft_data.append(
                    {
                        "prompt": ex.prompt,
                        "response": ex.response,
                        "quality_score": ex.quality_score or 1.0,
                    }
                )

        return sft_data

    def export_for_alignment(
        self, output_dir: str = "data/training", formats: List[str] = ["dpo", "sft"]
    ) -> Dict[str, str]:
        """
        Export training data in various formats.

        Args:
            output_dir: Directory to save training files
            formats: List of formats to export ["dpo", "sft", "reward"]

        Returns:
            Dict mapping format name to output file path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}

        if "dpo" in formats:
            dpo_path = output_path / "dpo_training.jsonl"
            pairs = self.prepare_dpo_pairs()
            with open(dpo_path, "w") as f:
                for pair in pairs:
                    f.write(
                        json.dumps(
                            {
                                "chosen": pair.chosen,
                                "rejected": pair.rejected,
                                "prompt": pair.prompt,
                            }
                        )
                        + "\n"
                    )
            results["dpo"] = str(dpo_path)
            print(f"Exported {len(pairs)} DPO pairs to {dpo_path}")

        if "sft" in formats:
            sft_path = output_path / "sft_training.jsonl"
            sft_data = self.prepare_sft_data()
            with open(sft_path, "w") as f:
                for item in sft_data:
                    f.write(json.dumps(item) + "\n")
            results["sft"] = str(sft_path)
            print(f"Exported {len(sft_data)} SFT examples to {sft_path}")

        if "reward" in formats:
            reward_path = output_path / "reward_training.jsonl"
            examples = self.get_training_examples()

            # Format: [prompt, response, reward_score]
            # reward_score = 1 for thumbs_up, 0 for thumbs_down
            with open(reward_path, "w") as f:
                for ex in examples:
                    if ex.prompt:
                        reward = 1.0 if ex.rating == "thumbs_up" else 0.0
                        f.write(
                            json.dumps(
                                {"prompt": ex.prompt, "response": ex.response, "reward": reward}
                            )
                            + "\n"
                        )
            results["reward"] = str(reward_path)
            print(f"Exported {len(examples)} reward examples to {reward_path}")

        return results

    def get_training_stats(self) -> Dict:
        """Get statistics about available training data."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT rating, COUNT(*) as count 
            FROM feedback f
            JOIN messages m ON f.message_id = m.id
            WHERE m.role = 'assistant'
            GROUP BY rating
        """)
        rating_counts = dict(cursor.fetchall())

        cursor.execute("""
            SELECT COUNT(DISTINCT conversation_id)
            FROM messages
        """)
        conv_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM messages WHERE role = 'assistant'")
        total_responses = cursor.fetchone()[0]

        conn.close()

        pairs = self.prepare_dpo_pairs()
        sft_examples = self.prepare_sft_data()

        return {
            "total_conversations": conv_count,
            "total_responses": total_responses,
            "thumbs_up": rating_counts.get("thumbs_up", 0),
            "thumbs_down": rating_counts.get("thumbs_down", 0),
            "available_dpo_pairs": len(pairs),
            "available_sft_examples": len(sft_examples),
        }

    def export_dpo(self, filepath: str) -> int:
        """Export DPO pairs to JSONL file. Returns count."""
        pairs = self.prepare_dpo_pairs()
        count = 0
        with open(filepath, "w") as f:
            for pair in pairs:
                f.write(
                    json.dumps(
                        {
                            "chosen": pair.chosen,
                            "rejected": pair.rejected,
                            "prompt": pair.prompt,
                        }
                    )
                    + "\n"
                )
                count += 1
        return count

    def export_sft(self, filepath: str) -> int:
        """Export SFT examples to JSONL file. Returns count."""
        data = self.prepare_sft_data()
        count = 0
        with open(filepath, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
                count += 1
        return count


def create_training_pipeline(db_path: str = "data/feedback.db") -> FeedbackTrainer:
    """Factory function to create a feedback trainer."""
    return FeedbackTrainer(db_path)


if __name__ == "__main__":
    trainer = create_training_pipeline()

    print("=" * 60)
    print("Feedback Training Pipeline")
    print("=" * 60)

    stats = trainer.get_training_stats()
    print("\nAvailable Training Data:")
    print(f"  Total conversations: {stats['total_conversations']}")
    print(f"  Total responses: {stats['total_responses']}")
    print(f"  Thumbs up: {stats['thumbs_up']}")
    print(f"  Thumbs down: {stats['thumbs_down']}")
    print(f"  DPO pairs available: {stats['available_dpo_pairs']}")
    print(f"  SFT examples available: {stats['available_sft_examples']}")

    if stats["available_dpo_pairs"] >= 10:
        print("\nExporting training data...")
        results = trainer.export_for_alignment()
        print("\nExported files:")
        for fmt, path in results.items():
            print(f"  {fmt}: {path}")
