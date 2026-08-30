"""
Training pipeline for feedback-based fine-tuning.

Uses collected feedback data to fine-tune the model using:
1. DPO (Direct Preference Optimization)
2. Supervised Fine-Tuning (SFT) on positive examples
3. RLHF-style reward modeling
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from mogdb import MogDB

logger = logging.getLogger("slo.feedback.training")


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
        legacy = Path(db_path)
        if legacy.is_file():
            # Migrate a legacy SQLite feedback database into MogDB.
            from domains.feedback.database import FeedbackDB

            FeedbackDB(db_path=str(legacy))
        self._db = MogDB(db_path)
        self._messages = self._db.collection("messages")
        self._feedback = self._db.collection("feedback")

    def _message_by_id(self, message_id: str) -> Optional[Dict]:
        """Return the message document with ``message_id``, or ``None``."""
        return self._messages.find_one({"_id": message_id})

    def _latest_prior_user_message(self, conversation_id: str, created_at: str) -> str:
        """Return the content of the most recent user message before ``created_at``.

        Mirrors the legacy SQL subquery: the newest user message in the same
        conversation with ``created_at < message.created_at``.
        """
        docs = self._messages.find(
            {
                "conversation_id": conversation_id,
                "role": "user",
                "created_at": {"$lt": created_at},
            },
            sort=[("created_at", -1)],
            limit=1,
        )
        return docs[0]["content"] if docs else ""

    def _prompt_for(self, message: Dict) -> str:
        """Prompt (latest prior user message) for an assistant message."""
        return self._latest_prior_user_message(
            message.get("conversation_id", ""), message.get("created_at", "")
        )

    def get_training_examples(
        self, min_quality: float = 0.0, limit: int = 10000
    ) -> List[TrainingExample]:
        """Get training examples from the feedback database.

        Joins feedback rows against their assistant messages (in Python —
        MogDB has no SQL joins) and resolves each prompt as the latest prior
        user message in the same conversation.

        Args:
            min_quality: Minimum quality score; ``None`` scores pass always.
            limit: Maximum number of examples to return.

        Returns:
            List of :class:`TrainingExample`, newest feedback first.

        Side effects:
            - none (read-only)
        """
        examples: List[TrainingExample] = []
        dated: List[tuple] = []
        for fb in self._feedback.find():
            if fb.get("rating") is None:
                continue
            quality = fb.get("quality_score")
            if quality is not None and quality < min_quality:
                continue
            message = self._message_by_id(fb.get("message_id") or "")
            if message is None or message.get("role") != "assistant":
                continue
            dated.append(
                (
                    fb.get("created_at", ""),
                    TrainingExample(
                        prompt=self._prompt_for(message),
                        response=message.get("content", ""),
                        rating=fb.get("rating", ""),
                        quality_score=quality,
                    ),
                )
            )

        dated.sort(key=lambda item: item[0], reverse=True)
        return [ex for _, ex in dated[:limit]]

    def prepare_dpo_pairs(self, min_pairs: int = 10) -> List[DPOPair]:
        """
        Prepare DPO (Direct Preference Optimization) training pairs.

        Creates (chosen, rejected) pairs from feedback where:
        - chosen = response with thumbs_up
        - rejected = response with thumbs_down in same conversation
        """
        # Group assistant messages-with-feedback by conversation, in
        # feedback order, to keep the first thumbs_up/thumbs_down semantics
        # of the legacy query.
        conv_groups: Dict[str, List[Dict]] = {}
        fb_by_message: Dict[str, Dict] = {}
        for fb in self._feedback.find():
            fb_by_message[fb.get("message_id") or ""] = fb

        for message in self._messages.find():
            if message.get("role") != "assistant":
                continue
            fb = fb_by_message.get(message.get("id") or message.get("_id") or "")
            if fb is None:
                continue
            conv_groups.setdefault(message.get("conversation_id", ""), []).append(
                {"message": message, "feedback": fb}
            )

        pairs: List[DPOPair] = []
        for conv_id in sorted(conv_groups)[:1000]:
            entries = conv_groups[conv_id]
            ratings = {e["feedback"].get("rating") for e in entries if e["feedback"].get("rating")}
            if len(ratings) <= 1:
                continue

            chosen = None
            rejected = None
            prompt = ""
            for entry in entries:
                rating = entry["feedback"].get("rating")
                if rating == "thumbs_up" and chosen is None:
                    chosen = entry["message"].get("content", "")
                    prompt = self._prompt_for(entry["message"])
                elif rating == "thumbs_down" and rejected is None:
                    rejected = entry["message"].get("content", "")

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
            logger.info("Exported %d DPO pairs to %s", len(pairs), dpo_path, extra={"tag": "INFRA"})

        if "sft" in formats:
            sft_path = output_path / "sft_training.jsonl"
            sft_data = self.prepare_sft_data()
            with open(sft_path, "w") as f:
                for item in sft_data:
                    f.write(json.dumps(item) + "\n")
            results["sft"] = str(sft_path)
            logger.info("Exported %d SFT examples to %s", len(sft_data), sft_path, extra={"tag": "INFRA"})

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
            logger.info("Exported %d reward examples to %s", len(examples), reward_path, extra={"tag": "INFRA"})

        return results

    def get_training_stats(self) -> Dict:
        """Get statistics about available training data."""
        rating_counts: Dict[str, int] = {}
        conversations: set = set()
        total_responses = 0

        for fb in self._feedback.find():
            message = self._message_by_id(fb.get("message_id") or "")
            if message is None or message.get("role") != "assistant":
                continue
            rating = fb.get("rating") or ""
            rating_counts[rating] = rating_counts.get(rating, 0) + 1

        for message in self._messages.find():
            conversations.add(message.get("conversation_id", ""))
            if message.get("role") == "assistant":
                total_responses += 1

        pairs = self.prepare_dpo_pairs()
        sft_examples = self.prepare_sft_data()

        return {
            "total_conversations": len(conversations),
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

    logger.info("=" * 60)
    logger.info("Feedback Training Pipeline")
    logger.info("=" * 60)

    stats = trainer.get_training_stats()
    logger.info("Available Training Data:")
    logger.info("  Total conversations: %d", stats['total_conversations'])
    logger.info("  Total responses: %d", stats['total_responses'])
    logger.info("  Thumbs up: %d", stats['thumbs_up'])
    logger.info("  Thumbs down: %d", stats['thumbs_down'])
    logger.info("  DPO pairs available: %d", stats['available_dpo_pairs'])
    logger.info("  SFT examples available: %d", stats['available_sft_examples'])

    if stats["available_dpo_pairs"] >= 10:
        logger.info("Exporting training data...")
        results = trainer.export_for_alignment()
        logger.info("Exported files:")
        for fmt, path in results.items():
            logger.info("  %s: %s", fmt, path)
