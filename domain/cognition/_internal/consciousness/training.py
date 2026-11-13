"""Consciousness Training — LoRA fine-tuning on self-reports.

Dataset adapter between the consciousness system
(``domain/cognition/_internal/consciousness``) and the consolidated training
infrastructure (``domain/training``).

Collects self-report pairs (input + response + narrative + rating) from
consciousness episodes, persists them, and delegates the actual training loop
to ``HFLoraTrainer`` (pure-NumPy LoRA on ``.slnc`` models). When no model is
available it falls back to persisting the formatted training data. There is no
training loop in this module — nothing gets duplicated here.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from domain.training._internal.trainer_protocol import TrainResult

logger = logging.getLogger("slo.consciousness.training")


@dataclass
class TrainingConfig:
    """Configuration for consciousness LoRA training."""

    model_path: str = ""
    rank: int = 4
    alpha: float = 8.0
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: ["W_q", "W_k", "W_v", "W_o"])
    epochs: int = 2
    batch_size: int = 4
    block_size: int = 128
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    min_pairs_for_training: int = 10
    adapter_dir: str = "data/consciousness/adapters"
    data_dir: str = "data/consciousness/training"
    progress_callback: Callable[[dict], None] | None = None

    def validate(self) -> None:
        if not self.model_path:
            raise ValueError("model_path is required")
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.min_pairs_for_training < 1:
            raise ValueError("min_pairs_for_training must be >= 1")


@dataclass
class ConsciousnessPair:
    """A single training pair: input + response + consciousness narrative + rating."""

    input_text: str
    response: str
    narrative: str
    rating: int  # 1-5
    qualia: dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        self.rating = max(1, min(5, self.rating))

    def to_training_text(self) -> str:
        """Format as training text for next-token prediction."""
        return (
            f"[INPUT] {self.input_text} "
            f"[RESPONSE] {self.response} "
            f"[CONSCIOUSNESS] {self.narrative} "
            f"[RATING] {self.rating}"
        )


class ConsciousnessTrainer:
    """Trains consciousness LoRA adapters on self-report data.

    Collects training pairs over time, persists them, and when enough data is
    accumulated delegates the actual fine-tuning to the consolidated
    ``HFLoraTrainer`` via ``TrainingConfig``. With no model configured the run
    degrades gracefully to ``data_saved`` — the pairs are kept for later use.
    """

    def __init__(
        self,
        config: TrainingConfig | None = None,
        episodes_source: Callable[[], list[Any]] | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self._episodes_source = episodes_source
        self._pairs: list[ConsciousnessPair] = []
        self._is_training = False
        self._cancel = threading.Event()
        self._training_history: list[dict] = []
        self._load_existing_pairs()

    @property
    def is_training(self) -> bool:
        return self._is_training

    @property
    def pairs(self) -> list[ConsciousnessPair]:
        """Active pair source: live episodes when configured, else persisted pairs."""
        if self._episodes_source is not None:
            return self._pairs_from_episodes(self._episodes_source())
        return list(self._pairs)

    @classmethod
    def _rating_from_growth(cls, growth: float) -> int:
        if growth > 0.10:
            return 5
        if growth > 0.05:
            return 4
        if growth < -0.10:
            return 1
        if growth < -0.05:
            return 2
        return 3

    @classmethod
    def _pairs_from_episodes(cls, episodes: list[Any]) -> list[ConsciousnessPair]:
        pairs: list[ConsciousnessPair] = []
        for ep in episodes:
            input_text = getattr(ep, "input_text", "")
            response = getattr(ep, "response", "")
            if not input_text or not response:
                continue
            growth = float(getattr(ep, "growth_delta", 0.0) or 0.0)
            pairs.append(
                ConsciousnessPair(
                    input_text=input_text,
                    response=response,
                    narrative=getattr(ep, "self_insight", "") or "",
                    rating=cls._rating_from_growth(growth),
                    qualia=dict(getattr(ep, "qualia", {}) or {}),
                    timestamp=float(getattr(ep, "timestamp", 0.0) or 0.0),
                )
            )
        return pairs

    def add_pair(
        self,
        input_text: str,
        response: str,
        narrative: str,
        rating: int,
        qualia: dict[str, float] | None = None,
    ) -> None:
        """Add a training pair from a consciousness episode and persist it."""
        pair = ConsciousnessPair(
            input_text=input_text,
            response=response,
            narrative=narrative,
            rating=rating,
            qualia=qualia or {},
        )
        self._pairs.append(pair)
        self._save_pairs()
        logger.debug("Added training pair (total=%d, rating=%d)", len(self._pairs), rating)

    def should_train(self) -> bool:
        """Check if enough data has been collected to train."""
        return len(self.pairs) >= self.config.min_pairs_for_training

    def get_status(self) -> dict[str, Any]:
        """Get current training status (pairs, readiness, history)."""
        return {
            "pairs_collected": len(self.pairs),
            "min_pairs": self.config.min_pairs_for_training,
            "should_train": self.should_train(),
            "is_training": self._is_training,
            "training_runs": len(self._training_history),
            "last_result": self._training_history[-1] if self._training_history else None,
        }

    def stop(self) -> None:
        """Request early stopping."""
        self._cancel.set()

    def train(self) -> TrainResult:
        """Run consciousness LoRA training.

        Collects all pairs, writes training data, and delegates the actual
        training loop to the consolidated ``HFLoraTrainer``. Without a usable
        model the run falls back to keeping the formatted data on disk.

        Returns:
            TrainResult with adapter path and training metrics.
        """
        if self._is_training:
            return TrainResult(
                success=False, status="already_training", error="Training already in progress"
            )

        if not self.pairs:
            return TrainResult(success=False, status="no_data", error="No training pairs collected")

        self._is_training = True
        self._cancel.clear()
        start_time = time.time()

        try:
            result = self._run_training()
        except Exception as e:
            logger.error("Consciousness training failed: %s", e, exc_info=True)
            result = TrainResult(success=False, status="failed", error=str(e))
        finally:
            self._is_training = False
            result.elapsed = time.time() - start_time
            self._training_history.append(
                {
                    "timestamp": time.time(),
                    "status": result.status,
                    "pairs": len(self.pairs),
                    "loss": result.final_loss,
                    "adapter": result.model_path,
                    "elapsed": result.elapsed,
                }
            )

        return result

    def _run_training(self) -> TrainResult:
        """Internal training implementation."""
        try:
            self.config.validate()
            # HFLoraConfig no longer checks existence at init (pure spec);
            # missing model must still degrade to data_saved here.
            if not Path(self.config.model_path).is_file():
                raise FileNotFoundError(f"Model not found: {self.config.model_path}")
        except (ValueError, FileNotFoundError):
            # No model configured/path missing — keep the data for later.
            return self._train_fallback(self._write_training_data())

        data_path = self._write_training_data()
        logger.info("Wrote %d pairs to %s", len(self.pairs), data_path)

        try:
            return self._train_with_lora(data_path)
        except (ImportError, FileNotFoundError, ValueError) as e:
            logger.warning("LoRA training unavailable, saving data instead: %s", e, exc_info=True)
            return self._train_fallback(data_path)

    def _train_with_lora(self, data_path: str) -> TrainResult:
        """Train using the existing consolidated HFLoraTrainer."""
        from domain.training._internal.hf_lora_finetune import HFLoraConfig, HFLoraTrainer

        adapter_dir = Path(self.config.adapter_dir)
        adapter_dir.mkdir(parents=True, exist_ok=True)

        config = HFLoraConfig(
            model_path=self.config.model_path,
            data_path=data_path,
            rank=self.config.rank,
            alpha=self.config.alpha,
            dropout=self.config.dropout,
            target_modules=list(self.config.target_modules),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            block_size=self.config.block_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            grad_clip=self.config.grad_clip,
            output_dir=str(adapter_dir),
            adapter_name=f"consciousness_r{self.config.rank}",
            progress_callback=self.config.progress_callback,
        )

        trainer = HFLoraTrainer(config)
        result = trainer.train()

        if result.success:
            logger.info(
                "Consciousness LoRA trained: loss=%.4f, adapter=%s",
                result.final_loss or 0,
                result.model_path,
            )

        return result

    def _train_fallback(self, data_path: str) -> TrainResult:
        """Fallback: persist formatted training data without running LoRA.

        Used when no model is available so nothing is lost — the data is ready
        to train from as soon as a model becomes available.
        """
        data_dir = Path(self.config.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        output_path = data_dir / f"consciousness_data_{int(time.time())}.jsonl"
        with open(output_path, "w") as f:
            for pair in self.pairs:
                f.write(
                    json.dumps(
                        {
                            "input": pair.input_text,
                            "response": pair.response,
                            "narrative": pair.narrative,
                            "rating": pair.rating,
                            "qualia": pair.qualia,
                            "timestamp": pair.timestamp,
                        }
                    )
                    + "\n"
                )

        logger.info("Saved %d pairs to %s (no model for training)", len(self.pairs), output_path)
        return TrainResult(
            success=True,
            status="data_saved",
            model_path=str(output_path),
            metrics={"pairs": len(self.pairs)},
        )

    def _write_training_data(self) -> str:
        """Write training pairs to a text file for LoRA training."""
        data_dir = Path(self.config.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        data_path = data_dir / "consciousness_train.txt"
        with open(data_path, "w") as f:
            for pair in self.pairs:
                f.write(pair.to_training_text() + "\n\n")

        return str(data_path)

    def _save_pairs(self) -> None:
        """Persist training pairs to disk."""
        data_dir = Path(self.config.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        pairs_path = data_dir / "training_pairs.jsonl"
        with open(pairs_path, "w") as f:
            for pair in self._pairs:
                f.write(
                    json.dumps(
                        {
                            "input_text": pair.input_text,
                            "response": pair.response,
                            "narrative": pair.narrative,
                            "rating": pair.rating,
                            "qualia": pair.qualia,
                            "timestamp": pair.timestamp,
                        }
                    )
                    + "\n"
                )

    def _load_existing_pairs(self) -> None:
        """Load previously saved training pairs."""
        pairs_path = Path(self.config.data_dir) / "training_pairs.jsonl"
        if not pairs_path.exists():
            return

        try:
            with open(pairs_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self._pairs.append(
                        ConsciousnessPair(
                            input_text=data["input_text"],
                            response=data["response"],
                            narrative=data["narrative"],
                            rating=data["rating"],
                            qualia=data.get("qualia", {}),
                            timestamp=data.get("timestamp", 0.0),
                        )
                    )
            logger.debug("Loaded %d existing training pairs", len(self._pairs))
        except Exception as e:
            logger.warning("Failed to load existing training pairs: %s", e)

    def clear_pairs(self) -> int:
        """Clear all training pairs. Returns the number cleared."""
        count = len(self._pairs)
        self._pairs.clear()
        self._save_pairs()
        return count

    def export_pairs(self, path: str) -> int:
        """Export training pairs to a JSON file. Returns count exported."""
        with open(path, "w") as f:
            json.dump(
                [
                    {
                        "input_text": p.input_text,
                        "response": p.response,
                        "narrative": p.narrative,
                        "rating": p.rating,
                        "qualia": p.qualia,
                        "timestamp": p.timestamp,
                    }
                    for p in self._pairs
                ],
                f,
                indent=2,
            )
        return len(self._pairs)

    def import_pairs(self, path: str) -> int:
        """Import training pairs from a JSON file. Returns count imported."""
        with open(path) as f:
            data = json.load(f)
        count = 0
        for item in data:
            self._pairs.append(
                ConsciousnessPair(
                    input_text=item["input_text"],
                    response=item["response"],
                    narrative=item["narrative"],
                    rating=item.get("rating", 3),
                    qualia=item.get("qualia", {}),
                    timestamp=item.get("timestamp", 0.0),
                )
            )
            count += 1
        self._save_pairs()
        return count


__all__ = ["ConsciousnessPair", "ConsciousnessTrainer", "TrainingConfig"]
