"""
Auto Trainer

Background thread that monitors new conversations and triggers
training when a threshold is reached. Eliminates the mobile→server
round-trip by training directly from server inference logs.

Env vars:
    SLO_AUTO_TRAIN=1         Enable auto-training (default: 0)
    SLO_AUTO_TRAIN_THRESHOLD  Conversations before trigger (default: 10)
    SLO_AUTO_TRAIN_INTERVAL   Min seconds between trains (default: 300)
"""

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from domains.shared import find_repo_root

logger = logging.getLogger("slo.training.auto_trainer")

_REPO_ROOT = find_repo_root(Path(__file__).resolve())
_SESSIONS_DIR = _REPO_ROOT / "data" / "chat_sessions"
_RESPONSE_LOGS_DIR = _REPO_ROOT / "data" / "response_logs"
_CAPTURED_CORPUS = _REPO_ROOT / "data" / "api_conversations" / "corpus.jsonl"


class AutoTrainer:
    """
    Background trainer that monitors inference logs and triggers
    training when enough new conversations accumulate.

    Attributes:
        threshold: Number of new conversations before triggering a train.
        interval_s: Minimum seconds between training runs.
    """

    def __init__(
        self,
        threshold: int = 10,
        interval_s: int = 300,
    ):
        self.threshold = threshold
        self.interval_s = interval_s
        self._last_train_ts: float = 0
        self._last_train_loss: float = 0
        self._last_train_checkpoint: str = ""
        self._conversation_count: int = 0
        self._total_trains: int = 0
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._sessions_mtime: float = 0
        self._logs_mtime: float = 0
        self._corpus_mtime: float = 0

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="auto-trainer"
        )
        self._thread.start()
        logger.info(
            "AutoTrainer started (threshold=%d, interval=%ds)",
            self.threshold, self.interval_s,
            extra={"tag": "TRAIN"},
        )

    def stop(self) -> None:
        """Stop the background monitoring thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("AutoTrainer stopped",
            extra={"tag": "TRAIN"},)

    def _loop(self) -> None:
        """Main monitoring loop — polls every 30s."""
        while not self._stop_event.is_set():
            try:
                self._check_and_train()
            except Exception as e:
                logger.error("AutoTrainer error: %s", e, exc_info=True,
                    extra={"tag": "TRAIN"},)
            self._stop_event.wait(30)

    def _check_and_train(self) -> None:
        """Check if new conversations exist and train if threshold reached."""
        # Check for new files
        sessions_mtime = self._dir_mtime(_SESSIONS_DIR)
        logs_mtime = self._dir_mtime(_RESPONSE_LOGS_DIR)
        corpus_mtime = self._dir_mtime(_CAPTURED_CORPUS)

        if (sessions_mtime == self._sessions_mtime
                and logs_mtime == self._logs_mtime
                and corpus_mtime == self._corpus_mtime):
            return

        # New data detected — count new conversations
        self._sessions_mtime = sessions_mtime
        self._logs_mtime = logs_mtime
        self._corpus_mtime = corpus_mtime
        self._conversation_count += 1

        # Check if we should train
        now = time.time()
        elapsed = now - self._last_train_ts

        if self._conversation_count >= self.threshold and elapsed >= self.interval_s:
            self._do_train()

    def _do_train(self) -> bool:
        """Extract pairs and spawn training subprocess."""
        from domains.training.pair_extractor import (
            extract_pairs_from_sessions,
            extract_pairs_from_corpus,
            extract_pairs_from_logs,
            write_training_text,
        )

        # Prefer session pairs, fall back to captured corpus, then logs
        pairs = extract_pairs_from_sessions(limit=self.threshold * 3, min_length=5)
        if len(pairs) < 5:
            pairs = extract_pairs_from_corpus(limit=self.threshold * 3, min_length=5)
        if len(pairs) < 5:
            pairs = extract_pairs_from_logs(limit=self.threshold * 3, min_length=5)

        if len(pairs) < 5:
            logger.info(
                "AutoTrainer: only %d pairs found (need 5), skipping",
                len(pairs),
                extra={"tag": "TRAIN"},
            )
            self._conversation_count = 0
            return False

        logger.info("AutoTrainer: found %d pairs, starting training", len(pairs),
            extra={"tag": "TRAIN"},)

        # Write text file
        text_file = write_training_text(pairs)

        # Spawn subprocess
        checkpoint_name = f"auto_{int(time.time())}"
        output_dir = _REPO_ROOT / "models" / "auto-training" / checkpoint_name
        output_dir.mkdir(parents=True, exist_ok=True)

        venv_python = _REPO_ROOT / ".venv" / "bin" / "python3"
        train_script = _REPO_ROOT / "scripts" / "hf_train.py"

        if not venv_python.exists():
            logger.error("AutoTrainer: .venv Python not found",
                extra={"tag": "TRAIN"},)
            return False

        t0 = time.time()
        try:
            proc = subprocess.run(
                [
                    str(venv_python),
                    str(train_script),
                    "--data", str(text_file),
                    "--output", str(output_dir),
                    "--model", "gpt2",
                    "--epochs", "1",
                    "--batch-size", "2",
                    "--lr", "5e-5",
                    "--max-seq-length", "256",
                    "--use-lora",
                    "--lora-rank", "8",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if proc.returncode != 0:
                logger.error("AutoTrainer subprocess failed: %s", proc.stderr[-500:],
                    extra={"tag": "TRAIN"},)
                return False

            try:
                result = json.loads(proc.stdout.strip().split("\n")[-1])
            except (json.JSONDecodeError, IndexError) as e:
                logger.error("AutoTrainer: invalid JSON from subprocess", extra={
                    "tag": "TRAIN", "error": str(e),
                    "stdout_tail": proc.stdout[-500:] if proc.stdout else "",
                })
                return False
            elapsed = time.time() - t0

            if result.get("success"):
                self._last_train_ts = time.time()
                self._last_train_loss = result.get("loss", 0)
                self._last_train_checkpoint = checkpoint_name
                self._total_trains += 1
                self._conversation_count = 0

                logger.info(
                    "AutoTrainer: training complete (loss=%.4f, steps=%d, %.1fs)",
                    result.get("loss", 0),
                    result.get("steps", 0),
                    elapsed,
                    extra={"tag": "TRAIN"},
                )

                # Store pairs in MogDB (with quality scoring)
                try:
                    from domains.training.mobile_training_store import get_training_store
                    from domains.training.quality_scorer import score_batch
                    quality_scores = score_batch(pairs)
                    store = get_training_store()
                    store.add_batch([
                        {
                            "user_msg": p["user_msg"],
                            "assistant_msg": p["assistant_msg"],
                            "session_id": p.get("session_id", ""),
                            "quality": quality_scores[i] if i < len(quality_scores) else 0,
                        }
                        for i, p in enumerate(pairs)
                    ])
                except Exception as e:
                    logger.warning("AutoTrainer: failed to store pairs in MogDB: %s", e,
                        extra={"tag": "TRAIN"},)

                return True
            else:
                logger.error("AutoTrainer training failed: %s", result.get("error"),
                    extra={"tag": "TRAIN"},)
                return False

        except subprocess.TimeoutExpired:
            logger.error("AutoTrainer: training timed out (300s)",
                extra={"tag": "TRAIN"},)
            return False
        except Exception as e:
            logger.error("AutoTrainer: training error: %s", e,
                extra={"tag": "TRAIN"},)
            return False

    @staticmethod
    def _dir_mtime(d: Path) -> float:
        """Get latest mtime of a directory's files (or a single file's mtime)."""
        if not d.exists():
            return 0
        if d.is_file():
            return d.stat().st_mtime
        latest = 0.0
        for f in d.iterdir():
            if f.is_file():
                latest = max(latest, f.stat().st_mtime)
        return latest

    def status(self) -> Dict[str, Any]:
        """Return current auto-trainer status."""
        session_count = len(list(_SESSIONS_DIR.glob("*.json"))) if _SESSIONS_DIR.exists() else 0
        log_count = len(list(_RESPONSE_LOGS_DIR.glob("*.jsonl"))) if _RESPONSE_LOGS_DIR.exists() else 0
        from domains.training.pair_extractor import count_pairs_in_corpus
        return {
            "enabled": self._thread is not None and self._thread.is_alive(),
            "threshold": self.threshold,
            "interval_s": self.interval_s,
            "pending_conversations": self._conversation_count,
            "total_trains": self._total_trains,
            "last_train": (
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._last_train_ts))
                if self._last_train_ts > 0
                else None
            ),
            "last_loss": self._last_train_loss,
            "last_checkpoint": self._last_train_checkpoint,
            "session_count": session_count,
            "response_log_count": log_count,
            "captured_count": count_pairs_in_corpus(),
        }


# Global singleton
_auto_trainer: Optional[AutoTrainer] = None


def get_auto_trainer() -> AutoTrainer:
    """Get or create the global AutoTrainer singleton."""
    global _auto_trainer
    if _auto_trainer is None:
        _auto_trainer = AutoTrainer(
            threshold=int(os.environ.get("SLO_AUTO_TRAIN_THRESHOLD", "10")),
            interval_s=int(os.environ.get("SLO_AUTO_TRAIN_INTERVAL", "300")),
        )
    return _auto_trainer


def start_auto_trainer_if_enabled() -> Optional[AutoTrainer]:
    """Start auto-trainer if SLO_AUTO_TRAIN=1. Returns the trainer or None."""
    if os.environ.get("SLO_AUTO_TRAIN", "0") != "1":
        return None
    trainer = get_auto_trainer()
    trainer.start()
    return trainer


def stop_auto_trainer() -> None:
    """Stop the global auto-trainer if running."""
    if _auto_trainer:
        _auto_trainer.stop()


__all__ = [
    "AutoTrainer",
    "get_auto_trainer",
    "start_auto_trainer_if_enabled",
    "stop_auto_trainer",
]
