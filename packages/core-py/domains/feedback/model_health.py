"""
Model Health Monitor — periodic perplexity tracking and drift detection.

Runs the model on a fixed benchmark corpus, logs perplexity over time,
and alerts when significant drift is detected. Integrates with the
quality guard system used by auto-feedback training.

Usage:
    monitor = ModelHealthMonitor(model, tokenizer)
    monitor.run_benchmark()  # measure current PPL
    monitor.get_trend()      # return history
    drift = monitor.detect_drift()  # check for significant changes
"""

from __future__ import annotations

import json
import time
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger("slo.model_health")

BENCHMARK_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "artificial intelligence is transforming how we interact with computers",
    "the sun rose over the mountains casting golden light across the valley",
    "machine learning models learn patterns from data to make predictions",
    "she opened the old wooden door and stepped into the dimly lit room",
    "natural language processing enables computers to understand human speech",
    "the scientist conducted experiments to verify the hypothesis",
    "a neural network consists of layers of interconnected nodes",
    "the river flowed gently through the ancient forest",
    "democracy is a system of government where citizens exercise power",
    "the chef prepared a magnificent meal with fresh local ingredients",
    "deep learning uses multiple layers to progressively extract features",
    "the spaceship launched into orbit carrying supplies to the station",
    "education is the most powerful weapon to change the world",
    "the musician played a melody that filled the concert hall with emotion",
]


@dataclass
class HealthSnapshot:
    timestamp: float
    perplexity: float
    loss: float
    num_sentences: int


class ModelHealthMonitor:
    """Periodically benchmarks model perplexity and tracks health over time.

    Uses a fixed benchmark corpus (15 sentences covering diverse topics)
    to measure model quality. History is persisted to a JSON file so
    trends survive server restarts.
    """

    def __init__(self, db_path: str = "data/model_health.json"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._history: List[Dict[str, Any]] = []
        self._model = None
        self._tokenizer = None
        self._load_history()

    def set_model(self, model, tokenizer) -> None:
        """Set the model and tokenizer to benchmark."""
        self._model = model
        self._tokenizer = tokenizer

    def _load_history(self) -> None:
        """Load benchmark history from disk."""
        if self.db_path.exists():
            try:
                data = json.loads(self.db_path.read_text())
                self._history = data if isinstance(data, list) else []
            except Exception as e:
                logger.warning("Failed to load health history from %s, resetting: %s", self.db_path, e)
                self._history = []

    def _save_history(self) -> None:
        """Persist benchmark history to disk."""
        try:
            self.db_path.write_text(json.dumps(self._history[-200:], indent=2))
        except Exception as e:
            logger.warning("Failed to save health history: %s", e, extra={"tag": "INFRA"})

    def run_benchmark(self) -> Optional[HealthSnapshot]:
        """Run model on the benchmark corpus and compute average perplexity.

        Returns a HealthSnapshot with PPL, loss, and sentence count.
        """
        import numpy as np
        from domains.training.slonet import tensor, SloLSTM

        net = self._model
        tok = self._tokenizer
        if net is None or tok is None:
            return None

        try:
            lstm_layers = [l for l in net.layers if isinstance(l, SloLSTM)]
            if not lstm_layers:
                return None
            lstm = lstm_layers[0]

            total_log_probs = 0.0
            total_tokens = 0

            for sentence in BENCHMARK_CORPUS:
                input_ids = tok.encode(sentence)
                if len(input_ids) < 2:
                    continue

                for i in range(len(input_ids) - 1):
                    ctx = np.array([input_ids[: i + 1][-64:]], dtype=np.int64)
                    x = tensor(ctx, requires_grad=False)
                    h = lstm.init_hidden()
                    logits_t, _ = lstm.forward(x, h)
                    l = logits_t.data[0, -1]
                    l = np.where(np.isfinite(l), l, -1e9)
                    ps = np.exp(l - l.max())
                    ps = ps / (ps.sum() + 1e-10)
                    actual = input_ids[i + 1]
                    if actual < len(ps):
                        total_log_probs += np.log(ps[actual] + 1e-10)
                        total_tokens += 1

            if total_tokens == 0:
                return None

            avg_nll = -total_log_probs / total_tokens
            ppl = float(np.exp(avg_nll))

            snapshot = HealthSnapshot(
                timestamp=time.time(),
                perplexity=ppl,
                loss=float(avg_nll),
                num_sentences=len(BENCHMARK_CORPUS),
            )

            self._history.append({
                "timestamp": snapshot.timestamp,
                "perplexity": round(snapshot.perplexity, 4),
                "loss": round(snapshot.loss, 4),
                "num_sentences": snapshot.num_sentences,
            })
            self._save_history()

            return snapshot
        except Exception as e:
            logger.warning("Health benchmark failed: %s", e, extra={"tag": "INFRA"})
            return None

    def get_trend(self, window: int = 20) -> Dict[str, Any]:
        """Get health trend data for the last N measurements.

        Returns current PPL, best PPL, worst PPL, and a list of
        (timestamp, ppl) points for charting.
        """
        if not self._history:
            return {"available": False, "message": "No benchmarks yet"}

        recent = self._history[-window:]
        ppl_values = [h["perplexity"] for h in recent]

        return {
            "available": True,
            "current": ppl_values[-1],
            "best": min(ppl_values),
            "worst": max(ppl_values),
            "average": sum(ppl_values) / len(ppl_values),
            "count": len(self._history),
            "recent_count": len(recent),
            "points": [
                {"t": h["timestamp"], "ppl": h["perplexity"]}
                for h in recent
            ],
        }

    def detect_drift(self, threshold_pct: float = 10.0) -> Dict[str, Any]:
        """Detect significant perplexity drift compared to recent baseline.

        Compares the latest measurement against the median of the previous
        5 measurements. If PPL increased by more than ``threshold_pct``
        percent, flags as drift.

        Returns:
            Dict with ``drifted`` (bool), ``current_ppl``, ``baseline_ppl``,
            ``change_pct``, and ``message``.
        """
        if len(self._history) < 3:
            return {"drifted": False, "message": "Not enough data"}

        latest = self._history[-1]["perplexity"]
        baseline_values = [h["perplexity"] for h in self._history[-6:-1]]
        if not baseline_values:
            return {"drifted": False, "message": "Not enough data"}
        baseline_values.sort()
        baseline = baseline_values[len(baseline_values) // 2]

        change_pct = ((latest - baseline) / baseline) * 100
        drifted = change_pct > threshold_pct

        return {
            "drifted": drifted,
            "current_ppl": round(latest, 4),
            "baseline_ppl": round(baseline, 4),
            "change_pct": round(change_pct, 2),
            "message": (
                f"PPL increased {change_pct:+.1f}% (drift detected)"
                if drifted
                else f"PPL changed {change_pct:+.1f}% (stable)"
            ),
        }

    def start_auto_monitoring(self, interval_seconds: int = 300) -> threading.Thread:
        """Start a background thread that periodically benchmarks the model.

        Args:
            interval_seconds: Seconds between benchmarks (default: 5 min)

        Returns:
            The background thread (daemon).
        """
        def _loop():
            while True:
                self.run_benchmark()
                drift = self.detect_drift()
                if drift.get("drifted"):
                    logger.warning(drift["message"], extra={"tag": "INFRA"})
                time.sleep(interval_seconds)

        thread = threading.Thread(target=_loop, daemon=True, name="health-monitor")
        thread.start()
        return thread

    def get_stats(self) -> Dict[str, Any]:
        """Get full health statistics including trend and drift."""
        trend = self.get_trend()
        drift = self.detect_drift() if len(self._history) >= 3 else {
            "drifted": False, "message": "Not enough data"
        }
        return {
            "trend": trend,
            "drift": drift,
            "last_benchmark": self._history[-1] if self._history else None,
        }


_health_monitor: Optional[ModelHealthMonitor] = None


def get_health_monitor() -> ModelHealthMonitor:
    """Get or create the global model health monitor."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = ModelHealthMonitor()
    return _health_monitor
