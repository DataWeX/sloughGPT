"""Training monitoring — alerts for failures, convergence issues, and resource exhaustion.

Usage:
    from infrastructure.training_monitor import get_training_monitor, TrainingAlert

    monitor = get_training_monitor()
    monitor.record_loss(epoch=1, loss=2.3)
    monitor.record_metric("accuracy", 0.85)
    alerts = monitor.get_alerts()
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of training alerts."""

    LOSS_DIVERGENCE = "loss_divergence"
    LOSS_STAGNATION = "loss_stagnation"
    GRADIENT_EXPLOSION = "gradient_explosion"
    GRADIENT_VANISHING = "gradient_vanishing"
    HIGH_LOSS = "high_loss"
    LOW_LEARNING_RATE = "low_learning_rate"
    TRAINING_SLOW = "training_slow"
    MEMORY_WARNING = "memory_warning"
    GPU_MEMORY_WARNING = "gpu_memory_warning"
    TRAINING_FAILED = "training_failed"
    CONVERGENCE_ISSUE = "convergence_issue"
    EPOCH_TIMEOUT = "epoch_timeout"


@dataclass
class TrainingAlert:
    """A single training alert."""

    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: float = field(default_factory=time.time)
    epoch: int | None = None
    step: int | None = None
    value: float | None = None
    threshold: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "epoch": self.epoch,
            "step": self.step,
            "value": self.value,
            "threshold": self.threshold,
            "metadata": self.metadata,
        }


@dataclass
class TrainingMetrics:
    """Collected training metrics for monitoring."""

    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    tokens_per_second: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    cpu_percent: float = 0.0
    elapsed_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "tokens_per_second": self.tokens_per_second,
            "gpu_memory_used": self.gpu_memory_used,
            "gpu_memory_total": self.gpu_memory_total,
            "cpu_percent": self.cpu_percent,
            "elapsed_seconds": self.elapsed_seconds,
            "timestamp": self.timestamp,
        }


class TrainingMonitor:
    """Monitors training and generates alerts for issues."""

    def __init__(self, max_history: int = 1000) -> None:
        self._lock = threading.Lock()
        self._loss_history: deque[float] = deque(maxlen=max_history)
        self._lr_history: deque[float] = deque(maxlen=max_history)
        self._gradient_history: deque[float] = deque(maxlen=max_history)
        self._alerts: list[TrainingAlert] = []
        self._metrics: list[TrainingMetrics] = []
        self._max_alerts = 100

        # Thresholds (configurable)
        self.loss_divergence_threshold: float = 10.0
        self.loss_stagnation_window: int = 50
        self.loss_stagnation_threshold: float = 0.001
        self.gradient_explosion_threshold: float = 100.0
        self.gradient_vanishing_threshold: float = 1e-7
        self.high_loss_threshold: float = 100.0
        self.gpu_memory_warning_percent: float = 0.9
        self.epoch_timeout_seconds: float = 3600.0

    def record_loss(self, loss: float, epoch: int = 0, step: int = 0) -> None:
        """Record a loss value and check for issues."""
        with self._lock:
            self._loss_history.append(loss)

            # Check for loss divergence
            if len(self._loss_history) >= 10:
                recent = list(self._loss_history)[-10:]
                if all(r > self.loss_divergence_threshold for r in recent):
                    self._add_alert(
                        AlertType.LOSS_DIVERGENCE,
                        AlertSeverity.ERROR,
                        f"Loss diverging: last 10 values all > {self.loss_divergence_threshold}",
                        epoch=epoch,
                        step=step,
                        value=loss,
                        threshold=self.loss_divergence_threshold,
                    )

            # Check for high loss
            if loss > self.high_loss_threshold:
                self._add_alert(
                    AlertType.HIGH_LOSS,
                    AlertSeverity.WARNING,
                    f"Loss {loss:.4f} exceeds threshold {self.high_loss_threshold}",
                    epoch=epoch,
                    step=step,
                    value=loss,
                    threshold=self.high_loss_threshold,
                )

            # Check for loss stagnation
            if len(self._loss_history) >= self.loss_stagnation_window:
                recent = list(self._loss_history)[-self.loss_stagnation_window:]
                loss_range = max(recent) - min(recent)
                if loss_range < self.loss_stagnation_threshold:
                    self._add_alert(
                        AlertType.LOSS_STAGNATION,
                        AlertSeverity.WARNING,
                        f"Loss stagnant over {self.loss_stagnation_window} steps (range: {loss_range:.6f})",
                        epoch=epoch,
                        step=step,
                        value=loss_range,
                        threshold=self.loss_stagnation_threshold,
                    )

    def record_gradient(self, gradient_norm: float, epoch: int = 0, step: int = 0) -> None:
        """Record gradient norm and check for explosion/vanishing."""
        with self._lock:
            self._gradient_history.append(gradient_norm)

            if gradient_norm > self.gradient_explosion_threshold:
                self._add_alert(
                    AlertType.GRADIENT_EXPLOSION,
                    AlertSeverity.ERROR,
                    f"Gradient explosion: norm {gradient_norm:.4f} > {self.gradient_explosion_threshold}",
                    epoch=epoch,
                    step=step,
                    value=gradient_norm,
                    threshold=self.gradient_explosion_threshold,
                )

            if gradient_norm < self.gradient_vanishing_threshold and gradient_norm > 0:
                self._add_alert(
                    AlertType.GRADIENT_VANISHING,
                    AlertSeverity.WARNING,
                    f"Gradient vanishing: norm {gradient_norm:.2e} < {self.gradient_vanishing_threshold}",
                    epoch=epoch,
                    step=step,
                    value=gradient_norm,
                    threshold=self.gradient_vanishing_threshold,
                )

    def record_learning_rate(self, lr: float, epoch: int = 0, step: int = 0) -> None:
        """Record learning rate."""
        with self._lock:
            self._lr_history.append(lr)

            if lr < 1e-7 and len(self._lr_history) > 10:
                self._add_alert(
                    AlertType.LOW_LEARNING_RATE,
                    AlertSeverity.WARNING,
                    f"Learning rate very low: {lr:.2e}",
                    epoch=epoch,
                    step=step,
                    value=lr,
                    threshold=1e-7,
                )

    def record_gpu_memory(self, used: float, total: float, epoch: int = 0, step: int = 0) -> None:
        """Record GPU memory usage."""
        if total > 0:
            usage_percent = used / total
            if usage_percent > self.gpu_memory_warning_percent:
                self._add_alert(
                    AlertType.GPU_MEMORY_WARNING,
                    AlertSeverity.WARNING,
                    f"GPU memory usage {usage_percent:.0%} ({used:.1f}/{total:.1f} GB)",
                    epoch=epoch,
                    step=step,
                    value=used,
                    threshold=total * self.gpu_memory_warning_percent,
                )

    def record_metrics(self, metrics: TrainingMetrics) -> None:
        """Record full metrics snapshot."""
        with self._lock:
            self._metrics.append(metrics)
            if len(self._metrics) > 1000:
                self._metrics = self._metrics[-1000:]

    def record_failure(self, error: str, epoch: int = 0, step: int = 0) -> None:
        """Record a training failure."""
        self._add_alert(
            AlertType.TRAINING_FAILED,
            AlertSeverity.CRITICAL,
            f"Training failed: {error}",
            epoch=epoch,
            step=step,
            metadata={"error": error},
        )

    def check_epoch_timeout(self, epoch_start: float, epoch: int) -> None:
        """Check if an epoch is taking too long."""
        elapsed = time.time() - epoch_start
        if elapsed > self.epoch_timeout_seconds:
            self._add_alert(
                AlertType.EPOCH_TIMEOUT,
                AlertSeverity.ERROR,
                f"Epoch {epoch} exceeded timeout ({elapsed:.0f}s > {self.epoch_timeout_seconds:.0f}s)",
                epoch=epoch,
                value=elapsed,
                threshold=self.epoch_timeout_seconds,
            )

    def _add_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        **kwargs: Any,
    ) -> None:
        """Add an alert if not duplicate."""
        # Deduplicate: don't add same type within 60 seconds
        now = time.time()
        for existing in self._alerts:
            if (
                existing.alert_type == alert_type
                and now - existing.timestamp < 60
            ):
                return

        alert = TrainingAlert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            **kwargs,
        )
        self._alerts.append(alert)

        if len(self._alerts) > self._max_alerts:
            self._alerts = self._alerts[-self._max_alerts:]

        log_fn = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical,
        }.get(severity, logger.warning)

        log_fn("Training alert: %s", message, extra={"tag": "TRAIN"})

    def check_convergence(self, epoch: int, min_improvement: float = 0.01, window: int = 5) -> bool:
        """Check if training is converging.

        Returns True if training is converging, False if convergence issues detected.
        """
        with self._lock:
            if len(self._loss_history) < window * 2:
                return True

            recent = list(self._loss_history)[-window:]
            previous = list(self._loss_history)[-window * 2:-window]

            recent_avg = sum(recent) / len(recent)
            previous_avg = sum(previous) / len(previous)

            if previous_avg == 0:
                return True

            improvement = (previous_avg - recent_avg) / previous_avg

            if improvement < min_improvement:
                self._add_alert(
                    AlertType.CONVERGENCE_ISSUE,
                    AlertSeverity.WARNING,
                    f"Convergence slowing: {improvement:.2%} improvement < {min_improvement:.2%} threshold",
                    epoch=epoch,
                    value=improvement,
                    threshold=min_improvement,
                    metadata={
                        "recent_avg_loss": recent_avg,
                        "previous_avg_loss": previous_avg,
                    },
                )
                return False

            return True

    def get_alerts(
        self,
        severity: AlertSeverity | None = None,
        limit: int = 50,
    ) -> list[TrainingAlert]:
        """Get training alerts, optionally filtered by severity."""
        with self._lock:
            alerts = self._alerts
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            return alerts[-limit:]

    def get_metrics_history(self, limit: int = 100) -> list[TrainingMetrics]:
        """Get recent metrics history."""
        with self._lock:
            return self._metrics[-limit:]

    def get_status(self) -> dict:
        """Get monitor status summary."""
        with self._lock:
            return {
                "total_alerts": len(self._alerts),
                "alerts_by_severity": {
                    s.value: sum(1 for a in self._alerts if a.severity == s)
                    for s in AlertSeverity
                },
                "alerts_by_type": {
                    t.value: sum(1 for a in self._alerts if a.alert_type == t)
                    for t in AlertType
                },
                "total_metrics": len(self._metrics),
                "loss_history_size": len(self._loss_history),
                "gradient_history_size": len(self._gradient_history),
                "thresholds": {
                    "loss_divergence": self.loss_divergence_threshold,
                    "loss_stagnation_window": self.loss_stagnation_window,
                    "gradient_explosion": self.gradient_explosion_threshold,
                    "gradient_vanishing": self.gradient_vanishing_threshold,
                    "high_loss": self.high_loss_threshold,
                    "gpu_memory_warning": self.gpu_memory_warning_percent,
                    "epoch_timeout": self.epoch_timeout_seconds,
                },
            }

    def reset(self) -> None:
        """Reset monitor state."""
        with self._lock:
            self._loss_history.clear()
            self._lr_history.clear()
            self._gradient_history.clear()
            self._alerts.clear()
            self._metrics.clear()


_global_monitor: TrainingMonitor | None = None


def get_training_monitor() -> TrainingMonitor:
    """Get the global training monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = TrainingMonitor()
    return _global_monitor
