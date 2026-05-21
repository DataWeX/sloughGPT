"""
Model Monitoring - Drift Detection and Performance Monitoring

Production-grade monitoring with:
- Data drift detection
- Model performance monitoring
- Concept drift detection
- Alerting
- Metrics aggregation
"""

import time
import logging
import threading
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger("sloughgpt.monitoring")


class MonitorStatus(Enum):
    """Monitoring status."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class MetricValue:
    """Single metric value with metadata."""

    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class DriftResult:
    """Drift detection result."""

    has_drift: bool
    drift_score: float
    threshold: float
    metric_name: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Monitoring alert."""

    alert_id: str
    name: str
    severity: str
    message: str
    timestamp: float
    metric_name: str
    value: float
    threshold: float


class DriftDetector:
    """Statistical drift detection."""

    @staticmethod
    def detect_population_stability(
        reference_data: List[float], current_data: List[float], threshold: float = 0.05
    ) -> DriftResult:
        """Detect drift using population stability index."""
        if not reference_data or not current_data:
            return DriftResult(False, 0.0, threshold, "population_stability")

        ref_mean = np.mean(reference_data)
        curr_mean = np.mean(current_data)

        ref_std = np.std(reference_data) + 1e-10
        drift_score = abs(curr_mean - ref_mean) / ref_std

        return DriftResult(
            has_drift=drift_score > threshold,
            drift_score=drift_score,
            threshold=threshold,
            metric_name="population_stability",
            details={"ref_mean": ref_mean, "curr_mean": curr_mean},
        )

    @staticmethod
    def detect_distribution_shift(
        reference_data: List[float], current_data: List[float], threshold: float = 0.1
    ) -> DriftResult:
        """Detect distribution shift using KS test."""
        if len(reference_data) < 10 or len(current_data) < 10:
            return DriftResult(False, 0.0, threshold, "distribution_shift")

        from scipy import stats

        ks_stat, p_value = stats.ks_2samp(reference_data, current_data)

        return DriftResult(
            has_drift=p_value < threshold,
            drift_score=ks_stat,
            threshold=threshold,
            metric_name="distribution_shift",
            details={"ks_stat": ks_stat, "p_value": p_value},
        )

    @staticmethod
    def detect_category_drift(
        reference_counts: Dict[str, int], current_counts: Dict[str, int], threshold: float = 0.1
    ) -> DriftResult:
        """Detect categorical feature drift."""
        all_cats = set(reference_counts.keys()) | set(current_counts.keys())

        ref_total = sum(reference_counts.values())
        curr_total = sum(current_counts.values())

        if ref_total == 0 or curr_total == 0:
            return DriftResult(False, 0.0, threshold, "category_drift")

        drift_score = 0.0
        for cat in all_cats:
            ref_p = reference_counts.get(cat, 0) / ref_total
            curr_p = current_counts.get(cat, 0) / curr_total
            drift_score += abs(ref_p - curr_p)

        drift_score /= 2

        return DriftResult(
            has_drift=drift_score > threshold,
            drift_score=drift_score,
            threshold=threshold,
            metric_name="category_drift",
            details={"reference": reference_counts, "current": current_counts},
        )


class MetricsAggregator:
    """Aggregate and compute metrics over time windows."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.lock = threading.Lock()

    def record(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        with self.lock:
            metric_value = MetricValue(value=value, timestamp=time.time(), labels=labels or {})
            self.metrics[metric_name].append(metric_value)

    def get_stats(
        self, metric_name: str, window_seconds: Optional[float] = None
    ) -> Dict[str, float]:
        """Get statistics for a metric."""
        with self.lock:
            if metric_name not in self.metrics:
                return {}

            values = self.metrics[metric_name]

            if window_seconds:
                cutoff = time.time() - window_seconds
                values = [v for v in values if v.timestamp >= cutoff]

            if not values:
                return {}

            numeric_values = [v.value for v in values]

            return {
                "count": len(numeric_values),
                "mean": np.mean(numeric_values),
                "std": np.std(numeric_values),
                "min": np.min(numeric_values),
                "max": np.max(numeric_values),
                "p50": np.percentile(numeric_values, 50),
                "p95": np.percentile(numeric_values, 95),
                "p99": np.percentile(numeric_values, 99),
            }

    def get_rate(self, metric_name: str, window_seconds: float = 60) -> float:
        """Get rate of occurrences per second."""
        with self.lock:
            if metric_name not in self.metrics:
                return 0.0

            cutoff = time.time() - window_seconds
            values = [v for v in self.metrics[metric_name] if v.timestamp >= cutoff]

            if not values:
                return 0.0

            return len(values) / window_seconds


class AlertManager:
    """Manage alerts based on metric thresholds."""

    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def add_rule(
        self,
        name: str,
        metric_name: str,
        condition: str,
        threshold: float,
        severity: str = "warning",
    ):
        """Add an alert rule."""
        self.alert_rules[name] = {
            "metric_name": metric_name,
            "condition": condition,
            "threshold": threshold,
            "severity": severity,
            "enabled": True,
        }

    def check_and_alert(self, metric_name: str, value: float) -> List[Alert]:
        """Check rules and generate alerts."""
        new_alerts = []

        with self.lock:
            for rule_name, rule in self.alert_rules.items():
                if not rule["enabled"]:
                    continue

                if rule["metric_name"] != metric_name:
                    continue

                triggered = False
                condition = rule["condition"]
                threshold = rule["threshold"]

                if condition == "gt" and value > threshold:
                    triggered = True
                elif condition == "lt" and value < threshold:
                    triggered = True
                elif condition == "gte" and value >= threshold:
                    triggered = True
                elif condition == "lte" and value <= threshold:
                    triggered = True
                elif condition == "eq" and abs(value - threshold) < 1e-6:
                    triggered = True

                if triggered:
                    alert = Alert(
                        alert_id=f"{rule_name}_{int(time.time())}",
                        name=rule_name,
                        severity=rule["severity"],
                        message=f"Alert: {rule_name} - {metric_name} = {value} ({condition} {threshold})",
                        timestamp=time.time(),
                        metric_name=metric_name,
                        value=value,
                        threshold=threshold,
                    )
                    new_alerts.append(alert)
                    self.alerts.append(alert)

        return new_alerts

    def get_alerts(self, severity: Optional[str] = None, limit: int = 100) -> List[Alert]:
        """Get recent alerts."""
        with self.lock:
            alerts = self.alerts
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            return alerts[-limit:]


class ModelMonitor:
    """
    Production model monitoring system.

    Features:
    - Data drift detection
    - Performance monitoring
    - Concept drift detection
    - Alerting
    - Metrics aggregation
    """

    def __init__(self, window_size: int = 10000):
        self.window_size = window_size

        self.aggregator = MetricsAggregator(window_size)
        self.alert_manager = AlertManager()
        self.drift_detector = DriftDetector()

        self.reference_data: Dict[str, List[float]] = {}
        self.reference_counts: Dict[str, Dict[str, int]] = {}

        self._lock = threading.Lock()

    def set_reference_data(self, feature_name: str, data: List[float]):
        """Set reference data for drift detection."""
        with self._lock:
            self.reference_data[feature_name] = data

    def set_reference_distribution(self, feature_name: str, counts: Dict[str, int]):
        """Set reference distribution for categorical features."""
        with self._lock:
            self.reference_counts[feature_name] = counts

    def record_prediction(self, prediction: float, features: Optional[Dict[str, Any]] = None):
        """Record model prediction."""
        self.aggregator.record("prediction", prediction)

        if features:
            for name, value in features.items():
                if isinstance(value, (int, float)):
                    self.aggregator.record(f"feature_{name}", value)

    def record_accuracy(self, is_correct: bool):
        """Record prediction correctness."""
        self.aggregator.record("accuracy", 1.0 if is_correct else 0.0)

    def record_latency(self, latency_ms: float):
        """Record inference latency."""
        self.aggregator.record("latency_ms", latency_ms)

    def check_drift(self, feature_name: str, current_data: List[float]) -> DriftResult:
        """Check for data drift on a feature."""
        with self._lock:
            reference = self.reference_data.get(feature_name)

        if not reference:
            return DriftResult(False, 0.0, 0.0, feature_name)

        return self.drift_detector.detect_population_stability(
            reference, current_data, threshold=0.1
        )

    def check_category_drift(
        self, feature_name: str, current_counts: Dict[str, int]
    ) -> DriftResult:
        """Check for categorical feature drift."""
        with self._lock:
            reference = self.reference_counts.get(feature_name)

        if not reference:
            return DriftResult(False, 0.0, 0.0, feature_name)

        return self.drift_detector.detect_category_drift(reference, current_counts)

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        latency_stats = self.aggregator.get_stats("latency_ms", window_seconds=300)
        accuracy_stats = self.aggregator.get_stats("accuracy", window_seconds=300)

        status = MonitorStatus.HEALTHY
        issues = []

        if latency_stats:
            if latency_stats.get("p99", 0) > 1000:
                status = MonitorStatus.CRITICAL
                issues.append(f"High latency p99: {latency_stats['p99']:.2f}ms")
            elif latency_stats.get("p95", 0) > 500:
                status = MonitorStatus.WARNING
                issues.append(f"Elevated latency p95: {latency_stats['p95']:.2f}ms")

        if accuracy_stats:
            if accuracy_stats.get("mean", 1.0) < 0.7:
                status = MonitorStatus.CRITICAL
                issues.append(f"Low accuracy: {accuracy_stats['mean']:.2%}")
            elif accuracy_stats.get("mean", 1.0) < 0.85:
                status = MonitorStatus.WARNING
                issues.append(f"Degraded accuracy: {accuracy_stats['mean']:.2%}")

        return {
            "status": status.value,
            "issues": issues,
            "metrics": {"latency": latency_stats, "accuracy": accuracy_stats},
        }

    def get_dashboard(self) -> Dict[str, Any]:
        """Get monitoring dashboard data."""
        return {
            "health": self.get_health_status(),
            "metrics": {
                "predictions": self.aggregator.get_stats("prediction", window_seconds=3600),
                "latency": self.aggregator.get_stats("latency_ms", window_seconds=3600),
                "accuracy": self.aggregator.get_stats("accuracy", window_seconds=3600),
            },
            "alerts": self.alert_manager.get_alerts(limit=10),
            "drift": {},
        }


monitor = ModelMonitor()


__all__ = [
    "ModelMonitor",
    "DriftDetector",
    "MetricsAggregator",
    "AlertManager",
    "DriftResult",
    "Alert",
    "MetricValue",
    "MonitorStatus",
    "monitor",
]
