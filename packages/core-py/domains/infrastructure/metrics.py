from __future__ import annotations

"""
Prometheus-compatible metrics collector.

Exposes request counts, latencies, error rates, and model inference stats
in Prometheus text exposition format.  No external dependencies.

Usage::

    from domains.infrastructure.metrics import get_metrics_collector
    collector = get_metrics_collector()
    collector.record_request("/chat", 200, 1.5)
    print(collector.render())
"""

import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional

try:  # pragma: no cover - psutil is optional (system metrics only)
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


class MetricsCollector:
    """In-memory Prometheus-style metrics collector."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time = time.time()

        # Counters
        self._request_count: Dict[str, int] = defaultdict(int)
        self._request_errors: Dict[str, int] = defaultdict(int)
        self._tokens_generated: int = 0
        self._inference_count: int = 0

        # Histograms (simple bucket approach)
        self._request_latencies: Dict[str, List[float]] = defaultdict(list)
        self._inference_latencies: List[float] = []

        # Gauges
        self._active_requests: int = 0
        self._model_loaded: bool = False
        self._model_name: str = ""

    # ── Recording methods ────────────────────────────────────────────

    def record_request(self, path: str, status: int, duration: float) -> None:
        """Record a completed HTTP request."""
        with self._lock:
            self._request_count[path] += 1
            self._request_latencies[path].append(duration)
            if status >= 400:
                self._request_errors[path] += 1

    def record_inference(self, duration: float, tokens: int = 0) -> None:
        """Record a model inference call."""
        with self._lock:
            self._inference_count += 1
            self._inference_latencies.append(duration)
            self._tokens_generated += tokens

    def set_active_requests(self, count: int) -> None:
        with self._lock:
            self._active_requests = count

    def get_active_requests(self) -> int:
        """Return the current in-flight request count (thread-safe)."""
        with self._lock:
            return self._active_requests

    def set_model_info(self, loaded: bool, name: str = "") -> None:
        with self._lock:
            self._model_loaded = loaded
            self._model_name = name

    # ── Prometheus text format ────────────────────────────────────────

    def render(self) -> str:
        """Render all metrics in Prometheus text exposition format."""
        lines: List[str] = []

        uptime = time.time() - self._start_time
        lines.append("# HELP sloughgpt_uptime_seconds Server uptime in seconds.")
        lines.append("# TYPE sloughgpt_uptime_seconds gauge")
        lines.append(f"sloughgpt_uptime_seconds {uptime:.1f}")

        with self._lock:
            # Request counter
            lines.append("# HELP sloughgpt_requests_total Total HTTP requests by path.")
            lines.append("# TYPE sloughgpt_requests_total counter")
            for path, count in sorted(self._request_count.items()):
                lines.append(f'sloughgpt_requests_total{{path="{path}"}} {count}')

            # Error counter
            lines.append("# HELP sloughgpt_request_errors_total Total HTTP errors by path.")
            lines.append("# TYPE sloughgpt_request_errors_total counter")
            for path, count in sorted(self._request_errors.items()):
                lines.append(f'sloughgpt_request_errors_total{{path="{path}"}} {count}')

            # Request latency histogram (simple: p50/p95/p99)
            lines.append("# HELP sloughgpt_request_duration_seconds Request latency percentiles.")
            lines.append("# TYPE sloughgpt_request_duration_seconds summary")
            for path, latencies in sorted(self._request_latencies.items()):
                if not latencies:
                    continue
                sorted_lat = sorted(latencies)
                n = len(sorted_lat)
                p50 = sorted_lat[int(n * 0.5)] if n > 0 else 0
                p95 = sorted_lat[int(n * 0.95)] if n > 1 else p50
                p99 = sorted_lat[int(n * 0.99)] if n > 2 else p95
                sum(sorted_lat) / n
                lines.append(f'sloughgpt_request_duration_seconds{{path="{path}",quantile="0.5"}} {p50:.4f}')
                lines.append(f'sloughgpt_request_duration_seconds{{path="{path}",quantile="0.95"}} {p95:.4f}')
                lines.append(f'sloughgpt_request_duration_seconds{{path="{path}",quantile="0.99"}} {p99:.4f}')
                lines.append(f'sloughgpt_request_duration_seconds_sum{{path="{path}"}} {sum(sorted_lat):.4f}')
                lines.append(f'sloughgpt_request_duration_seconds_count{{path="{path}"}} {n}')

            # Inference metrics
            lines.append("# HELP sloughgpt_inferences_total Total model inference calls.")
            lines.append("# TYPE sloughgpt_inferences_total counter")
            lines.append(f"sloughgpt_inferences_total {self._inference_count}")

            lines.append("# HELP sloughgpt_tokens_generated_total Total tokens generated.")
            lines.append("# TYPE sloughgpt_tokens_generated_total counter")
            lines.append(f"sloughgpt_tokens_generated_total {self._tokens_generated}")

            if self._inference_latencies:
                sorted_il = sorted(self._inference_latencies)
                n = len(sorted_il)
                sum(sorted_il) / n
                p50 = sorted_il[int(n * 0.5)]
                p95 = sorted_il[int(n * 0.95)] if n > 1 else p50
                lines.append("# HELP sloughgpt_inference_duration_seconds Inference latency percentiles.")
                lines.append("# TYPE sloughgpt_inference_duration_seconds summary")
                lines.append(f'sloughgpt_inference_duration_seconds{{quantile="0.5"}} {p50:.4f}')
                lines.append(f'sloughgpt_inference_duration_seconds{{quantile="0.95"}} {p95:.4f}')
                lines.append(f'sloughgpt_inference_duration_seconds_sum {sum(sorted_il):.4f}')
                lines.append(f'sloughgpt_inference_duration_seconds_count {n}')

            # Gauges
            lines.append("# HELP sloughgpt_active_requests Currently in-flight requests.")
            lines.append("# TYPE sloughgpt_active_requests gauge")
            lines.append(f"sloughgpt_active_requests {self._active_requests}")

            lines.append("# HELP sloughgpt_model_loaded Whether a model is loaded (1=yes).")
            lines.append("# TYPE sloughgpt_model_loaded gauge")
            lines.append(f"sloughgpt_model_loaded {1 if self._model_loaded else 0}")

            # System metrics (from psutil)
            if psutil is not None:
                cpu_pct = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory()
                lines.append("# HELP sloughgpt_system_cpu_usage CPU usage percentage.")
                lines.append("# TYPE sloughgpt_system_cpu_usage gauge")
                lines.append(f"sloughgpt_system_cpu_usage {cpu_pct:.1f}")

                lines.append("# HELP sloughgpt_system_memory_percent Memory usage percentage.")
                lines.append("# TYPE sloughgpt_system_memory_percent gauge")
                lines.append(f"sloughgpt_system_memory_percent {mem.percent:.1f}")

        return "\n".join(lines) + "\n"


# ── Singleton ────────────────────────────────────────────────────────

_collector: Optional[MetricsCollector] = None
_metrics_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    global _collector
    if _collector is None:
        with _metrics_lock:
            if _collector is None:
                _collector = MetricsCollector()
    return _collector


def reset_metrics_collector() -> None:
    """Reset the singleton (for testing)."""
    global _collector
    with _metrics_lock:
        _collector = None
