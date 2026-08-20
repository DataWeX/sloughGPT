"""
Thread-safe server state management.

Replaces module-level mutable globals in state.py with a Lock-protected
ServerState class. All reads and writes go through the singleton getter,
which ensures atomic access from any thread.
"""

from __future__ import annotations

from threading import Lock, RLock
from typing import Any, Optional, Callable, TypeVar
import time
import logging

logger = logging.getLogger("slo.infrastructure.server_state")

T = TypeVar("T")


class AtomicRef:
    """Thread-safe reference to a single value with change listeners."""

    def __init__(self, initial: T, name: str = ""):
        self._value: T = initial
        self._lock = Lock()
        self._name = name
        self._listeners: list[Callable[[T, T], None]] = []
        self._version = 0

    def get(self) -> T:
        with self._lock:
            return self._value

    def set(self, value: T) -> None:
        with self._lock:
            old = self._value
            self._value = value
            self._version += 1
        for listener in self._listeners:
            try:
                listener(old, value)
            except Exception as e:
                logger.warning("AtomicRef[%s] listener failed: %s", self._name, e, extra={"tag": "INFRA"})

    def swap(self, fn: Callable[[T], T]) -> T:
        """Atomically apply a function to the current value and return the new value."""
        with self._lock:
            old = self._value
            new = fn(old)
            self._value = new
            self._version += 1
        for listener in self._listeners:
            try:
                listener(old, new)
            except Exception as e:
                logger.warning("AtomicRef[%s] listener failed: %s", self._name, e, extra={"tag": "INFRA"})
        return new

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def on_change(self, listener: Callable[[T, T], None]) -> None:
        self._listeners.append(listener)


class ServerState:
    """Thread-safe singleton for all mutable server state.

    Usage::

        state = get_server_state()
        model = state.model.get()
        state.model.set(new_model)
    """

    def __init__(self) -> None:
        self.model = AtomicRef(None, "model")
        self.tokenizer = AtomicRef(None, "tokenizer")
        self.model_type = AtomicRef(None, "model_type")
        self.checkpoint = AtomicRef(None, "checkpoint")
        self.soul_engine = AtomicRef(None, "soul_engine")
        self.current_soul = AtomicRef(None, "current_soul")
        self.gen_config = AtomicRef(None, "gen_config")
        self.model_request_logger = AtomicRef(None, "model_request_logger")

        # Non-atomic fields (written once at startup, read-only after)
        self.torch_available: bool = False
        self.training_active: bool = False

        # Metrics
        self._started_at: float = time.time()
        self._request_count: int = 0
        self._error_count: int = 0
        self._lock = RLock()

        # Request history ring buffer (last 500 requests)
        self._request_history: list[dict] = []
        self._request_history_max: int = 500

        # Error history ring buffer (last 20 errors)
        self._error_history: list[dict] = []
        self._error_history_max: int = 20

        # Per-path latency accumulator: {path: [latency_ms, ...]}
        self._path_latencies: dict[str, list[float]] = {}
        self._path_latencies_max: int = 100

        # Inference-specific metrics
        self._inference_count: int = 0
        self._total_tokens: int = 0
        self._total_inference_ms: float = 0.0
        self._tokens_per_request: list[int] = []
        self._tokens_per_request_max: int = 100

        # Sliding window for recent tokens/s (last 30 inferences)
        self._recent_inferences: list[dict] = []  # [{tokens, elapsed_ms, ts}, ...]
        self._recent_inferences_max: int = 30

        # Per-model inference tracking: {model: {tokens, time_ms, count}}
        self._model_metrics: dict[str, dict] = {}

        # Model lifecycle events (load/unload/error)
        self._model_events: list[dict] = []
        self._model_events_max: int = 30

        # Health score history (snapshots over time)
        self._health_history: list[dict] = []
        self._health_history_max: int = 30

        self._last_trend_ts: float = 0.0

        # Memory snapshots (periodic RSS/virtual memory tracking)
        self._memory_history: list[dict] = []
        self._memory_history_max: int = 30

        # Rate limiting: {path: {window_start, count}}
        self._rate_limits: dict[str, dict] = {}
        self._rate_limit_violations: list[dict] = []
        self._rate_limit_violations_max: int = 20

        # Memory pressure tracking
        self._memory_pressure_blocks: int = 0  # times inference was blocked by >95% memory
        self._gc_cycles: int = 0  # GC triggers before inference

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._started_at

    def record_request(self) -> None:
        with self._lock:
            self._request_count += 1

    def record_error(self) -> None:
        with self._lock:
            self._error_count += 1

    def record_request_latency(
        self, path: str, method: str, status: int, elapsed_ms: float,
    ) -> None:
        """Append a request entry to the history ring buffer."""
        with self._lock:
            self._request_history.append({
                "path": path,
                "method": method,
                "status": status,
                "elapsed_ms": round(elapsed_ms, 1),
                "ts": time.time(),
            })
            if len(self._request_history) > self._request_history_max:
                self._request_history = self._request_history[-self._request_history_max:]

    def get_request_history(self, limit: int = 20) -> list[dict]:
        """Return the most recent requests (newest first)."""
        with self._lock:
            return list(reversed(self._request_history[-limit:]))

    def get_avg_latency(self, window: int = 20) -> float:
        """Average latency of the last N requests in ms."""
        with self._lock:
            recent = self._request_history[-window:]
            if not recent:
                return 0.0
            return round(sum(r["elapsed_ms"] for r in recent) / len(recent), 1)

    def get_p95_latency(self) -> float:
        """P95 latency of recent requests in ms (last 100 requests)."""
        with self._lock:
            recent = self._request_history[-100:]
            if not recent:
                return 0.0
            sorted_lat = sorted(r["elapsed_ms"] for r in recent)
            idx = int(len(sorted_lat) * 0.95)
            return round(sorted_lat[min(idx, len(sorted_lat) - 1)], 1)

    def record_error_detail(
        self, path: str, method: str, status: int, message: str, error_type: str = "",
    ) -> None:
        """Append an error entry to the history ring buffer."""
        with self._lock:
            self._error_count += 1
            self._error_history.append({
                "path": path,
                "method": method,
                "status": status,
                "message": message[:200],
                "error_type": error_type,
                "ts": time.time(),
            })
            if len(self._error_history) > self._error_history_max:
                self._error_history = self._error_history[-self._error_history_max:]

    def get_error_history(self, limit: int = 10) -> list[dict]:
        """Return the most recent errors (newest first)."""
        with self._lock:
            return list(reversed(self._error_history[-limit:]))

    def record_path_latency(self, path: str, elapsed_ms: float) -> None:
        """Track per-path latency for breakdown analysis."""
        with self._lock:
            if path not in self._path_latencies:
                self._path_latencies[path] = []
            self._path_latencies[path].append(elapsed_ms)
            if len(self._path_latencies[path]) > self._path_latencies_max:
                self._path_latencies[path] = self._path_latencies[path][-self._path_latencies_max:]

    def get_path_latencies(self, top_n: int = 5) -> list[dict]:
        """Return per-path average latency for the top N busiest paths."""
        with self._lock:
            result = []
            for path, latencies in self._path_latencies.items():
                if not latencies:
                    continue
                avg = sum(latencies) / len(latencies)
                result.append({
                    "path": path,
                    "avg_ms": round(avg, 1),
                    "count": len(latencies),
                    "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if len(latencies) >= 2 else round(avg, 1),
                })
            result.sort(key=lambda x: x["count"], reverse=True)
            return result[:top_n]

    def get_requests_per_minute(self) -> float:
        """Requests in the last 60 seconds."""
        cutoff = time.time() - 60
        with self._lock:
            return sum(1 for r in self._request_history if r["ts"] > cutoff)

    def record_inference(self, tokens: int, elapsed_ms: float, model: str = "") -> None:
        """Record a single inference: token count and generation time."""
        with self._lock:
            self._inference_count += 1
            self._total_tokens += tokens
            self._total_inference_ms += elapsed_ms
            self._tokens_per_request.append(tokens)
            if len(self._tokens_per_request) > self._tokens_per_request_max:
                self._tokens_per_request = self._tokens_per_request[-self._tokens_per_request_max:]
            # Sliding window for recent tokens/s
            self._recent_inferences.append({
                "tokens": tokens,
                "elapsed_ms": elapsed_ms,
                "ts": time.time(),
            })
            if len(self._recent_inferences) > self._recent_inferences_max:
                self._recent_inferences = self._recent_inferences[-self._recent_inferences_max:]
            if model:
                if model not in self._model_metrics:
                    self._model_metrics[model] = {"tokens": 0, "time_ms": 0.0, "count": 0}
                self._model_metrics[model]["tokens"] += tokens
                self._model_metrics[model]["time_ms"] += elapsed_ms
                self._model_metrics[model]["count"] += 1

    def get_tokens_per_second(self) -> float:
        """Recent tokens/sec — sliding window of last 10 inferences.

        Falls back to lifetime average when fewer than 3 inferences recorded
        (sliding window needs at least 2 data points for a meaningful rate).
        """
        with self._lock:
            recent = self._recent_inferences[-10:]
            if len(recent) < 3:
                # Not enough data — use lifetime average
                if self._total_inference_ms <= 0:
                    return 0.0
                return round(self._total_tokens / (self._total_inference_ms / 1000), 1)
            total_tokens = sum(r["tokens"] for r in recent)
            total_ms = sum(r["elapsed_ms"] for r in recent)
            if total_ms <= 0:
                return 0.0
            return round(total_tokens / (total_ms / 1000), 1)

    def get_avg_tokens_per_request(self) -> float:
        """Average tokens generated per inference request (last N)."""
        with self._lock:
            if not self._tokens_per_request:
                return 0.0
            return round(sum(self._tokens_per_request) / len(self._tokens_per_request), 1)

    @property
    def inference_count(self) -> int:
        with self._lock:
            return self._inference_count

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return self._total_tokens

    def get_model_metrics(self) -> list[dict]:
        """Return per-model inference stats, sorted by count descending."""
        with self._lock:
            result = []
            for model, m in self._model_metrics.items():
                tps = round(m["tokens"] / (m["time_ms"] / 1000), 1) if m["time_ms"] > 0 else 0.0
                result.append({
                    "model": model,
                    "count": m["count"],
                    "total_tokens": m["tokens"],
                    "tokens_per_sec": tps,
                    "avg_tokens": round(m["tokens"] / m["count"], 0) if m["count"] > 0 else 0,
                })
            result.sort(key=lambda x: x["count"], reverse=True)
            return result

    def get_health_score(self, cpu_percent: float = 0.0, memory_percent: float = 0.0) -> dict:
        """Composite health score.

        Returns a dict with score (0-100), status, and summary.
        Requires ``health_flow`` module — if unavailable, returns a basic score.
        """
        try:
            from .health_flow import run_health_flow
        except ImportError:
            return self._basic_health_score(cpu_percent, memory_percent)

        with self._lock:
            req_count = self._request_count
            err_count = self._error_count

        avg_lat = self.get_avg_latency()
        tps = self.get_tokens_per_second()
        uptime = time.time() - self._started_at
        model = self.model.get()
        model_loaded = model is not None
        model_type = getattr(model, "name_or_path", "") or (self.model_type.get() or "")

        result = run_health_flow(
            req_count=req_count,
            err_count=err_count,
            avg_latency_ms=avg_lat,
            tokens_per_sec=tps,
            uptime_seconds=uptime,
            model_loaded=model_loaded,
            model_type=model_type,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
        )

        return {
            "score": result.score,
            "status": result.status,
            "summary": result.summary,
            "diagnoses": [
                {"check": d.check, "severity": d.severity.value, "score": round(d.score), "message": d.message}
                for d in result.diagnoses
            ],
        }

    def _basic_health_score(self, cpu_percent: float, memory_percent: float) -> dict:
        """Fallback health score when health_flow is not available."""
        score = 100
        issues = []
        with self._lock:
            err_count = self._error_count
            req_count = self._request_count
        if req_count > 0:
            err_rate = err_count / req_count
            if err_rate > 0.1:
                score -= 30
                issues.append(f"High error rate: {err_rate:.1%}")
            elif err_rate > 0.01:
                score -= 10
                issues.append(f"Elevated error rate: {err_rate:.1%}")
        if cpu_percent > 90:
            score -= 20
            issues.append(f"High CPU: {cpu_percent:.0f}%")
        if memory_percent > 90:
            score -= 20
            issues.append(f"High memory: {memory_percent:.0f}%")
        if score >= 80:
            status = "healthy"
        elif score >= 50:
            status = "degraded"
        else:
            status = "unhealthy"
        return {
            "score": score,
            "status": status,
            "summary": "; ".join(issues) if issues else "All clear",
            "diagnoses": [],
        }

    def record_model_event(self, event_type: str, model: str, detail: str = "") -> None:
        """Record a model lifecycle event: load, unload, error, swap."""
        with self._lock:
            self._model_events.append({
                "type": event_type,
                "model": model,
                "detail": detail[:200],
                "ts": time.time(),
            })
            if len(self._model_events) > self._model_events_max:
                self._model_events = self._model_events[-self._model_events_max:]

    def get_model_events(self, limit: int = 10) -> list[dict]:
        """Return the most recent model events (newest first)."""
        with self._lock:
            return list(reversed(self._model_events[-limit:]))

    def record_health_snapshot(self) -> None:
        """Snapshot the current health score into history for trend analysis."""
        score_data = self.get_health_score()
        with self._lock:
            self._health_history.append({
                "score": score_data["score"],
                "status": score_data["status"],
                "ts": time.time(),
            })
            if len(self._health_history) > self._health_history_max:
                self._health_history = self._health_history[-self._health_history_max:]

    def record_trend_snapshots(self, interval_s: float = 5.0) -> None:
        """Record health + memory trend snapshots if interval_s has elapsed.

        Throttled so high-frequency health polling does not flood the
        history ring buffers.

        Args:
            interval_s: minimum seconds between trend records.

        Returns:
            None

        Side effects:
            - appends to health_history and memory_history
        """
        now = time.time()
        with self._lock:
            if now - self._last_trend_ts < interval_s:
                return
            self._last_trend_ts = now
        self.record_health_snapshot()
        self.record_memory_snapshot()

    def get_health_history(self, limit: int = 20) -> list[dict]:
        """Return health score history (oldest first, for charting)."""
        with self._lock:
            return list(self._health_history[-limit:])

    def record_memory_snapshot(self) -> None:
        """Snapshot current memory usage (RSS + virtual) for trend tracking."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            process = psutil.Process()
            proc_mem = process.memory_info()
            rss_mb = proc_mem.rss / (1024 * 1024)
            virtual_mb = proc_mem.vms / (1024 * 1024)
            system_percent = mem.percent
        except Exception:
            rss_mb = 0.0
            virtual_mb = 0.0
            system_percent = 0.0
        with self._lock:
            self._memory_history.append({
                "rss_mb": round(rss_mb, 1),
                "virtual_mb": round(virtual_mb, 1),
                "system_percent": round(system_percent, 1),
                "ts": time.time(),
            })
            if len(self._memory_history) > self._memory_history_max:
                self._memory_history = self._memory_history[-self._memory_history_max:]

    def get_memory_history(self, limit: int = 20) -> list[dict]:
        """Return memory usage history (oldest first, for charting)."""
        with self._lock:
            return list(self._memory_history[-limit:])

    def check_rate_limit(self, path: str, max_per_second: int = 30) -> bool:
        """Check if a path exceeds rate limit. Returns True if allowed, False if blocked."""
        now = time.time()
        with self._lock:
            if path not in self._rate_limits:
                self._rate_limits[path] = {"window_start": now, "count": 0}
            entry = self._rate_limits[path]
            if now - entry["window_start"] >= 1.0:
                entry["window_start"] = now
                entry["count"] = 0
            entry["count"] += 1
            if entry["count"] > max_per_second:
                self._rate_limit_violations.append({
                    "path": path,
                    "count": entry["count"],
                    "limit": max_per_second,
                    "ts": now,
                })
                if len(self._rate_limit_violations) > self._rate_limit_violations_max:
                    self._rate_limit_violations = self._rate_limit_violations[-self._rate_limit_violations_max:]
                return False
            return True

    def get_rate_limit_violations(self, limit: int = 10) -> list[dict]:
        """Return recent rate limit violations (newest first)."""
        with self._lock:
            return list(reversed(self._rate_limit_violations[-limit:]))

    def record_memory_pressure_block(self) -> None:
        """Record that inference was blocked due to memory pressure."""
        with self._lock:
            self._memory_pressure_blocks += 1

    def record_gc_cycle(self) -> None:
        """Record that a GC cycle was triggered before inference."""
        with self._lock:
            self._gc_cycles += 1

    def get_memory_pressure_stats(self) -> dict:
        """Return memory pressure stats for monitoring."""
        with self._lock:
            return {
                "pressure_blocks": self._memory_pressure_blocks,
                "gc_cycles": self._gc_cycles,
            }

    @property
    def request_count(self) -> int:
        with self._lock:
            return self._request_count

    @property
    def error_count(self) -> int:
        with self._lock:
            return self._error_count


_server_state: Optional[ServerState] = None
_server_state_lock = Lock()


def get_server_state() -> ServerState:
    """Get (or create) the singleton ServerState."""
    global _server_state
    if _server_state is None:
        with _server_state_lock:
            if _server_state is None:
                _server_state = ServerState()
    return _server_state
