"""
Health flow — composable diagnostic pipeline.

Each check is a function that takes a snapshot and returns a Diagnosis.
Checks run in order; the first critical/summary check wins.

Design:
  - Each check owns ONE concern (errors, latency, throughput, model, uptime)
  - Each check returns a severity + human-readable sentence
  - The flow aggregates into overall score, status, and a single summary line
  - No hardcoded thresholds scattered across files — all in one place
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class Severity(str, Enum):
    OK = "ok"
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"


@dataclass
class Diagnosis:
    """Result of a single health check."""
    check: str
    severity: Severity
    score: float  # 0-100
    message: str  # human-readable, no jargon
    detail: str = ""


@dataclass
class HealthFlowResult:
    """Aggregated result of the full diagnostic flow."""
    score: int
    status: str  # healthy / degraded / unhealthy
    summary: str  # one-liner for toast / status bar
    diagnoses: list[Diagnosis] = field(default_factory=list)
    model_loaded: bool = False
    model_type: str = ""


def _check_errors(req_count: int, err_count: int) -> Diagnosis:
    """Check error rate and produce a human diagnosis."""
    if req_count == 0:
        return Diagnosis("errors", Severity.OK, 100, "No requests yet — all clear.")

    err_rate = err_count / req_count
    score = max(0.0, 1.0 - err_rate / 0.05) * 100

    if err_rate == 0:
        msg = f"All {req_count} requests OK."
    elif err_rate < 0.01:
        msg = f"{err_count}/{req_count} errors — rare."
    elif err_rate < 0.05:
        msg = f"{err_count}/{req_count} errors — degrading."
    else:
        msg = f"{err_count}/{req_count} errors — failing."

    severity = (
        Severity.OK if score >= 80
        else Severity.WARN if score >= 50
        else Severity.CRITICAL
    )
    return Diagnosis("errors", severity, score, msg)


def _check_latency(avg_ms: float) -> Diagnosis:
    """Check average response latency."""
    if avg_ms == 0:
        return Diagnosis("latency", Severity.OK, 100, "No requests measured yet.")

    score = max(0.0, 1.0 - (avg_ms - 200) / 1800) * 100

    if avg_ms < 300:
        msg = f"Snappy — {avg_ms:.0f}ms avg."
    elif avg_ms < 800:
        msg = f"Fine — {avg_ms:.0f}ms avg."
    elif avg_ms < 1500:
        msg = f"Slow — {avg_ms:.0f}ms avg."
    else:
        msg = f"Very slow — {avg_ms:.0f}ms avg."

    severity = (
        Severity.OK if score >= 80
        else Severity.WARN if score >= 50
        else Severity.CRITICAL
    )
    return Diagnosis("latency", severity, score, msg)


def _check_throughput(tokens_per_sec: float) -> Diagnosis:
    """Check generation speed."""
    if tokens_per_sec == 0:
        return Diagnosis("throughput", Severity.INFO, 50, "No generation data yet.")

    score = min(1.0, max(0.0, (tokens_per_sec - 5) / 45)) * 100

    if tokens_per_sec >= 30:
        msg = f"Fast — {tokens_per_sec:.0f} tok/s."
    elif tokens_per_sec >= 10:
        msg = f"OK — {tokens_per_sec:.0f} tok/s."
    elif tokens_per_sec >= 5:
        msg = f"Slow — {tokens_per_sec:.0f} tok/s."
    else:
        msg = f"Very slow — {tokens_per_sec:.0f} tok/s."

    severity = (
        Severity.OK if score >= 80
        else Severity.WARN if score >= 50
        else Severity.CRITICAL
    )
    return Diagnosis("throughput", severity, score, msg)


def _check_model(model_loaded: bool, model_type: str) -> Diagnosis:
    """Check if a model is loaded and ready."""
    if model_loaded and model_type:
        return Diagnosis("model", Severity.OK, 100, f"{model_type} loaded.")
    elif model_loaded:
        return Diagnosis("model", Severity.OK, 100, "Model loaded.")
    else:
        return Diagnosis("model", Severity.WARN, 40, "No model loaded.")


def _check_uptime(uptime_seconds: float) -> Diagnosis:
    """Check server stability (uptime as proxy)."""
    if uptime_seconds > 3600:
        msg = f"Up {uptime_seconds / 3600:.1f}h."
    elif uptime_seconds > 60:
        msg = f"Up {uptime_seconds / 60:.0f}m."
    elif uptime_seconds > 10:
        msg = "Warming up."
    else:
        msg = "Just booted."

    score = 100 if uptime_seconds > 60 else 50 if uptime_seconds > 10 else 30
    severity = Severity.OK if score >= 80 else Severity.INFO
    return Diagnosis("uptime", severity, score, msg)


def _check_resources(cpu_percent: float, memory_percent: float) -> Diagnosis:
    """Check system resource pressure (CPU + memory).

    High CPU (>85%) or high memory (>90%) degrades the score. Both combined
    is worse than either alone. A server at 99% CPU and 95% memory should
    show degraded/unhealthy, not healthy.
    """
    if cpu_percent <= 0 and memory_percent <= 0:
        return Diagnosis("resources", Severity.INFO, 80, "Resource data not available yet.")

    # CPU penalty: 0% at 50%, linear to 0 at 100%
    cpu_score = max(0.0, 1.0 - max(0, cpu_percent - 50) / 50) * 100
    # Memory penalty: 0% at 70%, linear to 0 at 100%
    mem_score = max(0.0, 1.0 - max(0, memory_percent - 70) / 30) * 100

    # Weighted average (CPU 40%, memory 60% — memory pressure is more dangerous)
    score = cpu_score * 0.4 + mem_score * 0.6

    # Build message
    parts = []
    if cpu_percent > 85:
        parts.append(f"CPU at {cpu_percent:.0f}% — throttling risk")
    elif cpu_percent > 50:
        parts.append(f"CPU at {cpu_percent:.0f}%")
    if memory_percent > 90:
        parts.append(f"Memory at {memory_percent:.0f}% — near limit")
    elif memory_percent > 70:
        parts.append(f"Memory at {memory_percent:.0f}%")

    if not parts:
        msg = f"CPU {cpu_percent:.0f}%, memory {memory_percent:.0f}% — headroom OK."
    else:
        msg = " ".join(parts) + "."

    severity = (
        Severity.OK if score >= 70
        else Severity.WARN if score >= 40
        else Severity.CRITICAL
    )
    return Diagnosis("resources", severity, score, msg)


# ── The flow ──────────────────────────────────────────────────────────────

def run_health_flow(
    req_count: int,
    err_count: int,
    avg_latency_ms: float,
    tokens_per_sec: float,
    uptime_seconds: float,
    model_loaded: bool,
    model_type: str = "",
    cpu_percent: float = 0.0,
    memory_percent: float = 0.0,
) -> HealthFlowResult:
    """Run all checks in order, aggregate into a single health verdict.

    Each check is independent — order doesn't matter for scoring, but we
    put errors first because they're the most actionable.
    """
    diagnoses = [
        _check_errors(req_count, err_count),
        _check_latency(avg_latency_ms),
        _check_throughput(tokens_per_sec),
        _check_model(model_loaded, model_type),
        _check_uptime(uptime_seconds),
        _check_resources(cpu_percent, memory_percent),
    ]

    # Weighted score — resources added to reflect system pressure
    weights = {
        "errors": 0.25,
        "latency": 0.20,
        "throughput": 0.15,
        "model": 0.10,
        "uptime": 0.05,
        "resources": 0.25,
    }
    total = 0.0
    for d in diagnoses:
        w = weights.get(d.check, 0.1)
        total += d.score * w
    score = round(total)

    # Status from score
    if score >= 80:
        status = "healthy"
    elif score >= 50:
        status = "degraded"
    else:
        status = "unhealthy"

    # Summary: pick the most interesting diagnosis (worst severity first)
    severity_order = {Severity.CRITICAL: 0, Severity.WARN: 1, Severity.INFO: 2, Severity.OK: 3}
    worst = min(diagnoses, key=lambda d: severity_order[d.severity])
    summary = worst.message if worst.severity != Severity.OK else diagnoses[0].message

    # Prepend model info if loaded
    if model_loaded and model_type:
        summary = f"{model_type}: {summary}"

    return HealthFlowResult(
        score=score,
        status=status,
        summary=summary,
        diagnoses=diagnoses,
        model_loaded=model_loaded,
        model_type=model_type,
    )
