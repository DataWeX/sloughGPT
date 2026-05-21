"""Thin adapters from the TUI to HTTP APIs and local trainers."""

from apps.tui.adapters.http_api import (
    ApiJsonResult,
    HealthFetchResult,
    fetch_health,
    fetch_health_detailed,
    fetch_metrics,
)
from apps.tui.adapters.local_status import LocalStatusSnapshot, scan_local_repo
from apps.tui.adapters.training import (
    HttpTrainAdapter,
    HttpTrainResult,
    LocalTrainAdapter,
    TrainConfig,
    TrainProgress,
    TrainResult,
)
from apps.tui.adapters.docker import DockerAdapter

__all__ = [
    "ApiJsonResult",
    "HealthFetchResult",
    "LocalStatusSnapshot",
    "fetch_health",
    "fetch_health_detailed",
    "fetch_metrics",
    "scan_local_repo",
    "HttpTrainAdapter",
    "HttpTrainResult",
    "LocalTrainAdapter",
    "TrainConfig",
    "TrainProgress",
    "TrainResult",
    "DockerAdapter",
]
