"""
Performance Monitor - Ported from recovered performance_optimizer.py
Real-time monitoring and optimization for training.
"""

import time
import psutil
from typing import Dict, List


class PerformanceMonitor:
    """Monitor training performance and suggest optimizations."""

    def __init__(self):
        self.metrics = {"start_time": None, "epoch_times": [], "memory_usage": [], "cpu_usage": []}
        self._device = self._detect_device()

    def _detect_device(self) -> str:
        """Detect best available device."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _get_memory_info(self) -> Dict:
        """Get memory usage info."""
        memory = psutil.virtual_memory()

        info = {
            "system_total_gb": memory.total / (1024**3),
            "system_used_gb": memory.used / (1024**3),
            "system_available_gb": memory.available / (1024**3),
        }

        try:
            import torch

            if torch.cuda.is_available():
                info["gpu_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                info["gpu_allocated_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
                info["gpu_reserved_gb"] = torch.cuda.memory_reserved(0) / (1024**3)
        except ImportError:
            pass

        return info

    def _get_cpu_info(self) -> Dict:
        """Get CPU usage info."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
        }

    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        self.metrics["start_time"] = time.time()

    def record_epoch_time(self, epoch_time: float) -> None:
        """Record epoch completion time."""
        self.metrics["epoch_times"].append(epoch_time)

    def record_memory(self) -> None:
        """Record current memory usage."""
        self.metrics["memory_usage"].append(self._get_memory_info())

    def get_average_epoch_time(self) -> float:
        """Get average epoch time."""
        if not self.metrics["epoch_times"]:
            return 0.0
        return sum(self.metrics["epoch_times"]) / len(self.metrics["epoch_times"])

    def get_current_stats(self) -> Dict:
        """Get current performance stats."""
        return {
            "device": self._device,
            "memory": self._get_memory_info(),
            "cpu": self._get_cpu_info(),
            "avg_epoch_time": self.get_average_epoch_time(),
        }

    def suggest_optimizations(self) -> List[str]:
        """Suggest optimizations based on current stats."""
        suggestions = []

        memory = self._get_memory_info()

        if memory.get("system_used_gb", 0) / memory.get("system_total_gb", 1) > 0.9:
            suggestions.append("System memory is running low. Consider reducing batch size.")

        if self._device == "cpu":
            suggestions.append("Running on CPU. For better performance, use CUDA or MPS.")

        return suggestions


class PerformanceOptimizer:
    """Optimize performance based on metrics."""

    def __init__(self):
        self.monitor = PerformanceMonitor()

    def auto_tune_batch_size(self, initial_batch_size: int = 32) -> int:
        """Auto-tune batch size based on available memory."""
        memory = self.monitor._get_memory_info()

        available_gb = memory.get("system_available_gb", 8)

        if available_gb > 16:
            return initial_batch_size * 2
        elif available_gb < 4:
            return max(1, initial_batch_size // 2)

        return initial_batch_size

    def get_optimal_workers(self) -> int:
        """Get optimal number of data loading workers."""
        cpu_count = psutil.cpu_count() or 4
        return max(1, cpu_count - 2)


__all__ = ["PerformanceMonitor", "PerformanceOptimizer"]
