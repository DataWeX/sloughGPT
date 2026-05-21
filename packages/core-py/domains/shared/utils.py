"""
SloughGPT Utilities
Common utility functions for the SloughGPT AI Framework
"""

import json
import hashlib
import random
import string
from datetime import datetime
from typing import Any, Dict, List, Optional


def generate_id(prefix: str = "") -> str:
    """Generate a random ID."""
    chars = string.ascii_lowercase + string.digits
    random_id = "".join(random.choices(chars, k=8))
    return f"{prefix}{random_id}" if prefix else random_id


def hash_string(s: str, algorithm: str = "sha256") -> str:
    """Hash a string."""
    if algorithm == "sha256":
        return hashlib.sha256(s.encode()).hexdigest()
    elif algorithm == "md5":
        return hashlib.md5(s.encode()).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(s.encode()).hexdigest()
    return s


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def format_time(seconds: float) -> str:
    """Format seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    else:
        return f"{seconds / 86400:.1f}d"


def load_json(path: str) -> Dict:
    """Load JSON from file."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Save data to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def merge_dicts(*dicts: Dict) -> Dict:
    """Merge multiple dictionaries."""
    result = {}
    for d in dicts:
        result.update(d)
    return result


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def retry(max_attempts: int = 3, delay: float = 1.0):
    """Retry decorator."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed: {e}")
                    import time

                    time.sleep(delay)

        return wrapper

    return decorator


class Timer:
    """Simple timer context manager."""

    def __init__(self):
        self.start = None
        self.end = None
        self.elapsed = None

    def __enter__(self):
        import time

        self.start = time.time()
        return self

    def __exit__(self, *args):
        import time

        self.end = time.time()
        self.elapsed = self.end - self.start


class Cache:
    """Simple in-memory cache."""

    def __init__(self, max_size: int = 100):
        self._cache = {}
        self._max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        if len(self._cache) >= self._max_size:
            # Remove oldest
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = value

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


class RateLimiter:
    """Simple rate limiter."""

    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls = []

    def __call__(self, func):
        import time

        def wrapper(*args, **kwargs):
            now = time.time()
            self.calls = [c for c in self.calls if now - c < self.period]

            if len(self.calls) >= self.max_calls:
                raise Exception(
                    f"Rate limit exceeded. Max {self.max_calls} calls per {self.period}s"
                )

            self.calls.append(now)
            return func(*args, **kwargs)

        return wrapper


def validate_config(config: Dict, required_keys: List[str]) -> bool:
    """Validate config has required keys."""
    return all(key in config for key in required_keys)


def get_timestamp() -> str:
    """Get ISO timestamp."""
    return datetime.now().isoformat()


__all__ = [
    "generate_id",
    "hash_string",
    "format_size",
    "format_time",
    "load_json",
    "save_json",
    "merge_dicts",
    "clamp",
    "retry",
    "Timer",
    "Cache",
    "RateLimiter",
    "validate_config",
    "get_timestamp",
    "find_available_port",
]


def find_available_port(host: str = "", start_port: int = 8000, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port.

    Args:
        host: Bind host (empty string for all interfaces)
        start_port: Starting port number
        max_attempts: Maximum number of ports to try

    Returns:
        Available port number

    Raises:
        RuntimeError: If no port is available in range
    """
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.close()
            return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find available port in range {start_port}-{start_port + max_attempts}")
