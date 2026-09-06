"""
Dynamic shell completion with TTL cache.

Provides cached completions for model names, dataset names, soul names,
and other API-backed values.  Used by both the Click ``completion`` command
and the interactive REPL readline completer.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional


class CompletionCache:
    """TTL cache for dynamic completion values.

    First call fetches from the API; subsequent calls within *ttl_sec*
    return the cached list.  If the API is unreachable, the cache returns
    the last known values (or an empty list on the very first call).
    """

    def __init__(self, ttl_sec: float = 30.0):
        self._ttl = ttl_sec
        self._store: Dict[str, tuple[float, List[str]]] = {}

    def get(self, key: str, fetcher: Callable[[], List[str]]) -> List[str]:
        """Return cached values for *key*, fetching if stale or missing."""
        now = time.monotonic()
        entry = self._store.get(key)
        if entry is not None:
            ts, values = entry
            if now - ts < self._ttl:
                return values
        try:
            values = fetcher()
        except Exception:
            # On error, return stale data if available
            if entry is not None:
                return entry[1]
            values = []
        self._store[key] = (now, values)
        return values

    def invalidate(self, key: Optional[str] = None) -> None:
        """Invalidate one key or all cached values."""
        if key is None:
            self._store.clear()
        else:
            self._store.pop(key, None)


# ── Singleton ──────────────────────────────────────────────────────────

_cache = CompletionCache(ttl_sec=30.0)


def get_cache() -> CompletionCache:
    """Return the global completion cache."""
    return _cache


# ── Fetchers (API-backed) ─────────────────────────────────────────────

def _fetch_models(host: str = "localhost", port: int = 8000) -> List[str]:
    """Fetch available model names from the API."""
    import requests
    base = f"http://{host}:{port}"
    try:
        r = requests.get(f"{base}/models", timeout=3)
        if r.status_code == 200:
            data = r.json()
            models = data if isinstance(data, list) else data.get("models", [])
            return [m.get("name", m.get("id", "")) for m in models if isinstance(m, dict)]
    except requests.RequestException:
        pass
    return []


def _fetch_hf_models(host: str = "localhost", port: int = 8000) -> List[str]:
    """Fetch HuggingFace model names from the API."""
    import requests
    base = f"http://{host}:{port}"
    try:
        r = requests.get(f"{base}/models/hf", timeout=5)
        if r.status_code == 200:
            data = r.json()
            models = data if isinstance(data, list) else data.get("models", [])
            return [m.get("id", m.get("name", "")) for m in models if isinstance(m, dict)]
    except requests.RequestException:
        pass
    return []


def _fetch_souls(host: str = "localhost", port: int = 8000) -> List[str]:
    """Fetch soul/personality names from the API."""
    import requests
    base = f"http://{host}:{port}"
    try:
        r = requests.get(f"{base}/souls", timeout=3)
        if r.status_code == 200:
            data = r.json()
            souls = data if isinstance(data, list) else data.get("souls", [])
            return [s.get("name", "") for s in souls if isinstance(s, dict)]
    except requests.RequestException:
        pass
    return []


def _fetch_datasets(host: str = "localhost", port: int = 8000) -> List[str]:
    """Fetch dataset names from the API."""
    import requests
    base = f"http://{host}:{port}"
    try:
        r = requests.get(f"{base}/datasets", timeout=3)
        if r.status_code == 200:
            data = r.json()
            datasets = data if isinstance(data, list) else data.get("datasets", [])
            return [d.get("name", "") for d in datasets if isinstance(d, dict)]
    except requests.RequestException:
        pass
    return []


def _fetch_checkpoints(host: str = "localhost", port: int = 8000) -> List[str]:
    """Fetch checkpoint names from the API."""
    import requests
    base = f"http://{host}:{port}"
    try:
        r = requests.get(f"{base}/training/checkpoints", timeout=3)
        if r.status_code == 200:
            data = r.json()
            cps = data if isinstance(data, list) else data.get("checkpoints", [])
            return [c.get("name", "") for c in cps if isinstance(c, dict)]
    except requests.RequestException:
        pass
    return []


# ── Public API ─────────────────────────────────────────────────────────

def complete_models(host: str = "localhost", port: int = 8000) -> List[str]:
    """Complete model names (cached)."""
    return _cache.get("models", lambda: _fetch_models(host, port))


def complete_hf_models(host: str = "localhost", port: int = 8000) -> List[str]:
    """Complete HuggingFace model names (cached)."""
    return _cache.get("hf_models", lambda: _fetch_hf_models(host, port))


def complete_souls(host: str = "localhost", port: int = 8000) -> List[str]:
    """Complete soul names (cached)."""
    return _cache.get("souls", lambda: _fetch_souls(host, port))


def complete_datasets(host: str = "localhost", port: int = 8000) -> List[str]:
    """Complete dataset names (cached)."""
    return _cache.get("datasets", lambda: _fetch_datasets(host, port))


def complete_checkpoints(host: str = "localhost", port: int = 8000) -> List[str]:
    """Complete checkpoint names (cached)."""
    return _cache.get("checkpoints", lambda: _fetch_checkpoints(host, port))


def complete_paths(prefix: str) -> List[str]:
    """Complete file system paths (no cache needed)."""
    from pathlib import Path
    if not prefix or prefix in (".", ".."):
        search_dir = Path(".")
        partial = prefix
    else:
        p = Path(prefix)
        if prefix.endswith("/"):
            search_dir = p
            partial = ""
        else:
            search_dir = p.parent
            partial = p.name
        if not search_dir.exists():
            return []
    try:
        candidates = []
        for entry in search_dir.iterdir():
            name = entry.name
            if name.startswith("."):
                continue
            if name.startswith(partial):
                suffix = "/" if entry.is_dir() else ""
                candidates.append(str(entry) + suffix)
        return sorted(candidates)
    except PermissionError:
        return []


# ── Command → completer mapping ───────────────────────────────────────

COMMAND_COMPLETERS: Dict[str, Callable[[], List[str]]] = {
    "load": lambda: complete_models(),
    "unload": lambda: complete_models(),
    "gen": lambda: complete_models(),
    "protect": lambda: complete_models(),
    "unprotect": lambda: complete_models(),
    "switch": lambda: complete_souls(),
    "datasets": lambda: complete_datasets(),
    "dataset": lambda: complete_datasets(),
    "checkpoints": lambda: complete_checkpoints(),
    "train": lambda: ["status", "follow", "stop", "distill", "hf", "auto", "load", "del"],
    "model": lambda: ["list", "status", "info", "download", "convert", "compare"],
}


def get_completions_for_command(cmd: str, host: str = "localhost", port: int = 8000) -> List[str]:
    """Return dynamic completion candidates for a command's arguments."""
    fetcher = COMMAND_COMPLETERS.get(cmd)
    if fetcher:
        return fetcher()
    return complete_paths("")
