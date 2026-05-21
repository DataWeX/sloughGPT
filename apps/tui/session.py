"""Persistent TUI session: repository root and API attachment (Phase 1).

Supports save/restore to JSON for persisting session state across launches.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


def _is_sloughgpt_repo_root(path: Path) -> bool:
    pyproject = path / "pyproject.toml"
    if pyproject.is_file():
        try:
            text = pyproject.read_text(encoding="utf-8")
        except OSError:
            return False
        if 'name = "sloughgpt"' in text or "name='sloughgpt'" in text:
            return True
    if (path / "config.yaml").is_file() and (path / "packages" / "core-py").is_dir():
        return True
    return False


def discover_repo_root(start: Optional[Path] = None, *, max_depth: int = 32) -> Optional[Path]:
    """Walk parents from ``start`` (default: cwd) until a SloughGPT repo root is found."""
    cur = (start or Path.cwd()).resolve()
    for _ in range(max_depth):
        if _is_sloughgpt_repo_root(cur):
            return cur
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return None


@dataclass
class TuiSession:
    """One interactive session: where the repo lives and which API to talk to.

    Persists ``last_checkpoint``, ``last_soul_path``, ``last_job_id``,
    ``api_host``, ``api_port``, and ``device`` to ``~/.sloughgpt/tui_session.json``
    so the TUI resumes where you left off across launches.
    """

    repo_root: Path
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    device: Optional[str] = None
    last_checkpoint: Optional[Path] = None
    last_soul_path: Optional[Path] = None
    last_job_id: Optional[str] = None
    last_error: Optional[str] = None
    meta: dict = field(default_factory=dict)

    _STATE_DIR: Path = Path.home() / ".sloughgpt"
    _STATE_FILE: Path = _STATE_DIR / "tui_session.json"

    def __post_init__(self) -> None:
        from apps.tui.adapters.http_api import TuiApiClient
        self._api_client: Optional[TuiApiClient] = None

    @property
    def api_base_url(self) -> str:
        return f"http://{self.api_host}:{self.api_port}"

    @property
    def api_client(self) -> TuiApiClient:
        if self._api_client is None:
            from apps.tui.adapters.http_api import TuiApiClient
            self._api_client = TuiApiClient(host=self.api_host, port=self.api_port)
        return self._api_client

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write session state to ``~/.sloughgpt/tui_session.json``."""
        self._STATE_DIR.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d.pop("_api_client", None)
        for key in ("repo_root", "last_checkpoint", "last_soul_path"):
            v = d.get(key)
            if v is not None:
                d[key] = str(v)
        with open(self._STATE_FILE, "w") as f:
            json.dump(d, f, indent=2, default=str)

    @classmethod
    def load(cls, repo_root: Path) -> "TuiSession":
        """Restore session state from disk, falling back to defaults.

        Args:
            repo_root: Resolved repo root path (always required).

        Returns:
            TuiSession with persisted fields merged over defaults.
        """
        session = cls(repo_root=repo_root)
        if cls._STATE_FILE.exists():
            try:
                data = json.loads(cls._STATE_FILE.read_text())
                for key in ("api_host", "api_port", "device", "last_job_id"):
                    if key in data:
                        setattr(session, key, data[key])
                for key in ("last_checkpoint", "last_soul_path"):
                    v = data.get(key)
                    if v:
                        p = Path(v)
                        if p.exists():
                            setattr(session, key, p)
                if "meta" in data and isinstance(data["meta"], dict):
                    session.meta = data["meta"]
            except Exception:
                pass  # Corrupted state file — use defaults
        return session

    def forget_error(self) -> None:
        """Clear the last error so it doesn't show on next launch."""
        self.last_error = None
