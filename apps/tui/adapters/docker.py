"""Docker compose adapters for TUI Phase 4 (Ops)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class DockerService:
    name: str
    status: str
    ports: str = ""


@dataclass
class DockerStatus:
    running: bool
    services: List[DockerService] = None


class DockerAdapter:
    """Wraps docker compose for TUI Docker ops room."""

    def __init__(self, compose_file: Optional[Path] = None, profile: Optional[str] = None):
        self.compose_file = compose_file
        self.profile = profile

    def _get_compose_file(self) -> Path:
        if self.compose_file:
            return self.compose_file
        return Path("docker-compose.yml")

    def _run(self, args: List[str], check: bool = False) -> subprocess.CompletedProcess:
        cmd = ["docker", "compose", "-f", str(self._get_compose_file())] + args
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=check)
        except FileNotFoundError:
            raise RuntimeError("docker not found. Install Docker first.")

    def start(self, dev: bool = False, gpu: bool = False) -> bool:
        cmd = ["up", "-d"]
        if dev:
            cmd.extend(["--profile", "dev"])
        elif gpu:
            cmd.extend(["--profile", "gpu"])
        try:
            self._run(cmd, check=True)
            return True
        except RuntimeError as e:
            raise e
        except subprocess.CalledProcessError as e:
            print(f"Error: {e.stderr}")
            return False

    def stop(self) -> bool:
        try:
            self._run(["down"], check=True)
            return True
        except RuntimeError as e:
            raise e
        except subprocess.CalledProcessError as e:
            print(f"Error: {e.stderr}")
            return False

    def status(self) -> Optional[str]:
        try:
            result = self._run(["ps"])
            return result.stdout
        except RuntimeError as e:
            raise e

    def logs(self, service: Optional[str] = None, follow: bool = False) -> None:
        cmd = ["logs"]
        if follow:
            cmd.append("-f")
        if service:
            cmd.append(service)
        try:
            self._run(cmd)
        except RuntimeError as e:
            raise e

    def build(self, no_cache: bool = False) -> bool:
        cmd = ["build"]
        if no_cache:
            cmd.append("--no-cache")
        try:
            self._run(cmd, check=True)
            return True
        except RuntimeError as e:
            raise e
        except subprocess.CalledProcessError as e:
            print(f"Error: {e.stderr}")
            return False

    def is_available(self) -> bool:
        try:
            subprocess.run(["docker", "info"], capture_output=True, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False
