"""
Validator (Doctor) - Environment and project validation.

Checks for common issues before running commands.
"""
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CheckResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    message: str = ""
    suggestion: str = ""

    def __str__(self) -> str:
        status = "ok" if self.passed else "err"
        return f"[{status}] {self.name}: {self.message}"


@dataclass
class ValidationResult:
    """Overall validation result."""
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if c.message.startswith("Warning"))

    def add(self, check: CheckResult):
        self.checks.append(check)

    def add_pass(self, name: str, message: str = "OK"):
        self.checks.append(CheckResult(name=name, passed=True, message=message))

    def add_fail(self, name: str, message: str, suggestion: str = ""):
        self.checks.append(
            CheckResult(name=name, passed=False, message=message, suggestion=suggestion)
        )

    def add_warn(self, name: str, message: str, suggestion: str = ""):
        self.checks.append(
            CheckResult(name=name, passed=True, message=f"Warning: {message}", suggestion=suggestion)
        )


class Doctor:
    """Environment and project validator."""

    def __init__(self, root_dir: Optional[Path] = None):
        self.root = root_dir or Path.cwd()
        self.result = ValidationResult()

    def run_all(self) -> ValidationResult:
        """Run all validation checks."""
        self._check_python_version()
        self._check_required_dirs()
        self._check_required_packages()
        self._check_disk_space()
        self._check_accelerator()
        self._check_api_server()
        self._check_env_file()
        return self.result

    def _check_python_version(self):
        """Check Python version meets minimum requirements."""
        major, minor = sys.version_info.major, sys.version_info.minor
        if major >= 3 and minor >= 9:
            self.result.add_pass("Python", f"{major}.{minor}.{sys.version_info.micro}")
        else:
            self.result.add_fail(
                "Python",
                f"Version {major}.{minor} unsupported",
                "Python 3.9+ is required",
            )

    def _check_required_dirs(self):
        """Check required directories exist."""
        for dir_name in ["models", "data"]:
            path = self.root / dir_name
            if path.exists():
                self.result.add_pass(dir_name)
            else:
                self.result.add_fail(
                    dir_name,
                    "Directory missing",
                    f"Run 'mkdir {dir_name}' or 'cli.py setup'",
                )

    def _check_required_packages(self):
        """Check critical Python packages are importable."""
        critical = ["numpy", "click"]
        optional = ["torch", "transformers", "fastapi", "uvicorn", "psutil"]
        for pkg in critical:
            try:
                mod = __import__(pkg)
                ver = getattr(mod, "__version__", "?")
                self.result.add_pass(f"pkg:{pkg}", ver)
            except ImportError:
                self.result.add_fail(
                    f"pkg:{pkg}",
                    "Not installed",
                    f"pip install {pkg}",
                )
        for pkg in optional:
            try:
                mod = __import__(pkg)
                ver = getattr(mod, "__version__", "?")
                self.result.add_pass(f"pkg:{pkg}", ver)
            except ImportError:
                self.result.add_warn(
                    f"pkg:{pkg}",
                    "Not installed (optional)",
                    f"pip install {pkg}",
                )

    def _check_disk_space(self):
        """Check available disk space."""
        try:
            import shutil
            usage = shutil.disk_usage(str(self.root))
            free_gb = usage.free / (1024 ** 3)
            if free_gb >= 5:
                self.result.add_pass("Disk", f"{free_gb:.1f} GB free")
            elif free_gb >= 1:
                self.result.add_warn("Disk", f"{free_gb:.1f} GB free (low)")
            else:
                self.result.add_fail(
                    "Disk",
                    f"{free_gb:.1f} GB free (critical)",
                    "Free up disk space before training",
                )
        except Exception:
            self.result.add_warn("Disk", "Could not determine free space")

    def _check_accelerator(self):
        """Check SloNet accelerator availability."""
        try:
            from domains.training.slonet import _get_accelerator
            acc = _get_accelerator()
            if acc is not None:
                self.result.add_pass("Accelerator", f"{acc.name}")
            else:
                self.result.add_pass("Accelerator", "CPU (numpy)")
        except Exception:
            self.result.add_warn("Accelerator", "Could not probe")

    def _check_api_server(self):
        """Check if API server is reachable."""
        try:
            import requests
            r = requests.get("http://localhost:8000/health", timeout=2)
            if r.status_code == 200:
                self.result.add_pass("API Server", "Running on :8000")
            else:
                self.result.add_warn("API Server", f"Status {r.status_code}")
        except Exception:
            self.result.add_warn(
                "API Server",
                "Not reachable",
                "Run 'cli.py dev' to start",
            )

    def _check_env_file(self):
        """Check .env file exists."""
        env_path = self.root / ".env"
        if env_path.exists():
            self.result.add_pass(".env", "Found")
        else:
            self.result.add_warn(".env", "Not found", "Copy from .env.example if available")
