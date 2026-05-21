"""
Version tracking for SloughGPT CLI.

Provides version information and update checking.
"""
import json
import os
from pathlib import Path
from typing import Optional


__version__ = "0.1.0"
__version_tuple__ = (0, 1, 0)

CLI_NAME = "sloughgpt"
REPO_URL = "https://github.com/anomalyco/sloughgpt"


class VersionInfo:
    """Version information and metadata."""

    def __init__(
        self,
        version: str = __version__,
        python_version: Optional[str] = None,
        platform: Optional[str] = None,
    ):
        self.version = version
        self.python_version = python_version or _get_python_version()
        self.platform = platform or _get_platform()

    def __str__(self) -> str:
        return f"{CLI_NAME} v{self.version}"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "cli": CLI_NAME,
            "version": self.version,
            "python": self.python_version,
            "platform": self.platform,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def _get_python_version() -> str:
    """Get Python version string."""
    import sys
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _get_platform() -> str:
    """Get platform string."""
    import platform
    return f"{platform.system()} {platform.machine()}"


def get_version() -> VersionInfo:
    """Get current CLI version info."""
    return VersionInfo()


def check_for_updates() -> Optional[str]:
    """Check for CLI updates (placeholder for future implementation).

    Returns:
        New version string if available, None otherwise
    """
    # TODO: Implement update checking against PyPI or GitHub releases
    return None


def format_version_display() -> str:
    """Format version for display in help/about."""
    info = get_version()
    return f"{CLI_NAME} v{info.version} ({info.platform}, Python {info.python_version})"
