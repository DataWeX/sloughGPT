"""
CLI helpers — Docker, banner, output, and utility functions.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .framework import (
    BOLD as _BOLD, DIM as _DIM, CYAN as _CYAN,
    GREEN as _GREEN, YELLOW as _YELLOW, RED as _RED,
    echo, click,
)


def ns(**kwargs):
    """Create a SimpleNamespace from keyword arguments."""
    from types import SimpleNamespace
    return SimpleNamespace(**kwargs)


def output(data, *, raw=False, json_output=False, **kwargs):
    """Output data as JSON or formatted text."""
    if json_output or raw:
        echo(json.dumps(data, indent=2, default=str))
        return data
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                echo(f"{_BOLD}{key}:{_RESET}")
                echo(json.dumps(value, indent=2, default=str))
            else:
                echo(f"{_BOLD}{key}:{_RESET} {value}")
    elif isinstance(data, list):
        for item in data:
            echo(f"• {item}")
    else:
        echo(str(data))
    
    return data


def confirm(message, default=False):
    """Ask for confirmation."""
    return click.confirm(message, default=default)


def verbose(message, **kwargs):
    """Print verbose output (only when verbose mode is enabled)."""
    echo(f"{_DIM}[verbose] {message}{_RESET}", **kwargs)


def docker_action(action, service=None, **kwargs):
    """Execute a Docker action (up, down, restart, logs, status)."""
    from .version import format_version_display
    
    docker_compose = os.environ.get("DOCKER_COMPOSE", "docker-compose")
    
    if action == "status":
        cmd = [docker_compose, "ps"]
    elif action == "up":
        cmd = [docker_compose, "up", "-d"]
        if service:
            cmd.append(service)
    elif action == "down":
        cmd = [docker_compose, "down"]
        if service:
            cmd.append(service)
    elif action == "restart":
        cmd = [docker_compose, "restart"]
        if service:
            cmd.append(service)
    elif action == "logs":
        cmd = [docker_compose, "logs", "-f"]
        if service:
            cmd.append(service)
    else:
        echo(f"{_RED}Unknown action: {action}{_RESET}")
        return False
    
    echo(f"{_CYAN}$ {' '.join(cmd)}{_RESET}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        echo(f"{_RED}Docker command failed with exit code {e.returncode}{_RESET}")
        return False
    except FileNotFoundError:
        echo(f"{_RED}docker-compose not found. Install Docker Compose.{_RESET}")
        return False


def show_welcome_banner(version_info=None):
    """Show the welcome banner with version info."""
    from .version import format_version_display
    
    if version_info is None:
        version_info = format_version_display()
    
    echo(f"""
{_BOLD}{_CYAN}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   {_YELLOW}██████╗ ██╗      ██████╗  ██████╗██╗  ██╗{_CYAN}                ║
║   {_YELLOW}██╔══██╗██║     ██╔═══██╗██╔════╝██║ ██╔╝{_CYAN}                ║
║   {_YELLOW}██████╔╝██║     ██║   ██║██║     █████╔╝{_CYAN}                 ║
║   {_YELLOW}██╔══██╗██║     ██║   ██║██║     ██╔═██╗{_CYAN}                 ║
║   {_YELLOW}██████╔╝███████╗╚██████╔╝╚██████╗██║  ██╗{_CYAN}                ║
║   {_YELLOW}╚═════╝ ╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝{_CYAN}                ║
║                                                              ║
║   {_DIM}The local AI platform{_RESET}                                      ║
║   {_DIM}{version_info}{_RESET}                                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{_RESET}
""")


def show_server_status(host="localhost", port=8000):
    """Show the current server status."""
    import urllib.request
    import urllib.error
    
    url = f"http://{host}:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            data = json.loads(response.read())
            echo(f"{_GREEN}✓ Server is running{_RESET} at {url}")
            if isinstance(data, dict):
                for key, value in data.items():
                    echo(f"  {_DIM}{key}: {value}{_RESET}")
            return True
    except (urllib.error.URLError, ConnectionRefusedError, OSError):
        echo(f"{_YELLOW}✗ Server is not running{_RESET} at {url}")
        return False
    except Exception as e:
        echo(f"{_YELLOW}✗ Server status unknown{_RESET}: {e}")
        return False


_RESET = "\033[0m"


__all__ = [
    "ns", "output", "confirm", "verbose",
    "docker_action", "show_welcome_banner", "show_server_status",
]
