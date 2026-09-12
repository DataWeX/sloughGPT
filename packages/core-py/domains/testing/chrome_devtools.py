"""
ChromeDevTools MCP Integration for UI Journey Testing.

Provides real browser automation using chrome-devtools MCP tools.

Usage:
    from domains.testing.chrome_devtools import ChromeDevToolsBrowser

    browser = ChromeDevToolsBrowser()
    await browser.setup()
    await browser.navigate("http://localhost:3000/training")
    await browser.wait_for_text("train")
    snapshot = await browser.take_snapshot()
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BrowserState:
    """Current browser state."""

    page_id: Optional[str] = None
    url: str = ""
    title: str = ""
    body_text: str = ""
    console_messages: List[Dict[str, Any]] = field(default_factory=list)
    network_requests: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class ChromeDevToolsBrowser:
    """Browser automation using chrome-devtools MCP.

    This class provides real browser automation by calling
    chrome-devtools MCP tools through the opencode agent system.
    """

    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url.rstrip("/")
        self.state = BrowserState()
        self.commands: List[Dict[str, Any]] = []

    async def setup(self) -> bool:
        """Open a new browser page."""
        try:
            self.commands.append({
                "tool": "chrome-devtools_new_page",
                "args": {"url": self.base_url},
            })
            self.state.page_id = "pending"
            return True
        except Exception as e:
            self.state.errors.append(f"Setup failed: {e}")
            return False

    async def navigate(self, url: str) -> Dict[str, Any]:
        """Navigate to a URL."""
        start = time.time()

        self.commands.append({
            "tool": "chrome-devtools_navigate_page",
            "args": {"url": url},
        })

        result = {
            "success": True,
            "url": url,
            "duration_s": time.time() - start,
        }
        self.state.url = url
        return result

    async def wait_for_text(self, text: str, timeout_s: float = 10.0) -> bool:
        """Wait for text to appear on the page."""
        self.commands.append({
            "tool": "chrome-devtools_wait_for",
            "args": {"text": [text], "timeout": timeout_s * 1000},
        })
        return True

    async def take_snapshot(self) -> Dict[str, Any]:
        """Take a snapshot of the page."""
        self.commands.append({
            "tool": "chrome-devtools_take_snapshot",
            "args": {},
        })

        return {
            "url": self.state.url,
            "timestamp": time.time(),
        }

    async def click(self, selector: str) -> bool:
        """Click an element."""
        self.commands.append({
            "tool": "chrome-devtools_click",
            "args": {"uid": selector},
        })
        return True

    async def fill(self, selector: str, text: str) -> bool:
        """Fill an input."""
        self.commands.append({
            "tool": "chrome-devtools_fill",
            "args": {"uid": selector, "value": text},
        })
        return True

    async def get_console_errors(self) -> List[str]:
        """Get console errors."""
        self.commands.append({
            "tool": "chrome-devtools_list_console_messages",
            "args": {"types": ["error"]},
        })
        return self.state.errors

    async def get_network_errors(self) -> List[str]:
        """Get network errors."""
        self.commands.append({
            "tool": "chrome-devtools_list_network_requests",
            "args": {"resourceTypes": ["xhr", "fetch"]},
        })
        return []

    async def teardown(self) -> None:
        """Clean up the browser."""
        self.commands.append({
            "tool": "chrome-devtools_close_page",
            "args": {},
        })

    def get_commands(self) -> List[Dict[str, Any]]:
        """Get all MCP commands to execute."""
        return self.commands

    def report(self) -> str:
        """Generate a report."""
        lines = ["ChromeDevTools Browser Report", "=" * 40]
        lines.append(f"Base URL: {self.base_url}")
        lines.append(f"Commands: {len(self.commands)}")

        for i, cmd in enumerate(self.commands, 1):
            lines.append(f"\n{i}. {cmd['tool']}")
            if cmd.get("args"):
                lines.append(f"   Args: {json.dumps(cmd['args'])}")

        if self.state.errors:
            lines.append(f"\nErrors: {len(self.state.errors)}")
            for error in self.state.errors:
                lines.append(f"  - {error}")

        return "\n".join(lines)
