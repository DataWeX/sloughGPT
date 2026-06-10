"""
Window Manager — i3/dwm-style tiling terminal WM with workspaces,
pane shell mode, and keyboard-driven navigation.

Package structure:
  core.py       — Pane, Workspace, LayoutType, layout computation
  shell_mode.py — PaneShell: interactive REPL in a pane
  commands.py   — WMCommands: command pattern dispatch
  renderer.py   — WMRenderer: curses rendering
  manager.py    — WindowManager: orchestrator
"""

from .core import Pane, Workspace, LayoutType, compute_tiled_rects
from .shell_mode import PaneShell
from .commands import WMCommands
from .renderer import WMRenderer
from .manager import WindowManager

__all__ = [
    "Pane",
    "Workspace",
    "LayoutType",
    "compute_tiled_rects",
    "PaneShell",
    "WMCommands",
    "WMRenderer",
    "WindowManager",
]
