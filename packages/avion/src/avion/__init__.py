"""Avion — web autoclicker/agent: find, click, type, wait, report.

Usage::

    async with Avion(base_url="http://localhost:5175") as a:
        await a.goto("/chat")
        await a.click_text("Start Training")
        await a.fill(ElementLocator.css("input[name=q]"), "hello")

(Formerly voyager → arken; ``Arken``/``ArkenConfig`` remain as aliases.)
"""

from __future__ import annotations

from avion.ai.agent import Agent, AgentConfig, AgentResult
from avion.ai.models import Action, ActionType, validate_action
from avion.core.element import (
    Backend,
    Element,
    ElementFinder,
    ElementLocator,
    ElementNotFoundError,
    PageControls,
)
from avion.core.navigator import NavigationEntry, Navigator
from avion.core.session import Arken, ArkenConfig

# Canonical names (arken shim re-exports these).
Avion = Arken
AvionConfig = ArkenConfig
from avion.interact.primitives import (
    BoundingBox,
    Coordinate,
    InteractionChain,
    Keyboard,
    KeyModifier,
    Mouse,
    MouseButton,
)
from avion.sync import SyncRunner, get_default_runner

__all__ = [
    "Action",
    "ActionType",
    "Agent",
    "AgentConfig",
    "AgentResult",
    "Arken",
    "ArkenConfig",
    "Avion",
    "AvionConfig",
    "Backend",
    "BoundingBox",
    "Coordinate",
    "Element",
    "ElementFinder",
    "ElementLocator",
    "ElementNotFoundError",
    "InteractionChain",
    "Keyboard",
    "KeyModifier",
    "Mouse",
    "MouseButton",
    "NavigationEntry",
    "Navigator",
    "PageControls",
    "SyncRunner",
    "get_default_runner",
    "validate_action",
]
