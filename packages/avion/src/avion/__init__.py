"""Arken — minimal web autoclicker: find, click, type.

Usage::

    async with Arken(base_url="http://localhost:3000") as a:
        await a.goto("/chat")
        await a.click_text("Start Training")
        await a.fill(ElementLocator.css("input[name=q]"), "hello")
"""

from avion.ai.agent import Agent, AgentConfig, AgentResult
from avion.ai.models import Action, ActionType, validate_action
from avion.core.element import (
    Backend,
    Element,
    ElementFinder,
    ElementLocator,
    ElementNotFoundError,
)
from avion.core.navigator import NavigationEntry, Navigator
from avion.core.session import Arken, ArkenConfig
from avion.interact.primitives import (
    BoundingBox,
    Coordinate,
    InteractionChain,
    Keyboard,
    KeyModifier,
    Mouse,
    MouseButton,
)

__all__ = [
    "Action",
    "ActionType",
    "Agent",
    "AgentConfig",
    "AgentResult",
    "Arken",
    "ArkenConfig",
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
    "validate_action",
]
