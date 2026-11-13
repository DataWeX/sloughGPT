"""Back-compat shim: ``arken`` was renamed to ``avion``.

Import from ``avion`` instead — this shim only re-exports the Avion API
and will be removed in a later release.
"""

from avion import (
    Action,
    ActionType,
    Agent,
    AgentConfig,
    AgentResult,
    Avion,
    AvionConfig,
    Backend,
    BoundingBox,
    Coordinate,
    Element,
    ElementFinder,
    ElementLocator,
    ElementNotFoundError,
    InteractionChain,
    Keyboard,
    KeyModifier,
    Mouse,
    MouseButton,
    NavigationEntry,
    Navigator,
    validate_action,
)

Arken = Avion
ArkenConfig = AvionConfig

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
    "validate_action",
]
