"""Back-compat shim: ``voyager`` was renamed to ``arken``.

Import from ``arken`` instead — this shim only re-exports the Arken API
and will be removed in a later release.
"""

from arken import (
    Arken,
    ArkenConfig,
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
)

Voyager = Arken
VoyagerConfig = ArkenConfig

__all__ = [
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
    "Voyager",
    "VoyagerConfig",
]
