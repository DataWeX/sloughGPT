"""Low-level interaction primitives.

Provides coordinate-based mouse, keyboard, and gesture actions that work
at the OS/display level — not tied to DOM elements. These are the building
blocks for computer-use style automation.

Usage::

    from arken.interact import Mouse, Keyboard, Coordinate

    mouse = Mouse(backend)
    await mouse.move_to(Coordinate(500, 300))
    await mouse.click()
    await mouse.double_click()

    kb = Keyboard(backend)
    await kb.type_text("hello world")
    await kb.press("Control+a")
    await kb.press("Enter")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Coordinate:
    """A 2D screen coordinate."""

    x: float
    y: float

    def distance_to(self, other: Coordinate) -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def midpoint(self, other: Coordinate) -> Coordinate:
        return Coordinate(x=(self.x + other.x) / 2, y=(self.y + other.y) / 2)

    def offset(self, dx: float = 0, dy: float = 0) -> Coordinate:
        return Coordinate(x=self.x + dx, y=self.y + dy)

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True)
class BoundingBox:
    """A rectangular region on screen."""

    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Coordinate:
        return Coordinate(x=self.x + self.width / 2, y=self.y + self.height / 2)

    @property
    def top_left(self) -> Coordinate:
        return Coordinate(x=self.x, y=self.y)

    @property
    def top_right(self) -> Coordinate:
        return Coordinate(x=self.x + self.width, y=self.y)

    @property
    def bottom_left(self) -> Coordinate:
        return Coordinate(x=self.x, y=self.y + self.height)

    @property
    def bottom_right(self) -> Coordinate:
        return Coordinate(x=self.x + self.width, y=self.y + self.height)

    def contains(self, point: Coordinate) -> bool:
        return (
            self.x <= point.x <= self.x + self.width and self.y <= point.y <= self.y + self.height
        )

    def overlaps(self, other: BoundingBox) -> bool:
        return not (
            self.x + self.width < other.x
            or other.x + other.width < self.x
            or self.y + self.height < other.y
            or other.y + other.height < self.y
        )

    def expand(self, px: float = 0, py: float = 0) -> BoundingBox:
        return BoundingBox(
            x=self.x - px,
            y=self.y - py,
            width=self.width + 2 * px,
            height=self.height + 2 * py,
        )

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


class MouseButton(Enum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class KeyModifier(Enum):
    CONTROL = "Control"
    ALT = "Alt"
    SHIFT = "Shift"
    META = "Meta"


@dataclass
class KeyEvent:
    """A keyboard event."""

    key: str
    modifiers: list[KeyModifier] = field(default_factory=list)
    hold_ms: float = 0.0


@runtime_checkable
class InteractionBackend(Protocol):
    """Backend protocol for low-level interactions."""

    async def mouse_move(self, x: float, y: float) -> None: ...
    async def mouse_down(self, button: MouseButton = MouseButton.LEFT) -> None: ...
    async def mouse_up(self, button: MouseButton = MouseButton.LEFT) -> None: ...
    async def mouse_click(
        self, x: float, y: float, button: MouseButton = MouseButton.LEFT
    ) -> None: ...
    async def mouse_double_click(self, x: float, y: float) -> None: ...
    async def mouse_scroll(self, x: float, y: float, delta_x: int, delta_y: int) -> None: ...
    async def keyboard_down(self, key: str) -> None: ...
    async def keyboard_up(self, key: str) -> None: ...
    async def keyboard_press(self, key: str, hold_ms: float = 0) -> None: ...
    async def keyboard_type(self, text: str, delay_ms: float = 0) -> None: ...
    async def get_viewport_size(self) -> tuple[int, int]: ...
    async def get_screenshot_as_bytes(self) -> bytes: ...


class Mouse:
    """High-level mouse interaction controller.

    Provides coordinate-based mouse actions with easing, recording,
    and replay capabilities.
    """

    def __init__(self, backend: InteractionBackend):
        self._backend = backend
        self._position = Coordinate(x=0, y=0)
        self._history: list[tuple[str, Coordinate, float]] = []

    @property
    def position(self) -> Coordinate:
        return self._position

    @property
    def history(self) -> list[tuple[str, Coordinate, float]]:
        return list(self._history)

    async def move_to(self, target: Coordinate, steps: int = 1) -> None:
        """Move mouse to target coordinate with optional interpolation."""
        start = self._position
        for i in range(1, steps + 1):
            t = i / steps
            x = start.x + (target.x - start.x) * t
            y = start.y + (target.y - start.y) * t
            await self._backend.mouse_move(x, y)
            self._position = Coordinate(x=x, y=y)
        self._history.append(("move", target, time.time()))

    async def click(
        self, coord: Coordinate | None = None, button: MouseButton = MouseButton.LEFT
    ) -> None:
        """Click at coordinate (or current position)."""
        target = coord or self._position
        if coord:
            await self.move_to(coord)
        await self._backend.mouse_click(target.x, target.y, button)
        self._history.append(("click", target, time.time()))

    async def double_click(self, coord: Coordinate | None = None) -> None:
        target = coord or self._position
        if coord:
            await self.move_to(coord)
        await self._backend.mouse_double_click(target.x, target.y)
        self._history.append(("double_click", target, time.time()))

    async def right_click(self, coord: Coordinate | None = None) -> None:
        await self.click(coord, MouseButton.RIGHT)

    async def middle_click(self, coord: Coordinate | None = None) -> None:
        await self.click(coord, MouseButton.MIDDLE)

    async def drag(self, start: Coordinate, end: Coordinate, steps: int = 20) -> None:
        """Drag from start to end."""
        await self.move_to(start, steps=1)
        await self._backend.mouse_down(MouseButton.LEFT)
        await self.move_to(end, steps=steps)
        await self._backend.mouse_up(MouseButton.LEFT)
        self._history.append(("drag", start, time.time()))

    async def scroll(self, x: float, y: float, delta_x: int = 0, delta_y: int = -3) -> None:
        """Scroll at position."""
        await self._backend.mouse_scroll(x, y, delta_x, delta_y)
        self._history.append(("scroll", Coordinate(x=x, y=y), time.time()))

    async def hover(self, coord: Coordinate) -> None:
        """Hover over coordinate."""
        await self.move_to(coord)
        self._history.append(("hover", coord, time.time()))


class Keyboard:
    """High-level keyboard interaction controller.

    Supports text typing, key combos, and key sequences.
    """

    def __init__(self, backend: InteractionBackend):
        self._backend = backend
        self._history: list[tuple[str, float]] = []

    @property
    def history(self) -> list[tuple[str, float]]:
        return list(self._history)

    async def type_text(self, text: str, delay_ms: float = 0) -> None:
        """Type text character by character."""
        await self._backend.keyboard_type(text, delay_ms)
        self._history.append((f"type:{text[:50]}", time.time()))

    async def press(self, key: str, hold_ms: float = 0) -> None:
        """Press a key or key combo (e.g. 'Control+a', 'Enter')."""
        await self._backend.keyboard_press(key, hold_ms)
        self._history.append((f"press:{key}", time.time()))

    async def hotkey(self, *keys: str) -> None:
        """Press a key combination (e.g. hotkey('Control', 'Shift', 'p'))."""
        for key in keys[:-1]:
            await self._backend.keyboard_down(key)
        await self._backend.keyboard_press(keys[-1])
        for key in reversed(keys[:-1]):
            await self._backend.keyboard_up(key)
        self._history.append((f"hotkey:{'+'.join(keys)}", time.time()))

    async def type_slowly(self, text: str, char_delay_ms: float = 50) -> None:
        """Type text with a delay between characters (for慢速 input)."""
        for char in text:
            await self._backend.keyboard_type(char, 0)
            if char_delay_ms > 0:
                import asyncio

                await asyncio.sleep(char_delay_ms / 1000)
        self._history.append((f"type_slow:{text[:50]}", time.time()))

    async def clear_field(self) -> None:
        """Select all and delete to clear a field."""
        await self.hotkey("Control", "a")
        await self.press("Backspace")
        self._history.append(("clear_field", time.time()))


class InteractionChain:
    """Chain multiple interactions for replay.

    Usage::

        chain = InteractionChain()
        chain.add_move(Coordinate(100, 200))
        chain.add_click()
        chain.add_type("hello")
        chain.add_press("Enter")

        await chain.replay(mouse, keyboard)
    """

    def __init__(self):
        self._actions: list[tuple[str, dict[str, Any]]] = []

    def add_move(self, target: Coordinate, steps: int = 1) -> InteractionChain:
        self._actions.append(("move", {"target": target, "steps": steps}))
        return self

    def add_click(self, coord: Coordinate | None = None) -> InteractionChain:
        self._actions.append(("click", {"coord": coord}))
        return self

    def add_double_click(self, coord: Coordinate | None = None) -> InteractionChain:
        self._actions.append(("double_click", {"coord": coord}))
        return self

    def add_right_click(self, coord: Coordinate | None = None) -> InteractionChain:
        self._actions.append(("right_click", {"coord": coord}))
        return self

    def add_type(self, text: str, delay_ms: float = 0) -> InteractionChain:
        self._actions.append(("type", {"text": text, "delay_ms": delay_ms}))
        return self

    def add_press(self, key: str) -> InteractionChain:
        self._actions.append(("press", {"key": key}))
        return self

    def add_hotkey(self, *keys: str) -> InteractionChain:
        self._actions.append(("hotkey", {"keys": keys}))
        return self

    def add_drag(self, start: Coordinate, end: Coordinate) -> InteractionChain:
        self._actions.append(("drag", {"start": start, "end": end}))
        return self

    def add_scroll(self, x: float, y: float, delta_y: int = -3) -> InteractionChain:
        self._actions.append(("scroll", {"x": x, "y": y, "delta_y": delta_y}))
        return self

    def add_wait(self, ms: float) -> InteractionChain:
        self._actions.append(("wait", {"ms": ms}))
        return self

    def add_custom(self, action: str, **kwargs: Any) -> InteractionChain:
        self._actions.append((action, kwargs))
        return self

    @property
    def actions(self) -> list[tuple[str, dict[str, Any]]]:
        return list(self._actions)

    @property
    def length(self) -> int:
        return len(self._actions)

    async def replay(
        self,
        mouse: Mouse,
        keyboard: Keyboard,
        on_action: callable = None,
    ) -> list[dict[str, Any]]:
        """Replay the chain against mouse and keyboard controllers."""
        import asyncio

        results = []
        for action, params in self._actions:
            result = {"action": action, "params": params, "success": False, "error": ""}
            try:
                if action == "move":
                    await mouse.move_to(params["target"], params.get("steps", 1))
                elif action == "click":
                    await mouse.click(params.get("coord"))
                elif action == "double_click":
                    await mouse.double_click(params.get("coord"))
                elif action == "right_click":
                    await mouse.right_click(params.get("coord"))
                elif action == "type":
                    await keyboard.type_text(params["text"], params.get("delay_ms", 0))
                elif action == "press":
                    await keyboard.press(params["key"])
                elif action == "hotkey":
                    await keyboard.hotkey(*params["keys"])
                elif action == "drag":
                    await mouse.drag(params["start"], params["end"])
                elif action == "scroll":
                    await mouse.scroll(params["x"], params["y"], delta_y=params.get("delta_y", -3))
                elif action == "wait":
                    await asyncio.sleep(params["ms"] / 1000)
                result["success"] = True
            except Exception as e:
                result["error"] = str(e)
            results.append(result)
            if on_action:
                on_action(result)
        return results

    def to_dict(self) -> list[dict[str, Any]]:
        return [{"action": a, "params": p} for a, p in self._actions]

    @classmethod
    def from_dict(cls, data: list[dict[str, Any]]) -> InteractionChain:
        chain = cls()
        for item in data:
            chain._actions.append((item["action"], item.get("params", {})))
        return chain
