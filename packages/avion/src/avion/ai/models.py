"""AI model interface for computer-use action selection.

An abstract model interface plus test/compound implementations for using
LLMs to decide actions from screen state. Real LLM backends plug in via
the VisionModel protocol.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class ActionType(Enum):
    """Actions an AI agent can take on a computer."""

    MOUSE_MOVE = "mouse_move"
    MOUSE_CLICK = "mouse_click"
    MOUSE_DOUBLE_CLICK = "mouse_double_click"
    MOUSE_RIGHT_CLICK = "mouse_right_click"
    MOUSE_DRAG = "mouse_drag"
    MOUSE_SCROLL = "mouse_scroll"
    KEYBOARD_TYPE = "keyboard_type"
    KEYBOARD_PRESS = "keyboard_press"
    KEYBOARD_HOTKEY = "keyboard_hotkey"
    SCREENSHOT = "screenshot"
    WAIT = "wait"
    NAVIGATE = "navigate"
    TOOL_CALL = "tool_call"
    DONE = "done"
    FAIL = "fail"


@dataclass(frozen=True)
class Action:
    """A single action to execute."""

    action_type: ActionType
    params: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "params": self.params,
            "reasoning": self.reasoning,
            "confidence": round(self.confidence, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Action:
        return cls(
            action_type=ActionType(data["action_type"]),
            params=data.get("params", {}),
            reasoning=data.get("reasoning", ""),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class Step:
    """One agent step: observation → action → result."""

    step_number: int
    screenshot_b64: str = ""
    accessibility_tree: dict[str, Any] | None = None
    action: Action | None = None
    observation: str = ""
    reward: float = 0.0
    done: bool = False
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "action": self.action.to_dict() if self.action else None,
            "observation": self.observation,
            "reward": self.reward,
            "done": self.done,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class Trajectory:
    """A complete agent trajectory (sequence of steps)."""

    task: str
    steps: list[Step] = field(default_factory=list)
    success: bool = False
    total_reward: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: Step) -> None:
        self.steps.append(step)
        self.total_reward += step.reward
        if step.done:
            self.success = step.reward > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "steps": [s.to_dict() for s in self.steps],
            "success": self.success,
            "total_reward": round(self.total_reward, 4),
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        return (
            f"Trajectory('{self.task}'): {len(self.steps)} steps, "
            f"reward={self.total_reward:.2f}, success={self.success}"
        )


# ── Tool cards ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ActionCard:
    """Documents one action for models and validation."""

    action_type: ActionType
    description: str
    params: dict[str, str] = field(default_factory=dict)
    example: dict[str, Any] = field(default_factory=dict)


ACTION_CARDS: dict[ActionType, ActionCard] = {
    ActionType.MOUSE_MOVE: ActionCard(
        ActionType.MOUSE_MOVE,
        "Move the cursor without clicking.",
        {"x": "horizontal pixel", "y": "vertical pixel"},
        {"x": 500, "y": 300},
    ),
    ActionType.MOUSE_CLICK: ActionCard(
        ActionType.MOUSE_CLICK,
        "Click at a coordinate.",
        {"x": "horizontal pixel", "y": "vertical pixel"},
        {"x": 500, "y": 300},
    ),
    ActionType.MOUSE_DOUBLE_CLICK: ActionCard(
        ActionType.MOUSE_DOUBLE_CLICK,
        "Double-click at a coordinate.",
        {"x": "horizontal pixel", "y": "vertical pixel"},
        {"x": 500, "y": 300},
    ),
    ActionType.MOUSE_RIGHT_CLICK: ActionCard(
        ActionType.MOUSE_RIGHT_CLICK,
        "Right-click at a coordinate.",
        {"x": "horizontal pixel", "y": "vertical pixel"},
        {"x": 500, "y": 300},
    ),
    ActionType.MOUSE_DRAG: ActionCard(
        ActionType.MOUSE_DRAG,
        "Drag from one coordinate to another.",
        {"start_x": "drag start", "start_y": "", "end_x": "drag end", "end_y": ""},
        {"start_x": 100, "start_y": 200, "end_x": 300, "end_y": 200},
    ),
    ActionType.MOUSE_SCROLL: ActionCard(
        ActionType.MOUSE_SCROLL,
        "Scroll at a position.",
        {
            "x": "horizontal pixel (default 640)",
            "y": "vertical pixel (default 360)",
            "delta_y": "scroll amount, negative = down",
        },
        {"delta_y": -3},
    ),
    ActionType.KEYBOARD_TYPE: ActionCard(
        ActionType.KEYBOARD_TYPE,
        "Type text into the focused field.",
        {"text": "text to type"},
        {"text": "hello"},
    ),
    ActionType.KEYBOARD_PRESS: ActionCard(
        ActionType.KEYBOARD_PRESS,
        "Press a key or combo.",
        {"key": "e.g. Enter, Escape, Control+a"},
        {"key": "Enter"},
    ),
    ActionType.KEYBOARD_HOTKEY: ActionCard(
        ActionType.KEYBOARD_HOTKEY,
        "Hold modifiers and press the last key.",
        {"keys": "list, e.g. [Control, Shift, p]"},
        {"keys": ["Control", "s"]},
    ),
    ActionType.SCREENSHOT: ActionCard(
        ActionType.SCREENSHOT, "Capture the screen for observation.", {}, {}
    ),
    ActionType.WAIT: ActionCard(
        ActionType.WAIT,
        "Pause before the next step.",
        {"ms": "milliseconds (default 1000)"},
        {"ms": 500},
    ),
    ActionType.NAVIGATE: ActionCard(
        ActionType.NAVIGATE,
        "Go to a URL.",
        {"url": "destination URL"},
        {"url": "http://localhost:3000/chat"},
    ),
    ActionType.TOOL_CALL: ActionCard(
        ActionType.TOOL_CALL,
        "Call one outside tool from the provided registry.",
        {"tool": "tool name", "args": "dict of arguments (optional)"},
        {"tool": "file_search", "args": {"query": "notes"}},
    ),
    ActionType.DONE: ActionCard(
        ActionType.DONE, "Task completed. Ends the run successfully.", {}, {}
    ),
    ActionType.FAIL: ActionCard(
        ActionType.FAIL, "Task cannot be completed. Ends the run as failed.", {}, {}
    ),
}

_REQUIRED_PARAMS: dict[ActionType, list[str]] = {
    ActionType.MOUSE_MOVE: ["x", "y"],
    ActionType.MOUSE_CLICK: ["x", "y"],
    ActionType.MOUSE_DOUBLE_CLICK: ["x", "y"],
    ActionType.MOUSE_RIGHT_CLICK: ["x", "y"],
    ActionType.MOUSE_DRAG: ["start_x", "start_y", "end_x", "end_y"],
    ActionType.KEYBOARD_TYPE: ["text"],
    ActionType.KEYBOARD_PRESS: ["key"],
    ActionType.KEYBOARD_HOTKEY: ["keys"],
    ActionType.NAVIGATE: ["url"],
    ActionType.TOOL_CALL: ["tool"],
}


def validate_action(action: Action) -> str | None:
    """Check an action against its card. Returns error text or None."""
    card = ACTION_CARDS.get(action.action_type)
    if card is None:
        return f"unknown action: {action.action_type}"
    missing = [p for p in _REQUIRED_PARAMS.get(action.action_type, []) if p not in action.params]
    if missing:
        return f"{action.action_type.value} missing params: {', '.join(missing)}"
    if action.action_type in (
        ActionType.MOUSE_MOVE,
        ActionType.MOUSE_CLICK,
        ActionType.MOUSE_DOUBLE_CLICK,
        ActionType.MOUSE_RIGHT_CLICK,
    ):
        for axis in ("x", "y"):
            if not isinstance(action.params[axis], (int, float)):
                return f"{axis} must be a number"
    if action.action_type == ActionType.KEYBOARD_HOTKEY:
        if not isinstance(action.params["keys"], list) or not action.params["keys"]:
            return "keys must be a non-empty list"
    if action.action_type == ActionType.NAVIGATE:
        if not str(action.params["url"]).strip():
            return "url must not be empty"
    return None


def tool_cards_text() -> str:
    """Human-readable tool documentation for model prompts."""
    lines = []
    for at in ActionType:
        card = ACTION_CARDS[at]
        params = ", ".join(f"{k}: {v}" for k, v in card.params.items()) or "none"
        lines.append(f"- {at.value}: {card.description} Params({params})")
    return "\n".join(lines)


# ── Model interface ─────────────────────────────────────────────────────


@runtime_checkable
class VisionModel(Protocol):
    """Decides computer actions from screen state.

    Implementations can wrap local models, API models, or custom logic.
    """

    @property
    def model_name(self) -> str: ...

    @property
    def supports_vision(self) -> bool: ...

    async def predict_action(
        self,
        task: str,
        screenshot_b64: str,
        accessibility_tree: dict[str, Any] | None = None,
        history: list[Step] | None = None,
        available_actions: list[ActionType] | None = None,
    ) -> Action:
        """Given current state, predict the next action to take."""
        ...

    async def predict_actions_batch(
        self,
        task: str,
        screenshots_b64: list[str],
    ) -> list[Action]:
        """Batch prediction for multiple screenshots."""
        ...


@runtime_checkable
class RewardModel(Protocol):
    """Evaluates action quality."""

    async def evaluate(
        self,
        task: str,
        step: Step,
        goal_state: dict[str, Any] | None = None,
    ) -> float:
        """Return reward for a step (higher = better)."""
        ...


# ── Concrete implementations ────────────────────────────────────────────


class RuleBasedModel:
    """Fixed policy baseline: click the first clickable element found."""

    def __init__(self):
        self._name = "rule_based"

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def supports_vision(self) -> bool:
        return False

    async def predict_action(
        self,
        task: str,
        screenshot_b64: str,
        accessibility_tree: dict[str, Any] | None = None,
        history: list[Step] | None = None,
        available_actions: list[ActionType] | None = None,
    ) -> Action:
        if accessibility_tree:
            element = _find_clickable(accessibility_tree)
            if element:
                bbox = element.get("bbox", {})
                if bbox:
                    return Action(
                        action_type=ActionType.MOUSE_CLICK,
                        params={"x": bbox.get("x", 0), "y": bbox.get("y", 0)},
                        reasoning=f"Rule-based: clicked {element.get('role', 'element')}",
                    )
        return Action(
            action_type=ActionType.SCREENSHOT,
            reasoning="Rule-based: taking screenshot to observe",
        )


class EchoModel:
    """Replays a pre-configured action sequence, then DONE. For tests."""

    def __init__(self, actions: list[Action] | None = None):
        self._actions = list(actions or [])
        self._index = 0

    @property
    def model_name(self) -> str:
        return "echo"

    @property
    def supports_vision(self) -> bool:
        return True

    async def predict_action(
        self,
        task: str,
        screenshot_b64: str,
        accessibility_tree: dict[str, Any] | None = None,
        history: list[Step] | None = None,
        available_actions: list[ActionType] | None = None,
    ) -> Action:
        if self._index < len(self._actions):
            action = self._actions[self._index]
            self._index += 1
            return action
        return Action(
            action_type=ActionType.DONE,
            reasoning="EchoModel: no more actions",
        )


class CompositeModel:
    """Delegates to sub-models by context, first matching condition wins."""

    def __init__(self):
        self._models: list[tuple[str, VisionModel, Any]] = []

    @property
    def model_name(self) -> str:
        return "composite"

    @property
    def supports_vision(self) -> bool:
        return any(m.supports_vision for _, m, _ in self._models)

    def add_model(self, name: str, model: VisionModel, condition=None) -> CompositeModel:
        self._models.append((name, model, condition or (lambda ctx: True)))
        return self

    async def predict_action(
        self,
        task: str,
        screenshot_b64: str,
        accessibility_tree: dict[str, Any] | None = None,
        history: list[Step] | None = None,
        available_actions: list[ActionType] | None = None,
    ) -> Action:
        context = {
            "task": task,
            "screenshot_b64": screenshot_b64,
            "accessibility_tree": accessibility_tree,
            "history": history or [],
            "available_actions": available_actions,
        }
        for name, model, condition in self._models:
            if condition(context):
                return await model.predict_action(
                    task, screenshot_b64, accessibility_tree, history, available_actions
                )
        return Action(action_type=ActionType.DONE, reasoning="CompositeModel: no model matched")


# ── Helpers ─────────────────────────────────────────────────────────────


def _find_clickable(tree: dict[str, Any]) -> dict[str, Any] | None:
    """First clickable element in an accessibility tree."""
    if not isinstance(tree, dict):
        return None
    role = tree.get("role", "")
    if role in ("button", "link", "tab", "menuitem", "option"):
        return tree
    for child in tree.get("children", []):
        result = _find_clickable(child)
        if result:
            return result
    return None
