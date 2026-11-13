"""Command inbox — direct a running agent from outside its loop.

The single-writer rule: the agent loop is the only writer of its own
state. Consoles (or scripts, or chat) never mutate the loop — they
submit typed :class:`Command` values to a :class:`CommandInbox`, and the
loop **drains the inbox at step boundaries**, applying each command to
its :class:`ControlState` (instructions/messages/hold flags) and to its
``AgentConfig`` (params, type-checked). That is what makes live
direction safe: a console can die mid-run, spam commands, or send
garbage, and the run either ignores it (validation) or applies it at a
well-defined point (boundary) — never in the middle of an action.

Pause holds the loop at a boundary and stops the run's deadline clock
(see ``Agent._control_boundary``): being paused must never age a live
run into ``stop_reason "deadline"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CommandKind(Enum):
    """Everything an operator may tell a running agent."""

    SET_INSTRUCTIONS = "set_instructions"
    SET_PARAMS = "set_params"
    PAUSE = "pause"
    RESUME = "resume"
    INJECT_MESSAGE = "inject_message"
    STOP = "stop"


_TEXT_KINDS = frozenset({CommandKind.SET_INSTRUCTIONS, CommandKind.INJECT_MESSAGE})


@dataclass(frozen=True)
class Command:
    """One typed directive. Validated on entry, applied at a boundary."""

    kind: CommandKind
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> Command:
        """Build a command from wire JSON. Raises ValueError when malformed."""
        if not isinstance(data, dict):
            raise ValueError("command must be a JSON object")
        raw = data.get("kind")
        if raw is None:
            raise ValueError("command needs a 'kind'")
        try:
            kind = CommandKind(str(raw))
        except ValueError:
            raise ValueError(f"unknown command kind: {raw!r}") from None
        payload = data.get("payload", {})
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
        if kind in _TEXT_KINDS:
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{kind.value} needs a non-empty text payload")
        return cls(kind=kind, payload=dict(payload))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "payload": self.payload}


@dataclass
class ControlState:
    """What the loop reads at step boundaries — written only by drain()."""

    instructions: str = ""
    messages: list[str] = field(default_factory=list)
    paused: bool = False
    stopped: bool = False
    #: injected messages kept for the prompt; oldest dropped past this
    message_limit: int = 20

    def effective_task(self, base: str) -> str:
        """The task text the model actually sees this step."""
        if not self.instructions and not self.messages:
            return base
        parts = [base]
        if self.instructions:
            parts.append(f"Updated instructions from the operator: {self.instructions}")
        if self.messages:
            parts.append(
                "Messages from the operator:\n" + "\n".join(f"- {m}" for m in self.messages)
            )
        return "\n\n".join(parts)


def _validate_param(current: Any, value: Any) -> tuple[bool, Any, str]:
    """Type-check one param against the current value. Returns (ok, value, why)."""
    if isinstance(current, bool):
        if not isinstance(value, bool):
            return False, value, "expected bool"
        return True, value, ""
    if isinstance(current, int):
        if isinstance(value, bool) or not isinstance(value, int):
            return False, value, "expected int"
        return True, value, ""
    if isinstance(current, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False, value, "expected number"
        return True, float(value), ""
    if isinstance(current, str):
        if not isinstance(value, str):
            return False, value, "expected string"
        return True, value, ""
    if current is None or isinstance(current, (list, dict)):
        if type(value) is not type(current):  # noqa: E721 — exact shape or reject
            return False, value, f"expected {type(current).__name__}"
        return True, value, ""
    if not isinstance(value, type(current)):
        return False, value, f"expected {type(current).__name__}"
    return True, value, ""


class CommandInbox:
    """FIFO of pending commands; drained by the agent loop only.

    ``submit`` is safe from any coroutine; ``drain`` runs on the loop at
    a step boundary and is the only place commands touch state.
    """

    def __init__(self) -> None:
        self._pending: list[Command] = []

    @property
    def pending(self) -> int:
        return len(self._pending)

    def submit(self, command: Command) -> None:
        self._pending.append(command)

    def submit_dict(self, data: Any) -> Command:
        """Validate wire JSON and queue it. Raises ValueError when malformed."""
        command = Command.from_dict(data)
        self.submit(command)
        return command

    def drain(self, control: ControlState, config: Any) -> list[dict[str, Any]]:
        """Apply every pending command in order; returns log records."""
        pending, self._pending = self._pending, []
        return [self._apply(command, control, config) for command in pending]

    # ── application (state changes happen here, and only here) ──────────

    @staticmethod
    def _apply(command: Command, control: ControlState, config: Any) -> dict[str, Any]:
        kind, payload = command.kind, command.payload

        if kind is CommandKind.SET_INSTRUCTIONS:
            control.instructions = payload["text"].strip()
            note = "instructions updated"
        elif kind is CommandKind.INJECT_MESSAGE:
            control.messages.append(payload["text"].strip())
            overflow = len(control.messages) - control.message_limit
            if overflow > 0:
                del control.messages[:overflow]
            note = f"message queued ({len(control.messages)} in effect)"
        elif kind is CommandKind.PAUSE:
            control.paused = True
            note = "paused at step boundary"
        elif kind is CommandKind.RESUME:
            control.paused = False
            note = "resumed"
        elif kind is CommandKind.STOP:
            control.stopped = True
            control.paused = False  # a stop always releases a hold
            note = "stop requested"
        else:  # SET_PARAMS
            applied: dict[str, Any] = {}
            rejected: list[str] = []
            for key, value in payload.items():
                if key.startswith("_") or not hasattr(config, key):
                    rejected.append(f"{key}: unknown parameter")
                    continue
                ok, coerced, why = _validate_param(getattr(config, key), value)
                if not ok:
                    rejected.append(f"{key}: {why}")
                    continue
                setattr(config, key, coerced)
                applied[key] = coerced
            note = (
                "applied " + ", ".join(f"{k}={v!r}" for k, v in applied.items())
                if applied
                else "no params applied"
            )
            if rejected:
                note += f" — rejected: {'; '.join(rejected)}"
            return {
                "kind": kind.value,
                "ok": not rejected,
                "note": note,
                "payload": {"applied": applied, "rejected": rejected},
            }

        return {"kind": kind.value, "ok": True, "note": note, "payload": payload}
