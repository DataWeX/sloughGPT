"""
planner.engine — Kanban state machine and transition engine.

Pure domain logic: no HTTP, no storage. Validates and applies card
transitions with WIP enforcement and event emission.

Canonical states and legal transitions are based on the Kanban Method
research: inbox → ready → doing → review → verify → done, with blocked
as a lateral state accessible from any active state.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Callable

# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------


class CardState(str, Enum):
    """Canonical kanban card states."""

    INBOX = "inbox"
    READY = "ready"
    DOING = "doing"
    REVIEW = "review"
    VERIFY = "verify"
    DONE = "done"
    BLOCKED = "blocked"


# Legal transitions: from_state → set of allowed to_states
LEGAL_TRANSITIONS: dict[CardState, set[CardState]] = {
    CardState.INBOX: {CardState.READY},
    CardState.READY: {CardState.DOING, CardState.BLOCKED},
    CardState.DOING: {CardState.REVIEW, CardState.BLOCKED, CardState.READY},
    CardState.REVIEW: {CardState.VERIFY, CardState.DOING, CardState.BLOCKED},
    CardState.VERIFY: {CardState.DONE, CardState.REVIEW, CardState.BLOCKED},
    CardState.DONE: set(),  # terminal
    CardState.BLOCKED: {CardState.INBOX, CardState.READY, CardState.DOING},
}

# Terminal states — no transitions out
TERMINAL_STATES: set[CardState] = {CardState.DONE}

# States that count toward WIP (active work, not backlog or done)
ACTIVE_STATES: set[CardState] = {
    CardState.READY,
    CardState.DOING,
    CardState.REVIEW,
    CardState.VERIFY,
}


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    CARD_MOVED = "card_moved"
    CARD_BLOCKED = "card_blocked"
    CARD_UNBLOCKED = "card_unblocked"
    WIP_EXCEEDED = "wip_exceeded"
    WIP_WARNED = "wip_warned"
    TRANSITION_REJECTED = "transition_rejected"
    CARD_CREATED = "card_created"
    CARD_ARCHIVED = "card_archived"


@dataclass
class KanbanEvent:
    """Emitted on every state change or rejection."""

    type: EventType
    card_id: str
    from_state: str = ""
    to_state: str = ""
    actor: str = ""
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "card_id": self.card_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "actor": self.actor,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class TransitionResult:
    """Outcome of a transition attempt."""

    ok: bool
    card_id: str = ""
    from_state: str = ""
    to_state: str = ""
    error: str = ""
    event: KanbanEvent | None = None
    wip_count: int = 0
    wip_limit: int = 0

    @property
    def wip_exceeded(self) -> bool:
        return self.wip_limit > 0 and self.wip_count > self.wip_limit


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class KanbanEngine:
    """State machine that validates and applies card transitions.

    Thread-safe. Emits events via a callback for observability.
    """

    def __init__(self, on_event: Callable[[KanbanEvent], None] | None = None):
        self._lock = threading.Lock()
        self._on_event = on_event
        self._event_log: list[KanbanEvent] = []

    # -- Event handling ----------------------------------------------------

    def _emit(self, event: KanbanEvent) -> None:
        if not event.timestamp:
            event.timestamp = datetime.now(UTC).isoformat()
        self._event_log.append(event)
        if self._on_event:
            self._on_event(event)

    @property
    def event_log(self) -> list[KanbanEvent]:
        return list(self._event_log)

    # -- Transition validation --------------------------------------------

    @staticmethod
    def is_legal(from_state: str | CardState, to_state: str | CardState) -> bool:
        """Check if a transition is legal without applying it."""
        try:
            src = CardState(from_state)
            dst = CardState(to_state)
        except ValueError:
            return False
        return dst in LEGAL_TRANSITIONS.get(src, set())

    @staticmethod
    def can_transition(card_id: str, from_state: str, to_state: str) -> bool:
        """Alias for is_legal with explicit card context."""
        return KanbanEngine.is_legal(from_state, to_state)

    # -- WIP checking -------------------------------------------------------

    @staticmethod
    def check_wip(
        board_columns: list[dict[str, Any]],
        board_cards: list[dict[str, Any]],
        to_state: str,
        exclude_card_id: str = "",
    ) -> tuple[int, int]:
        """Check WIP count and limit for a target state.

        Returns (current_count_after_move, wip_limit).
        """
        target = to_state
        limit = 0
        for col in board_columns:
            if col.get("name") == target:
                limit = col.get("wip_limit", 0)
                break

        count = sum(
            1 for c in board_cards if c.get("column") == target and c.get("id") != exclude_card_id
        )
        return count + 1, limit

    # -- Transition application ---------------------------------------------

    def transition(
        self,
        board_columns: list[dict[str, Any]],
        board_cards: list[dict[str, Any]],
        card_id: str,
        to_state: str,
        actor: str = "",
        force: bool = False,
    ) -> TransitionResult:
        """Apply a card transition. Returns TransitionResult.

        Args:
            board_columns: Column definitions (name, wip_limit, order).
            board_cards: All cards on the board.
            card_id: ID of the card to move.
            to_state: Target state (must be a valid CardState).
            actor: Who initiated the move (for event log).
            force: Skip WIP limit check (admin override).

        Raises nothing — all errors returned in TransitionResult.
        """
        with self._lock:
            # Find the card
            card = next((c for c in board_cards if c.get("id") == card_id), None)
            if card is None:
                result = TransitionResult(
                    ok=False,
                    card_id=card_id,
                    error=f"Card '{card_id}' not found",
                )
                self._emit(
                    KanbanEvent(
                        type=EventType.TRANSITION_REJECTED,
                        card_id=card_id,
                        metadata={"reason": result.error},
                    )
                )
                return result

            from_state = card.get("column", "")
            result = TransitionResult(
                ok=True,
                card_id=card_id,
                from_state=from_state,
                to_state=to_state,
            )

            # Idempotent: same state → no-op
            if from_state == to_state:
                result.event = KanbanEvent(
                    type=EventType.CARD_MOVED,
                    card_id=card_id,
                    from_state=from_state,
                    to_state=to_state,
                    actor=actor,
                    metadata={"noop": True},
                )
                self._emit(result.event)
                return result

            # Validate target state
            try:
                CardState(to_state)
            except ValueError:
                result.ok = False
                result.error = f"Unknown state '{to_state}'"
                result.event = KanbanEvent(
                    type=EventType.TRANSITION_REJECTED,
                    card_id=card_id,
                    from_state=from_state,
                    to_state=to_state,
                    actor=actor,
                    metadata={"reason": result.error},
                )
                self._emit(result.event)
                return result

            # Validate legal transition
            if not self.is_legal(from_state, to_state):
                result.ok = False
                legal = LEGAL_TRANSITIONS.get(CardState(from_state), set())
                result.error = (
                    f"Illegal transition: {from_state} → {to_state}. "
                    f"Legal: {[s.value for s in legal]}"
                )
                result.event = KanbanEvent(
                    type=EventType.TRANSITION_REJECTED,
                    card_id=card_id,
                    from_state=from_state,
                    to_state=to_state,
                    actor=actor,
                    metadata={"reason": result.error, "legal_targets": [s.value for s in legal]},
                )
                self._emit(result.event)
                return result

            # WIP enforcement (skip if force or target has no limit)
            if not force:
                wip_count, wip_limit = self.check_wip(
                    board_columns,
                    board_cards,
                    to_state,
                    exclude_card_id=card_id,
                )
                result.wip_count = wip_count
                result.wip_limit = wip_limit

                if wip_limit > 0 and wip_count > wip_limit:
                    result.ok = False
                    result.error = f"WIP limit exceeded for '{to_state}': {wip_count}/{wip_limit}"
                    result.event = KanbanEvent(
                        type=EventType.WIP_EXCEEDED,
                        card_id=card_id,
                        from_state=from_state,
                        to_state=to_state,
                        actor=actor,
                        metadata={
                            "wip_count": wip_count,
                            "wip_limit": wip_limit,
                            "reason": result.error,
                        },
                    )
                    self._emit(result.event)
                    return result

            # Apply the transition
            card["column"] = to_state
            card["updated_at"] = datetime.now(UTC).isoformat()

            # Emit appropriate event
            if to_state == CardState.BLOCKED.value:
                event_type = EventType.CARD_BLOCKED
            elif from_state == CardState.BLOCKED.value:
                event_type = EventType.CARD_UNBLOCKED
            else:
                event_type = EventType.CARD_MOVED

            result.event = KanbanEvent(
                type=event_type,
                card_id=card_id,
                from_state=from_state,
                to_state=to_state,
                actor=actor,
            )
            self._emit(result.event)
            return result

    # -- Utility ------------------------------------------------------------

    @staticmethod
    def next_states(from_state: str) -> list[str]:
        """Return list of legal next states from a given state."""
        try:
            src = CardState(from_state)
        except ValueError:
            return []
        return [s.value for s in LEGAL_TRANSITIONS.get(src, set())]

    @staticmethod
    def is_terminal(state: str) -> bool:
        try:
            return CardState(state) in TERMINAL_STATES
        except ValueError:
            return False

    @staticmethod
    def is_active(state: str) -> bool:
        try:
            return CardState(state) in ACTIVE_STATES
        except ValueError:
            return False
